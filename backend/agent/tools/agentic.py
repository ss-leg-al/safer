"""All agent-callable tools, decorated with @tool for LangGraph supervisor."""

from __future__ import annotations

import base64
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from langchain_core.tools import tool
from openai import OpenAI

from ...config import settings
from ...models.sam3_loader import get_sam3_processor, is_available as sam3_available
from ..job_store import get_store
from ..log_emitter import emit_log

PII_TYPES = {"face", "document", "screen", "nameplate", "id_card"}
MASK_STRATEGY = {
    "face": "blur",
    "document": "blackbox",
    "screen": "pixelate",
    "nameplate": "blackbox",
    "id_card": "blackbox",
}
# SAM3 receives natural-language phrases, not internal type names.
TEXT_PROMPT_MAP = {
    "face": "human face",
    "document": "paper document",
    "screen": "computer screen or monitor",
    "nameplate": "name badge",
    "id_card": "id card",
}


def _client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _bbox_from_polygon(polygon: np.ndarray) -> list[int]:
    """Compute axis-aligned bbox [x, y, w, h] from polygon points."""
    if polygon is None or len(polygon) == 0:
        return [0, 0, 0, 0]
    pts = np.asarray(polygon).reshape(-1, 2)
    x1, y1 = pts.min(axis=0).astype(int).tolist()
    x2, y2 = pts.max(axis=0).astype(int).tolist()
    return [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]


def _apply_polygon_mask(img: np.ndarray, polygon, strategy: str) -> np.ndarray:
    """Apply masking strategy within an arbitrary polygon region (pixel-precise)."""
    if polygon is None or len(polygon) < 3:
        return img
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    region = mask == 255
    if not region.any():
        return img

    if strategy == "blur":
        blurred = cv2.GaussianBlur(img, (51, 51), 15)
        img[region] = blurred[region]
    elif strategy == "blackbox":
        img[region] = 0
    elif strategy == "pixelate":
        ys, xs = np.where(region)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return img
        rh, rw = roi.shape[:2]
        block = max(2, min(rw, rh) // 12 or 2)
        small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        sub_mask = mask[y1:y2, x1:x2] == 255
        roi_out = roi.copy()
        roi_out[sub_mask] = pixelated[sub_mask]
        img[y1:y2, x1:x2] = roi_out
    return img


def _binary_mask_to_polygon(binary_mask: np.ndarray) -> list[list[int]] | None:
    """Convert a 2D binary mask (uint8 0/255 or bool) to its largest contour polygon."""
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1:
        return None
    return largest.reshape(-1, 2).tolist()


def _sam3_detect(image_path: str, target_types: list[str], conf_threshold: float = 0.3) -> list[dict]:
    """Run Meta SAM3 with text prompts on one image. Returns objects list with polygons.

    SAM3 takes ONE text prompt at a time, so we call once per requested type and merge.
    Each detection becomes one object with id, type, bbox(xywh), polygon, confidence.
    """
    import torch
    from PIL import Image

    processor = get_sam3_processor()
    image = Image.open(image_path).convert("RGB")

    out: list[dict] = []
    # autocast must be in this thread's call stack — uvicorn workers don't inherit
    # the one entered in startup.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for t in target_types:
            prompt = TEXT_PROMPT_MAP.get(t, t)
            result = processor.set_text_prompt(state=state, prompt=prompt)
            masks = result.get("masks")
            boxes = result.get("boxes")
            scores = result.get("scores")
            if masks is None or boxes is None or scores is None:
                continue
            masks_np = masks.detach().cpu().float().numpy()  # [N, 1, H, W]
            boxes_np = boxes.detach().cpu().float().numpy()  # [N, 4] xyxy
            scores_np = scores.detach().cpu().float().numpy()  # [N]
            for i in range(scores_np.shape[0]):
                conf = float(scores_np[i])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = boxes_np[i].tolist()
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                mask_2d = masks_np[i, 0] > 0.5
                polygon = _binary_mask_to_polygon(mask_2d.astype(np.uint8) * 255)
                out.append(
                    {
                        "id": len(out),
                        "type": t,
                        "bbox": [x1, y1, w, h],
                        "polygon": polygon,
                        "confidence": conf,
                    }
                )
    return out


@tool
def extract_frames(job_id: str) -> str:
    """Extract frames from the input video at 1 fps. MUST be called first, before any other tool."""
    store = get_store(job_id)
    if not store.video_path:
        return "ERROR: video_path is missing in store. Cannot extract frames."

    out_dir = settings.upload_path / job_id / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-i", store.video_path,
            "-vf", f"fps={settings.SAMPLE_FPS}",
            str(out_dir / "%04d.jpg"),
            "-y", "-loglevel", "error",
        ],
        check=True,
    )
    frames = sorted(out_dir.glob("*.jpg"))
    store.frames_dir = str(out_dir)
    if frames:
        store.sample_frame = str(frames[len(frames) // 10])
    emit_log(job_id, {"step": "tool", "action": "extract_frames", "result": {"count": len(frames)}})
    return f"Extracted {len(frames)} frames to {out_dir}. Sample frame ready for analysis."


@tool
def analyze_scene(job_id: str) -> str:
    """Use GPT-4o Vision to classify the scene type and recommend likely PII categories.
    Call AFTER extract_frames, BEFORE detect_pii. Returns scene_type and a list of expected PII types
    (subset of: face, document, screen, nameplate, id_card)."""
    store = get_store(job_id)
    if not store.frames_dir:
        return "ERROR: no frames available. Call extract_frames first."

    frames = sorted(Path(store.frames_dir).glob("*.jpg"))
    if not frames:
        return "ERROR: no frames available. Call extract_frames first."

    # 첫 호출: 10% 지점 프레임, 재시도: 50% 지점 프레임 (다른 장면 샘플링)
    store.scene_analyze_count += 1
    idx = len(frames) // 10 if store.scene_analyze_count == 1 else len(frames) // 2
    frame_path = frames[idx]

    b64 = base64.b64encode(frame_path.read_bytes()).decode()
    prompt = (
        "Classify this video frame.\n"
        "Respond ONLY in JSON: "
        '{"scene_type": "meeting|lecture|interview|public|other", '
        '"expected_pii": ["face", "document", "screen", "nameplate", "id_card"], '
        '"reasoning": "..."}'
    )
    resp = _client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    if not raw:
        store.scene_type = "other"
        store.expected_pii = []
        emit_log(job_id, {"step": "tool", "action": "analyze_scene", "result": {"scene_type": "other", "expected_pii": []}})
        return "scene_type=other, expected_pii=[]. Reasoning: GPT-4o returned empty content."

    parsed = json.loads(raw)
    store.scene_type = parsed.get("scene_type", "other")
    store.expected_pii = [p for p in parsed.get("expected_pii", []) if p in PII_TYPES]

    emit_log(job_id, {"step": "tool", "action": "analyze_scene", "result": parsed})
    return (
        f"scene_type={store.scene_type}, expected_pii={store.expected_pii}. "
        f"Reasoning: {parsed.get('reasoning', '')}"
    )


@tool
def detect_pii(job_id: str, target_types: list[str]) -> str:
    """Detect PII objects in the FIRST frame using SAM 3.1 with text prompts.
    target_types: subset of [face, document, screen, nameplate, id_card]. Each is
    converted to a natural-language phrase (e.g. 'face' -> 'human face') for SAM3.
    PREFERRED: pass ALL target types from analyze_scene's expected_pii in a SINGLE call.
    Same-type re-calls overwrite that type; different-type calls accumulate.
    Returns count, by-type breakdown, and confidence summary."""
    store = get_store(job_id)
    if not store.frames_dir:
        return "ERROR: frames not extracted yet."
    if not sam3_available():
        return "ERROR: SAM3 not loaded. Check /health for sam3_error."

    target_types = [t for t in target_types if t in PII_TYPES] or ["face"]
    frames = sorted(Path(store.frames_dir).glob("*.jpg"))
    first = frames[0]
    text_prompts = [TEXT_PROMPT_MAP[t] for t in target_types]

    new_objects = _sam3_detect(str(first), target_types, conf_threshold=0.3)

    store.detect_attempts += 1
    kept = [o for o in store.detected_objects if o["type"] not in target_types]
    next_id = max((o["id"] for o in kept), default=-1) + 1
    for i, o in enumerate(new_objects):
        o["id"] = next_id + i
    store.detected_objects = kept + new_objects

    all_objs = store.detected_objects
    by_type = Counter(o["type"] for o in all_objs)
    new_by_type = Counter(o["type"] for o in new_objects)
    avg_conf = sum(o["confidence"] for o in all_objs) / len(all_objs) if all_objs else 0.0
    summary = {
        "attempt": store.detect_attempts,
        "this_call_added": dict(new_by_type),
        "total_after_call": len(all_objs),
        "by_type_total": dict(by_type),
        "text_prompts_used": text_prompts,
        "avg_confidence_total": round(avg_conf, 3),
        "min_confidence_this_call": round(min((o["confidence"] for o in new_objects), default=0.0), 3),
    }
    emit_log(job_id, {"step": "tool", "action": "detect_pii", "result": summary})
    return (
        f"SAM3 with prompts {text_prompts}: this call added {dict(new_by_type)}. "
        f"Total in store: {len(all_objs)} ({dict(by_type)}). "
        f"Avg confidence: {avg_conf:.2f}. Min this call: {summary['min_confidence_this_call']:.2f}. "
        f"Retry threshold: {settings.CONFIDENCE_THRESHOLD}."
    )


@tool
def track_objects(job_id: str) -> str:
    """Run SAM 3.1 on EVERY extracted frame with the same text prompts as detect_pii,
    producing pixel-precise polygon masks per frame. (No CSRT — SAM3 is fast enough
    on GPU and per-frame zero-shot detection avoids drift.) Per-frame masks are
    stored in store.per_frame_bboxes (each entry has bbox + polygon + type)."""
    store = get_store(job_id)
    frames = sorted(Path(store.frames_dir).glob("*.jpg")) if store.frames_dir else []
    if not frames or not store.detected_objects:
        store.per_frame_bboxes = {}
        emit_log(job_id, {"step": "tool", "action": "track_objects", "result": "skipped"})
        return "Skipped: no frames or no detected objects."
    if not sam3_available():
        return "ERROR: SAM3 not loaded."

    types_present = sorted({o["type"] for o in store.detected_objects})
    text_prompts = [TEXT_PROMPT_MAP[t] for t in types_present]

    per_frame: dict[str, list[dict]] = {}
    total_masks = 0
    for fp in frames:
        objs = _sam3_detect(str(fp), types_present, conf_threshold=0.3)
        per_frame[fp.name] = objs
        total_masks += len(objs)

    store.per_frame_bboxes = per_frame
    emit_log(
        job_id,
        {
            "step": "tool", "action": "track_objects",
            "result": {
                "mode": "sam3_per_frame",
                "types": types_present,
                "frames": len(frames),
                "total_masks": total_masks,
                "avg_masks_per_frame": round(total_masks / len(frames), 2),
            },
        },
    )
    return (
        f"SAM3 ran on {len(frames)} frames with prompts {text_prompts}. "
        f"Generated {total_masks} polygon masks total "
        f"(avg {total_masks/len(frames):.1f}/frame)."
    )


@tool
def mask_frames(job_id: str) -> str:
    """Apply per-type masking (face→blur, document/nameplate/id_card→blackbox, screen→pixelate)
    to every frame. Uses pixel-precise polygons from SAM3 (track_objects) when available,
    otherwise falls back to bbox rectangles. Call AFTER track_objects."""
    store = get_store(job_id)
    frames = sorted(Path(store.frames_dir).glob("*.jpg")) if store.frames_dir else []
    out_dir = settings.upload_path / job_id / "masked_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    masked_regions = 0
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        for entry in store.per_frame_bboxes.get(fp.name, []):
            strategy = MASK_STRATEGY.get(entry.get("type", "face"), "blur")
            polygon = entry.get("polygon")
            if polygon and len(polygon) >= 3:
                img = _apply_polygon_mask(img, polygon, strategy)
            else:
                # Fallback to rectangle from bbox
                bbox = entry.get("bbox", [0, 0, 0, 0])
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    pseudo_polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    img = _apply_polygon_mask(img, pseudo_polygon, strategy)
            masked_regions += 1
        cv2.imwrite(str(out_dir / fp.name), img)

    store.masked_frames_dir = str(out_dir)
    for o in store.detected_objects:
        o["mask_strategy"] = MASK_STRATEGY.get(o["type"], "blur")
    emit_log(
        job_id,
        {"step": "tool", "action": "mask_frames",
         "result": {"frames": len(frames), "regions_masked": masked_regions}},
    )
    return f"Masked {len(frames)} frames ({masked_regions} polygon regions total)."


@tool
def compose_video(job_id: str) -> str:
    """Stitch masked frames into mp4, or copy original video if no masking was done.
    Call AFTER mask_frames (if masking was done) or directly after analyze_scene (if no PII)."""
    import shutil
    store = get_store(job_id)
    out_dir = settings.output_path / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.mp4"

    # PII 없어서 마스킹 안 한 경우 → 원본 영상 복사 (품질 손실 없음)
    if not store.masked_frames_dir and store.video_path:
        shutil.copy2(store.video_path, out_path)
        store.output_video_path = str(out_path)
        emit_log(job_id, {"step": "tool", "action": "compose_video", "result": {"path": str(out_path), "mode": "copy"}})
        return f"No masking applied — copied original video to {out_path}."

    if not store.masked_frames_dir:
        return "ERROR: masked frames not ready. Call mask_frames first."

    subprocess.run(
        [
            "ffmpeg", "-framerate", str(settings.SAMPLE_FPS),
            "-i", str(Path(store.masked_frames_dir) / "%04d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out_path),
            "-y", "-loglevel", "error",
        ],
        check=True,
    )
    store.output_video_path = str(out_path)
    emit_log(job_id, {"step": "tool", "action": "compose_video", "result": {"path": str(out_path), "mode": "encode"}})
    return f"Composed masked mp4 at {out_path}."


def _build_report_pdf(pdf_path: Path, report: dict, masked_frame_path: Path | None) -> None:
    """Build a one-page PDF report with summary table + thumbnail of first masked frame."""
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=12
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6
    )

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Privacy Guard Report",
    )
    story = []

    story.append(Paragraph("Privacy Guard Report", title_style))
    story.append(Paragraph(
        f"job_id: {report['job_id']} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        subtitle_style,
    ))

    # Summary table
    story.append(Paragraph("Summary", section_style))
    summary_rows = [
        ["Field", "Value"],
        ["Scene Type", str(report.get("scene_type") or "-")],
        ["Expected PII", ", ".join(report.get("expected_pii", []) or ["-"])],
        ["Total Detected Objects", str(report.get("total_objects", 0))],
        ["Detect Attempts", str(report.get("detect_attempts", 0))],
        ["Output Video", str(Path(report.get("output_video_path") or "").name or "-")],
    ]
    summary = Table(summary_rows, colWidths=[55 * mm, 110 * mm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(summary)

    # By-type breakdown
    story.append(Paragraph("By Type", section_style))
    by_type_rows = [["PII Type", "Count", "Mask Strategy"]]
    strategy_map = {"face": "blur", "document": "blackbox", "screen": "pixelate",
                    "nameplate": "blackbox", "id_card": "blackbox"}
    by_type = report.get("by_type", {}) or {}
    if by_type:
        for t, c in by_type.items():
            by_type_rows.append([t, str(c), strategy_map.get(t, "-")])
    else:
        by_type_rows.append(["(none detected)", "0", "-"])
    by_type_table = Table(by_type_rows, colWidths=[55 * mm, 30 * mm, 80 * mm])
    by_type_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(by_type_table)

    # Object details (top 12 by confidence)
    objects = sorted(
        report.get("detected_objects", []) or [],
        key=lambda o: o.get("confidence", 0),
        reverse=True,
    )[:12]
    if objects:
        story.append(Paragraph("Object Details (top 12 by confidence)", section_style))
        detail_rows = [["ID", "Type", "Confidence", "BBox [x, y, w, h]"]]
        for o in objects:
            bb = o.get("bbox", [0, 0, 0, 0])
            detail_rows.append([
                str(o.get("id", "-")),
                str(o.get("type", "-")),
                f"{float(o.get('confidence', 0)):.3f}",
                f"[{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}]",
            ])
        detail_table = Table(detail_rows, colWidths=[15 * mm, 30 * mm, 30 * mm, 90 * mm])
        detail_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ]))
        story.append(detail_table)

    # Thumbnail of first masked frame
    if masked_frame_path and masked_frame_path.exists():
        story.append(Paragraph("Sample Masked Frame", section_style))
        # Scale to fit content width (174mm) while preserving aspect
        story.append(RLImage(str(masked_frame_path), width=160 * mm, height=90 * mm, kind="proportional"))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "Generated by Privacy Guard Agent — GPT-4o supervisor + Meta SAM3 vision worker.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey,
                       alignment=1),
    ))

    doc.build(story)


@tool
def generate_report(job_id: str) -> str:
    """Write the final PII summary report to outputs/{job_id}/report.json AND report.pdf
    (PDF includes summary tables + thumbnail of first masked frame). Call LAST."""
    store = get_store(job_id)
    by_type = dict(Counter(o["type"] for o in store.detected_objects))
    report = {
        "job_id": job_id,
        "scene_type": store.scene_type,
        "expected_pii": store.expected_pii,
        "total_objects": len(store.detected_objects),
        "by_type": by_type,
        "detected_objects": store.detected_objects,
        "output_video_path": store.output_video_path,
        "detect_attempts": store.detect_attempts,
    }
    out_dir = settings.output_path / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pdf_path = out_dir / "report.pdf"
    masked_first = None
    if store.masked_frames_dir:
        candidates = sorted(Path(store.masked_frames_dir).glob("*.jpg"))
        if candidates:
            masked_first = candidates[0]
    pdf_ok = True
    pdf_error = None
    try:
        _build_report_pdf(pdf_path, report, masked_first)
    except Exception as e:
        pdf_ok = False
        pdf_error = str(e)

    store.report = report
    emit_log(
        job_id,
        {
            "step": "tool", "action": "generate_report",
            "result": {
                "total": len(store.detected_objects), "by_type": by_type,
                "pdf": str(pdf_path) if pdf_ok else None,
                "pdf_error": pdf_error,
            },
        },
    )
    pdf_msg = " + report.pdf" if pdf_ok else f" (PDF FAILED: {pdf_error})"
    return f"Report saved (report.json{pdf_msg}). {len(store.detected_objects)} objects total, by_type={by_type}."


ALL_TOOLS = [
    extract_frames,
    analyze_scene,
    detect_pii,
    track_objects,
    mask_frames,
    compose_video,
    generate_report,
]
