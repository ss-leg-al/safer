from langchain_core.messages import HumanMessage

from .graph import app_graph
from .job_store import get_store, reset_store
from .log_emitter import emit_log, write_status

RECURSION_LIMIT = 30


def run_agent_job(job_id: str, video_path: str) -> None:
    write_status(job_id, "running")
    store = get_store(job_id)
    store.video_path = video_path

    initial = {
        "messages": [
            HumanMessage(
                content=(
                    f"영상의 PII를 비식별화해주세요. job_id={job_id}. "
                    f"필요한 도구들을 처음부터 끝까지 실행해서 output.mp4와 report.json을 만들어주세요."
                )
            )
        ]
    }

    try:
        for chunk in app_graph.stream(initial, config={"recursion_limit": RECURSION_LIMIT}):
            if "agent" in chunk:
                msg = chunk["agent"]["messages"][-1]
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    rationale = (msg.content or "").strip()[:400]
                    for tc in tool_calls:
                        payload = {
                            "step": "supervisor",
                            "action": "decide",
                            "tool_call": f"{tc['name']}({_short_args(tc.get('args', {}))})",
                        }
                        if rationale:
                            payload["thinking"] = rationale
                        emit_log(job_id, payload)
                else:
                    emit_log(
                        job_id,
                        {
                            "step": "supervisor",
                            "action": "final",
                            "thinking": (msg.content or "").strip()[:600],
                        },
                    )

        write_status(job_id, "done")
        emit_log(job_id, {"event": "done"})
    except Exception as e:
        write_status(job_id, "failed", error=str(e))
        emit_log(job_id, {"event": "failed", "error": str(e)})
    finally:
        reset_store(job_id)


def _short_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)
