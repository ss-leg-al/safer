import { useEffect, useState } from "react";
import { fetchReport } from "../api/client";

export default function ReportPanel({ jobId, ready }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ready || !jobId) return;
    fetchReport(jobId).then(setReport).catch((e) => setError(e.message));
  }, [jobId, ready]);

  if (!ready) return null;
  if (error) return <p className="text-xs text-red-600">리포트 로드 실패: {error}</p>;
  if (!report) return <p className="text-xs text-muted">리포트 로딩 중…</p>;

  return (
    <div className="bg-white rounded-xl border p-4">
      <h2 className="text-sm font-semibold text-tertiary mb-3">PII 탐지 리포트</h2>
      <div className="grid grid-cols-2 gap-2 text-xs mb-3">
        <div>
          <span className="text-muted">씬 타입</span>
          <p className="font-medium">{report.scene_type || "-"}</p>
        </div>
        <div>
          <span className="text-muted">탐지 객체</span>
          <p className="font-medium">{report.total_objects}</p>
        </div>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-muted border-b">
            <th className="py-1">type</th>
            <th>bbox</th>
            <th>confidence</th>
            <th>strategy</th>
          </tr>
        </thead>
        <tbody>
          {report.detected_objects.map((o, i) => (
            <tr key={i} className="border-b last:border-0">
              <td className="py-1 font-mono">{o.type}</td>
              <td className="font-mono">[{o.bbox.join(", ")}]</td>
              <td>{(o.confidence * 100).toFixed(0)}%</td>
              <td>{o.mask_strategy || "-"}</td>
            </tr>
          ))}
          {report.detected_objects.length === 0 && (
            <tr>
              <td colSpan={4} className="text-muted py-2">탐지된 객체 없음</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
