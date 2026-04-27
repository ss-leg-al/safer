import { downloadUrl, reportPdfUrl } from "../api/client";

export default function DownloadButton({ jobId, ready }) {
  if (!ready) return null;
  return (
    <div className="flex flex-wrap gap-2">
      <a
        href={downloadUrl(jobId)}
        download
        className="inline-flex items-center justify-center px-4 py-2 bg-info text-white rounded-lg text-sm font-medium hover:bg-info/90 transition"
      >
        ⬇ 마스킹 영상 다운로드
      </a>
      <a
        href={reportPdfUrl(jobId)}
        download
        className="inline-flex items-center justify-center px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition"
      >
        📄 리포트 PDF
      </a>
    </div>
  );
}
