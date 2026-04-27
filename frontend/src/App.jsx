import { useState } from "react";
import UploadZone from "./components/UploadZone";
import AgentLog from "./components/AgentLog";
import ReportPanel from "./components/ReportPanel";
import DownloadButton from "./components/DownloadButton";
import VideoPreview from "./components/VideoPreview";
import { useJobStatus } from "./hooks/useJobStatus";

export default function App() {
  const [jobId, setJobId] = useState(null);
  const { status, error } = useJobStatus(jobId, { enabled: !!jobId });
  const ready = status === "done";

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b px-6 py-4">
        <h1 className="text-lg font-semibold">Privacy Guard Agent</h1>
        <p className="text-xs text-muted">영상 PII 자동 비식별화</p>
      </header>

      <main className="max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="space-y-4">
          <UploadZone onUploaded={setJobId} disabled={!!jobId && !ready && status !== "failed"} />
          {jobId && (
            <div className="bg-white rounded-xl border p-4 text-xs">
              <p className="text-muted">job_id</p>
              <p className="font-mono">{jobId}</p>
              <p className="text-muted mt-2">상태</p>
              <p className="font-medium">
                {status}
                {error && <span className="text-red-600 ml-2">({error})</span>}
              </p>
              {(ready || status === "failed") && (
                <button
                  onClick={() => setJobId(null)}
                  className="mt-3 text-info underline text-xs"
                >
                  새 영상 업로드
                </button>
              )}
            </div>
          )}
          <ReportPanel jobId={jobId} ready={ready} />
          <DownloadButton jobId={jobId} ready={ready} />
        </section>

        <section className="space-y-4">
          <div className="h-[400px]">
            <AgentLog jobId={jobId} />
          </div>
          <VideoPreview jobId={jobId} ready={ready} />
        </section>
      </main>
    </div>
  );
}
