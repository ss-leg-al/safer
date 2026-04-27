import { downloadUrl } from "../api/client";

export default function VideoPreview({ jobId, ready }) {
  if (!ready) return null;
  return (
    <div className="bg-black rounded-xl overflow-hidden">
      <video src={downloadUrl(jobId)} controls className="w-full" />
    </div>
  );
}
