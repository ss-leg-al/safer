import { useRef, useState } from "react";
import { uploadVideo } from "../api/client";

export default function UploadZone({ onUploaded, disabled }) {
  const inputRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);

  const handleFile = async (file) => {
    if (!file) return;
    setError(null);
    setUploading(true);
    setProgress(0);
    try {
      const { job_id } = await uploadVideo(file, (e) => {
        if (e.total) setProgress(Math.round((e.loaded / e.total) * 100));
      });
      onUploaded(job_id);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (disabled) return;
        const file = e.dataTransfer.files?.[0];
        handleFile(file);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition ${
        drag ? "border-info bg-info/5" : "border-gray-300 bg-white"
      } ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-info"}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
        disabled={disabled}
      />
      {uploading ? (
        <div>
          <p className="text-sm text-tertiary mb-2">업로드 중… {progress}%</p>
          <div className="w-full h-2 bg-gray-200 rounded">
            <div
              className="h-full bg-info rounded transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ) : (
        <>
          <p className="text-base font-medium">영상을 끌어다 놓거나 클릭해 선택</p>
          <p className="text-xs text-muted mt-1">mp4 / mov / mkv / webm · 최대 200MB · 5분 이내</p>
        </>
      )}
      {error && <p className="text-xs text-red-500 mt-3">{error}</p>}
    </div>
  );
}
