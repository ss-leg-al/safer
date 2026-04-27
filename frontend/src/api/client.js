import axios from "axios";

// Empty string => relative URLs, routed through vite proxy in dev (see vite.config.js).
// Set VITE_API_URL only when frontend and backend are on different origins.
export const API_BASE = import.meta.env.VITE_API_URL ?? "";

export const api = axios.create({
  baseURL: API_BASE,
});

export async function uploadVideo(file, onUploadProgress) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/api/jobs", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress,
  });
  return data;
}

export async function fetchReport(jobId) {
  const { data } = await api.get(`/api/jobs/${jobId}/report`);
  return data;
}

export async function fetchStatus(jobId) {
  const { data } = await api.get(`/api/jobs/${jobId}/status`);
  return data;
}

export function downloadUrl(jobId) {
  return `${API_BASE}/api/jobs/${jobId}/download`;
}

export function reportPdfUrl(jobId) {
  return `${API_BASE}/api/jobs/${jobId}/report.pdf`;
}

export function streamUrl(jobId) {
  return `${API_BASE}/api/jobs/${jobId}/stream`;
}
