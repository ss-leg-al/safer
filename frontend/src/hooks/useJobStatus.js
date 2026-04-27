import { useEffect, useState } from "react";
import { fetchStatus } from "../api/client";

export function useJobStatus(jobId, { enabled = true, intervalMs = 2000 } = {}) {
  const [status, setStatus] = useState("pending");
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId || !enabled) return;
    let active = true;

    const tick = async () => {
      try {
        const data = await fetchStatus(jobId);
        if (!active) return;
        setStatus(data.status);
        setError(data.error || null);
        if (data.status === "done" || data.status === "failed") {
          return;
        }
      } catch {
        // ignore transient errors
      }
      if (active) setTimeout(tick, intervalMs);
    };
    tick();
    return () => {
      active = false;
    };
  }, [jobId, enabled, intervalMs]);

  return { status, error };
}
