import { useEffect, useState } from "react";
import { streamUrl } from "../api/client";

export function useAgentLog(jobId) {
  const [logs, setLogs] = useState([]);
  const [done, setDone] = useState(false);
  const [failed, setFailed] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    setLogs([]);
    setDone(false);
    setFailed(null);

    const es = new EventSource(streamUrl(jobId));
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.event === "done") {
          setDone(true);
          es.close();
          return;
        }
        if (data.event === "failed") {
          setFailed(data.error || "unknown error");
          es.close();
          return;
        }
        setLogs((prev) => [...prev, data]);
      } catch {
        // ignore malformed
      }
    };
    es.onerror = () => {
      es.close();
    };
    return () => es.close();
  }, [jobId]);

  return { logs, done, failed };
}
