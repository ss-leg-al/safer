import { useAgentLog } from "../hooks/useAgentLog";

const STEP_LABELS = {
  decide: "🧠 Supervisor — 다음 행동 결정",
  final: "🧠 Supervisor — 최종 응답",
  extract_frames: "🔧 Tool — 프레임 추출",
  analyze_scene: "🔧 Tool — 씬 분석 (GPT-4o)",
  detect_pii: "🔧 Tool — PII 탐지 (SAM3)",
  track_objects: "🔧 Tool — SAM3 프레임 전파",
  mask_frames: "🔧 Tool — 마스킹 적용",
  compose_video: "🔧 Tool — 영상 합성",
  generate_report: "🔧 Tool — 리포트 생성",
};

const MSG_STYLE = {
  thinking: "italic text-sm text-tertiary bg-secondary px-3 py-2 rounded-lg",
  tool: "font-mono text-xs text-info bg-info/10 px-3 py-2 rounded",
  result: "text-xs text-success bg-success/10 px-3 py-2 rounded",
  error: "text-xs text-red-700 bg-red-100 px-3 py-2 rounded",
};

export default function AgentLog({ jobId }) {
  const { logs, done, failed } = useAgentLog(jobId);

  if (!jobId) return null;

  return (
    <div className="flex flex-col gap-2 p-4 bg-white rounded-xl border h-full overflow-y-auto">
      <h2 className="text-sm font-semibold text-tertiary mb-2">에이전트 판단 로그</h2>
      {logs.map((log, i) => (
        <div key={i}>
          {STEP_LABELS[log.action] && (
            <div className="text-xs text-tertiary border rounded px-2 py-0.5 w-fit mb-1">
              {STEP_LABELS[log.action]}
            </div>
          )}
          {log.thinking && <p className={MSG_STYLE.thinking}>"{log.thinking}"</p>}
          {log.tool_call && <p className={MSG_STYLE.tool}>→ {log.tool_call}</p>}
          {log.result && (
            <p className={MSG_STYLE.result}>
              ✓ {typeof log.result === "string" ? log.result : JSON.stringify(log.result)}
            </p>
          )}
          {log.error && <p className={MSG_STYLE.error}>⚠ {log.error}</p>}
          {log.warning && <p className={MSG_STYLE.error}>⚠ {log.warning}</p>}
        </div>
      ))}
      {done && <p className="text-xs text-success font-medium mt-2">✓ 처리 완료</p>}
      {failed && <p className="text-xs text-red-600 font-medium mt-2">✗ 실패: {failed}</p>}
    </div>
  );
}
