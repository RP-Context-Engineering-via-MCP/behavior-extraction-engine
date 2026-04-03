import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSession } from '../SessionContext';
import { api } from '../api';

/* ─── intent badge colours (white-theme friendly) ─── */
const INTENT_COLORS = {
  HABIT:         { bg: '#FEF3C7', text: '#92400E', border: '#FDE68A' },
  PREFERENCE:    { bg: '#DBEAFE', text: '#1E40AF', border: '#BFDBFE' },
  COMMUNICATION: { bg: '#E0E7FF', text: '#3730A3', border: '#C7D2FE' },
  SKILL:         { bg: '#D1FAE5', text: '#065F46', border: '#A7F3D0' },
  CONSTRAINT:    { bg: '#FCE7F3', text: '#9D174D', border: '#FBCFE8' },
  DEFAULT:       { bg: '#F3F4F6', text: '#374151', border: '#D1D5DB' },
};

const CHART_LINE_COLORS = {
  HABIT:         '#F59E0B',
  PREFERENCE:    '#3B82F6',
  COMMUNICATION: '#6366F1',
  SKILL:         '#10B981',
  CONSTRAINT:    '#EC4899',
  DEFAULT:       '#6B7280',
};

function IntentBadge({ intent }) {
  const c = INTENT_COLORS[intent] || INTENT_COLORS.DEFAULT;
  return (
    <span
      className="text-[11px] font-semibold px-2 py-0.5 rounded-full border"
      style={{ background: c.bg, color: c.text, borderColor: c.border }}
    >
      {intent || 'UNKNOWN'}
    </span>
  );
}

/* ─── Simple Canvas Decay-Curve Chart ─── */
function DecayCurveChart({ curves, gracePeriod }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!curves || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    // Padding
    const pad = { top: 30, right: 30, bottom: 50, left: 60 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;

    // Clear
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, W, H);

    // Pick first curve to get day range
    const intents = Object.keys(curves);
    if (intents.length === 0) return;
    const maxDay = curves[intents[0]].points.length - 1;

    // Axes
    ctx.strokeStyle = '#E5E7EB';
    ctx.lineWidth = 1;
    // y-axis grid lines
    for (let i = 0; i <= 10; i++) {
      const y = pad.top + ch - (i / 10) * ch;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + cw, y);
      ctx.stroke();
    }
    // x-axis grid lines
    const xSteps = Math.min(maxDay, 10);
    for (let i = 0; i <= xSteps; i++) {
      const day = Math.round((i / xSteps) * maxDay);
      const x = pad.left + (day / maxDay) * cw;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + ch);
      ctx.stroke();
    }

    // Grace period shaded area
    if (gracePeriod && gracePeriod > 0) {
      const gpX = pad.left + (gracePeriod / maxDay) * cw;
      ctx.fillStyle = 'rgba(16,185,129,0.06)';
      ctx.fillRect(pad.left, pad.top, gpX - pad.left, ch);
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(gpX, pad.top);
      ctx.lineTo(gpX, pad.top + ch);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#065F46';
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.fillText('Grace Period', pad.left + 6, pad.top + 16);
    }

    // Draw curves
    intents.forEach((intent) => {
      const { points } = curves[intent];
      const color = CHART_LINE_COLORS[intent] || '#6B7280';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((p, idx) => {
        const x = pad.left + (p.day / maxDay) * cw;
        const y = pad.top + ch - p.credibility * ch;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Axis labels
    ctx.fillStyle = '#6B7280';
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    // X axis
    for (let i = 0; i <= xSteps; i++) {
      const day = Math.round((i / xSteps) * maxDay);
      const x = pad.left + (day / maxDay) * cw;
      ctx.fillText(`${day}d`, x, pad.top + ch + 20);
    }
    ctx.fillText('Days', pad.left + cw / 2, pad.top + ch + 42);
    // Y axis
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i += 2) {
      const y = pad.top + ch - (i / 10) * ch;
      ctx.fillText((i / 10).toFixed(1), pad.left - 8, y + 4);
    }
    ctx.save();
    ctx.translate(16, pad.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Credibility', 0, 0);
    ctx.restore();

    // Axis lines
    ctx.strokeStyle = '#9CA3AF';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + ch);
    ctx.lineTo(pad.left + cw, pad.top + ch);
    ctx.stroke();
  }, [curves, gracePeriod]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full border border-gray-200 rounded-xl bg-white"
      style={{ height: 340 }}
    />
  );
}

/* ─── Credibility Bar ─── */
function CredBar({ value, max = 1, color = '#3B82F6', label }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-[10px] text-gray-400 w-14 shrink-0">{label}</span>}
      <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="text-xs font-mono text-gray-600 w-12 text-right">{value.toFixed(4)}</span>
    </div>
  );
}

/* ─── Behavior Card (Preview Mode) ─── */
function BehaviorPreviewCard({ b }) {
  const hasLoss = b.credibility_loss > 0.0001;
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 leading-snug">{b.behavior_text}</p>
          <div className="flex items-center gap-2 mt-1.5">
            <IntentBadge intent={b.intent} />
            {b.polarity && (
              <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${
                b.polarity === 'POSITIVE' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
              }`}>{b.polarity}</span>
            )}
            {b.in_grace_period && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                🛡 GRACE PERIOD
              </span>
            )}
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className="text-[10px] text-gray-400">λ = {b.decay_rate}/day</div>
          <div className="text-[10px] text-gray-400">
            Age: {b.age_days}d
          </div>
          {b.reinforcement_count > 0 && (
            <div className="text-[10px] text-blue-500 font-medium">
              ×{b.reinforcement_count} reinforced
            </div>
          )}
        </div>
      </div>

      {/* Stored vs Decayed credibility */}
      <div className="space-y-1.5">
        <CredBar value={b.stored_credibility} label="Stored" color="#9CA3AF" />
        <CredBar
          value={b.decayed_credibility}
          label="Decayed"
          color={hasLoss ? '#EF4444' : '#10B981'}
        />
      </div>

      {hasLoss ? (
        <div className="mt-2 flex items-center gap-2 text-[11px]">
          <span className="text-red-500 font-medium">
            ▼ {b.credibility_loss.toFixed(4)} loss
          </span>
          <span className="text-gray-400">
            ({b.full_days_for_decay} full day{b.full_days_for_decay !== 1 ? 's' : ''} elapsed)
          </span>
        </div>
      ) : (
        <div className="mt-2 text-[11px] text-green-600 font-medium">
          ✓ No decay pending
        </div>
      )}
    </div>
  );
}

/* ─── After-Apply Result Card ─── */
function BehaviorResultCard({ b }) {
  const changed = b.decay_applied;
  return (
    <div className={`border rounded-xl p-3 transition ${
      changed ? 'border-amber-300 bg-amber-50/40' : 'border-gray-200 bg-white'
    }`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm text-gray-900 leading-snug">{b.behavior_text}</p>
          <div className="flex items-center gap-2 mt-1">
            <IntentBadge intent={b.intent} />
            <span className="text-[10px] text-gray-400">λ = {b.decay_rate}/day</span>
          </div>
        </div>
        {changed ? (
          <div className="text-right shrink-0">
            <div className="text-xs font-mono text-gray-500 line-through">
              {b.before_credibility.toFixed(4)}
            </div>
            <div className="text-xs font-mono text-red-600 font-bold">
              → {b.after_credibility.toFixed(4)}
            </div>
            <div className="text-[10px] text-amber-600 mt-0.5">
              {b.days_elapsed}d decay applied
            </div>
          </div>
        ) : (
          <div className="text-xs font-mono text-gray-500">
            {b.before_credibility.toFixed(4)} (unchanged)
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MAIN PAGE                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */
export default function DecayPage() {
  const { userId } = useSession();

  // Decay config
  const [config, setConfig] = useState(null);
  // Simulation curves
  const [curves, setCurves] = useState(null);
  const [simDays, setSimDays] = useState(90);
  const [simCred, setSimCred] = useState(0.85);
  // Preview
  const [preview, setPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  // Apply result
  const [applyResult, setApplyResult] = useState(null);
  const [applyLoading, setApplyLoading] = useState(false);
  const [applyMode, setApplyMode] = useState(null); // 'lazy' | 'cron'

  // Tab state
  const [tab, setTab] = useState('curves'); // 'curves' | 'behaviors'

  /* ── Load config on mount ── */
  useEffect(() => {
    api.getDecayConfig().then(setConfig).catch(console.error);
  }, []);

  /* ── Load simulation curves ── */
  const loadCurves = useCallback(() => {
    api.simulateDecay(simDays, simCred).then((res) => {
      if (res.success) setCurves(res.data);
    }).catch(console.error);
  }, [simDays, simCred]);

  useEffect(() => { loadCurves(); }, [loadCurves]);

  /* ── Preview decay state ── */
  const loadPreview = async () => {
    if (!userId) return;
    setPreviewLoading(true);
    setApplyResult(null);
    try {
      const res = await api.previewDecay(userId);
      if (res.success) setPreview(res.data);
      else setPreview(null);
    } catch (e) { console.error(e); }
    setPreviewLoading(false);
  };

  /* ── Apply decay (lazy or cron) ── */
  const applyDecay = async (mode) => {
    if (!userId) return;
    setApplyLoading(true);
    setApplyMode(mode);
    try {
      const res = mode === 'lazy'
        ? await api.applyLazyDecay(userId)
        : await api.applyCronDecay(userId);
      if (res.success) {
        setApplyResult(res.data);
        // Refresh preview after applying
        setTimeout(() => loadPreview(), 500);
      }
    } catch (e) { console.error(e); }
    setApplyLoading(false);
  };

  const cfgData = config?.data;

  return (
    <div className="p-6 max-w-[1200px] mx-auto space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-gray-900">Decay & Credibility Demo</h2>
        <p className="text-sm text-gray-500 mt-1">
          Demonstrates how behavior credibility decays over time using exponential decay.
          Compare <strong>Lazy Decay</strong> (on-demand at retrieval) versus <strong>Cron Job</strong> (scheduled batch processing).
        </p>
      </div>

      {/* Config Summary */}
      {cfgData && (
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-800 mb-3">Algorithm Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-[10px] uppercase text-gray-400 font-medium tracking-wide">Formula</div>
              <div className="text-sm font-mono text-gray-800 mt-1">{cfgData.formula}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-[10px] uppercase text-gray-400 font-medium tracking-wide">Grace Period</div>
              <div className="text-sm font-semibold text-gray-800 mt-1">{cfgData.grace_period_days} days</div>
              <div className="text-[10px] text-gray-400">New behaviors don't decay</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-[10px] uppercase text-gray-400 font-medium tracking-wide">Default Rate</div>
              <div className="text-sm font-semibold text-gray-800 mt-1">{cfgData.default_decay_rate}/day</div>
            </div>
          </div>

          {/* Intent decay rates */}
          <div className="text-[10px] uppercase text-gray-400 font-medium tracking-wide mb-2">Intent Decay Rates (λ per day)</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(cfgData.intent_decay_rates).map(([intent, rate]) => {
              const c = INTENT_COLORS[intent] || INTENT_COLORS.DEFAULT;
              return (
                <div
                  key={intent}
                  className="flex items-center gap-2 border rounded-lg px-3 py-1.5"
                  style={{ background: c.bg, borderColor: c.border }}
                >
                  <span className="text-xs font-semibold" style={{ color: c.text }}>{intent}</span>
                  <span className="text-xs font-mono" style={{ color: c.text }}>λ = {rate}</span>
                </div>
              );
            })}
          </div>

          {/* Notes */}
          <div className="mt-3 flex flex-wrap gap-2">
            {cfgData.notes.map((note, i) => (
              <span key={i} className="text-[10px] text-gray-500 bg-gray-50 border border-gray-100 rounded px-2 py-1">
                ℹ {note}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-xl w-fit">
        <button
          onClick={() => setTab('curves')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
            tab === 'curves' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          📈 Decay Curves
        </button>
        <button
          onClick={() => setTab('behaviors')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
            tab === 'behaviors' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          🔍 Behavior Decay State
        </button>
      </div>

      {/* ═══ CURVES TAB ═══ */}
      {tab === 'curves' && (
        <div className="space-y-4">
          {/* Controls */}
          <div className="bg-white border border-gray-200 rounded-xl p-4 flex items-end gap-4 flex-wrap">
            <div>
              <label className="text-[10px] uppercase text-gray-400 font-medium tracking-wide block mb-1">
                Simulation Period
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={7}
                  max={365}
                  value={simDays}
                  onChange={(e) => setSimDays(Number(e.target.value))}
                  className="w-40 accent-blue-600"
                />
                <span className="text-sm font-mono text-gray-700 w-12">{simDays}d</span>
              </div>
            </div>
            <div>
              <label className="text-[10px] uppercase text-gray-400 font-medium tracking-wide block mb-1">
                Initial Credibility
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={simCred}
                  onChange={(e) => setSimCred(Number(e.target.value))}
                  className="w-40 accent-blue-600"
                />
                <span className="text-sm font-mono text-gray-700 w-12">{simCred.toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Chart */}
          {curves && (
            <div className="bg-white border border-gray-200 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-800 mb-3">
                Credibility Decay Over Time (all intent types)
              </h3>
              <DecayCurveChart curves={curves.curves} gracePeriod={curves.grace_period_days} />
              {/* Legend */}
              <div className="flex flex-wrap gap-3 mt-3 justify-center">
                {Object.entries(CHART_LINE_COLORS).map(([intent, color]) => (
                  <div key={intent} className="flex items-center gap-1.5">
                    <div className="w-4 h-[3px] rounded-full" style={{ background: color }} />
                    <span className="text-[11px] text-gray-600">{intent}</span>
                    <span className="text-[10px] text-gray-400 font-mono">
                      (λ={curves.curves[intent]?.decay_rate ?? '?'})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Explanation cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
              <h4 className="text-sm font-semibold text-blue-800 mb-2">⏱ Lazy Decay (On Demand)</h4>
              <ul className="text-xs text-blue-700 space-y-1.5">
                <li>• Decay is calculated <strong>when behaviors are retrieved</strong> (read-time)</li>
                <li>• No background process needed — zero infrastructure overhead</li>
                <li>• Computation happens in <code className="bg-blue-100 px-1 rounded">search_similar_behaviors()</code></li>
                <li>• Only behaviors that are accessed get decayed</li>
                <li>• Updated credibility is persisted back to DB after calculation</li>
                <li>• <strong>Pro:</strong> Simple, no scheduler needed</li>
                <li>• <strong>Con:</strong> Unaccessed behaviors retain stale credibility</li>
              </ul>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
              <h4 className="text-sm font-semibold text-amber-800 mb-2">🕐 Cron Job (Batch)</h4>
              <ul className="text-xs text-amber-700 space-y-1.5">
                <li>• A scheduled job runs periodically (e.g., once a week)</li>
                <li>• Applies decay to <strong>ALL behaviors</strong> regardless of access</li>
                <li>• Uses the same exponential formula: <code className="bg-amber-100 px-1 rounded">C × e^(-λ × days)</code></li>
                <li>• Batch UPDATE in a single database transaction</li>
                <li>• <strong>Pro:</strong> All behaviors always reflect current credibility</li>
                <li>• <strong>Con:</strong> Requires background scheduler (cron, Celery, APScheduler)</li>
                <li>• <strong>Con:</strong> May process many behaviors unnecessarily</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* ═══ BEHAVIORS TAB ═══ */}
      {tab === 'behaviors' && (
        <div className="space-y-4">
          {/* Controls */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <h3 className="text-sm font-semibold text-gray-800">User Behaviors Decay State</h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Preview what decay WOULD apply, or trigger it for real
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={loadPreview}
                  disabled={!userId || previewLoading}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200 disabled:opacity-50 transition"
                >
                  {previewLoading ? 'Loading…' : '🔍 Preview Decay'}
                </button>
                <button
                  onClick={() => applyDecay('lazy')}
                  disabled={!userId || applyLoading}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100 disabled:opacity-50 transition"
                >
                  {applyLoading && applyMode === 'lazy' ? 'Applying…' : '⏱ Apply Lazy Decay'}
                </button>
                <button
                  onClick={() => applyDecay('cron')}
                  disabled={!userId || applyLoading}
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100 disabled:opacity-50 transition"
                >
                  {applyLoading && applyMode === 'cron' ? 'Applying…' : '🕐 Apply Cron Decay'}
                </button>
              </div>
            </div>
          </div>

          {/* Apply Result Banner */}
          {applyResult && (
            <div className={`border rounded-xl p-4 ${
              applyResult.mode === 'lazy' ? 'bg-blue-50 border-blue-200' : 'bg-amber-50 border-amber-200'
            }`}>
              <div className="flex items-center justify-between mb-3">
                <h4 className={`text-sm font-semibold ${
                  applyResult.mode === 'lazy' ? 'text-blue-800' : 'text-amber-800'
                }`}>
                  {applyResult.mode === 'lazy' ? '⏱ Lazy Decay' : '🕐 Cron Batch Decay'} Applied
                </h4>
                <div className="flex items-center gap-3 text-xs">
                  <span className="font-medium text-gray-700">
                    {applyResult.total_behaviors} total
                  </span>
                  <span className="text-red-600 font-medium">
                    {applyResult.total_decayed} decayed
                  </span>
                  <span className="text-green-600 font-medium">
                    {applyResult.total_unchanged} unchanged
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[400px] overflow-y-auto">
                {applyResult.behaviors.map((b) => (
                  <BehaviorResultCard key={b.behavior_id} b={b} />
                ))}
              </div>
            </div>
          )}

          {/* Preview Grid */}
          {!userId && (
            <div className="text-center py-16 text-gray-400 text-sm">
              Enter a User ID in the top bar, then click <strong>Preview Decay</strong>
            </div>
          )}

          {userId && !preview && !previewLoading && !applyResult && (
            <div className="text-center py-16 text-gray-400 text-sm">
              Click <strong>Preview Decay</strong> to see the current decay state of behaviors
            </div>
          )}

          {preview && (
            <>
              {/* Summary Banner */}
              <div className="flex items-center gap-4 text-xs">
                <span className="text-gray-500">
                  <strong className="text-gray-800">{preview.total_behaviors}</strong> behaviors
                </span>
                <span className="text-amber-600">
                  <strong>{preview.behaviors_needing_decay}</strong> need decay
                </span>
                <span className="text-green-600">
                  <strong>{preview.total_behaviors - preview.behaviors_needing_decay}</strong> up to date
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {preview.behaviors.map((b) => (
                  <BehaviorPreviewCard key={b.behavior_id} b={b} />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
