import React, { useState, useRef } from 'react';
import { useSession } from '../SessionContext';
import { api } from '../api';

/* ─────────────────────────── constants & helpers ─────────────────────────── */

const ACTION_STYLES = {
  NEW_BEHAVIOR:           { bg: 'bg-emerald-50',  border: 'border-emerald-200', text: 'text-emerald-700', icon: '🆕', label: 'New Behavior' },
  DUPLICATE_REINFORCED:   { bg: 'bg-blue-50',     border: 'border-blue-200',    text: 'text-blue-700',    icon: '🔁', label: 'Reinforced' },
  CONFLICT_DETECTED:      { bg: 'bg-red-50',      border: 'border-red-200',     text: 'text-red-700',     icon: '⚡', label: 'Conflict Detected' },
  CONFLICT_AUTO_RESOLVED: { bg: 'bg-amber-50',    border: 'border-amber-200',   text: 'text-amber-700',   icon: '🤖', label: 'Auto-Resolved' },
  SUPERSEDED_EXISTING:    { bg: 'bg-orange-50',   border: 'border-orange-200',  text: 'text-orange-700',  icon: '🔄', label: 'Superseded Old' },
  IGNORED_NEW:            { bg: 'bg-gray-50',     border: 'border-gray-200',    text: 'text-gray-500',    icon: '🚫', label: 'Ignored (Low Cred)' },
  COMPATIBLE:             { bg: 'bg-teal-50',     border: 'border-teal-200',    text: 'text-teal-700',    icon: '✅', label: 'Compatible' },
  PRUNED:                 { bg: 'bg-gray-50',     border: 'border-gray-200',    text: 'text-gray-400',    icon: '✂️', label: 'Pruned' },
};

function getActionStyle(action) {
  return ACTION_STYLES[action] || ACTION_STYLES.PRUNED;
}

function Badge({ children, className = '' }) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${className}`}>
      {children}
    </span>
  );
}

function Card({ children, className = '', delay = 0 }) {
  return (
    <div
      className={`bg-white border border-gray-200 rounded-2xl p-5 shadow-sm animate-slide-in ${className}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/* ───────────────────── Credibility bar with color coding ───────────────────── */

function CredibilityBar({ value, label, size = 'md' }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? 'bg-emerald-500' :
    pct >= 60 ? 'bg-blue-500'    :
    pct >= 40 ? 'bg-amber-500'   :
                'bg-red-500';
  const h = size === 'sm' ? 'h-1.5' : 'h-2.5';
  return (
    <div className="flex items-center gap-2 min-w-0">
      {label && <span className="text-xs text-gray-500 shrink-0">{label}</span>}
      <div className={`flex-1 ${h} bg-gray-100 rounded-full overflow-hidden`}>
        <div className={`${h} ${color} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-semibold text-gray-700 shrink-0">{value.toFixed(4)}</span>
    </div>
  );
}

/* ─────────────────────── Canonical tag row ───────────────────────────── */

function CanonicalTags({ canonical }) {
  if (!canonical) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {canonical.intent && (
        <span className="px-2 py-0.5 bg-purple-50 text-purple-700 border border-purple-200 rounded-full text-[10px] font-medium">
          {canonical.intent}
        </span>
      )}
      {canonical.target && (
        <span className="px-2 py-0.5 bg-sky-50 text-sky-700 border border-sky-200 rounded-full text-[10px] font-medium">
          {canonical.target}
        </span>
      )}
      {canonical.context && (
        <span className="px-2 py-0.5 bg-gray-100 text-gray-600 border border-gray-200 rounded-full text-[10px] font-medium">
          {canonical.context}
        </span>
      )}
      {canonical.polarity && (
        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${
          canonical.polarity === 'POSITIVE'
            ? 'bg-green-50 text-green-700 border-green-200'
            : 'bg-red-50 text-red-700 border-red-200'
        }`}>
          {canonical.polarity}
        </span>
      )}
    </div>
  );
}

/* ──────────────────── Reinforcement detail panel ──────────────────────── */

function ReinforcementPanel({ info }) {
  if (!info) return null;
  const { credibility_before, credibility_after, credibility_boost, reinforcement_count_before, reinforcement_count_after } = info;

  return (
    <div className="mt-3 bg-blue-50/50 border border-blue-200 rounded-xl p-4 space-y-3">
      <div className="flex items-center gap-2 text-sm font-semibold text-blue-800">
        <span>📈</span> Credibility Reinforcement
      </div>

      {/* Before / After bars */}
      {credibility_before != null && (
        <CredibilityBar value={credibility_before} label="Before" size="sm" />
      )}
      {credibility_after != null && (
        <CredibilityBar value={credibility_after} label="After" size="sm" />
      )}

      {/* Delta */}
      {credibility_boost != null && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Boost</span>
          <span className="text-sm font-bold text-emerald-600">+{credibility_boost.toFixed(4)}</span>
          <span className="text-[10px] text-gray-400">
            (count {reinforcement_count_before ?? '?'} → {reinforcement_count_after ?? '?'})
          </span>
        </div>
      )}
    </div>
  );
}

/* ──────────────────── Conflict detail panel ──────────────────────────── */

function ConflictPanel({ info, matchedText, matchedId }) {
  if (!info) return null;

  return (
    <div className="mt-3 bg-red-50/50 border border-red-200 rounded-xl p-4 space-y-2">
      <div className="flex items-center gap-2 text-sm font-semibold text-red-800">
        <span>⚡</span> Conflict Details
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {info.conflict_type && (
          <>
            <span className="text-gray-500">Type</span>
            <span className="font-medium text-gray-800">{info.conflict_type}</span>
          </>
        )}
        {info.existing_polarity && (
          <>
            <span className="text-gray-500">Existing Polarity</span>
            <span className="font-medium text-gray-800">{info.existing_polarity}</span>
          </>
        )}
        {info.new_polarity && (
          <>
            <span className="text-gray-500">New Polarity</span>
            <span className="font-medium text-gray-800">{info.new_polarity}</span>
          </>
        )}
        {info.resolution && (
          <>
            <span className="text-gray-500">Resolution</span>
            <Badge className={
              info.resolution === 'SUPERSEDE_EXISTING' ? 'bg-orange-100 text-orange-700' :
              info.resolution === 'IGNORE_NEW' ? 'bg-gray-100 text-gray-600' :
              info.resolution === 'USER_DECISION_NEEDED' ? 'bg-red-100 text-red-700' :
              'bg-amber-100 text-amber-700'
            }>
              {info.resolution}
            </Badge>
          </>
        )}
        {info.explanation && (
          <>
            <span className="text-gray-500">Explanation</span>
            <span className="font-medium text-gray-800">{info.explanation}</span>
          </>
        )}
      </div>

      {info.llm_analysis && (
        <div className="mt-2 text-xs text-gray-600 bg-white/60 rounded-lg p-2 border border-red-100">
          <span className="font-medium text-gray-700">LLM Analysis: </span>
          {info.llm_analysis}
          {info.llm_confidence != null && (
            <span className="ml-2 text-gray-400">(confidence: {info.llm_confidence})</span>
          )}
        </div>
      )}

      {matchedText && (
        <div className="mt-2 text-xs">
          <span className="text-gray-500">Conflicting with: </span>
          <span className="font-medium text-gray-800">"{matchedText}"</span>
          {matchedId && <span className="text-gray-400 ml-1">({matchedId.slice(0, 8)}…)</span>}
        </div>
      )}
    </div>
  );
}

/* ──────────────────── Superseded detail panel ──────────────────────────── */

function SupersededPanel({ info, matchedText, matchedId }) {
  return (
    <div className="mt-3 bg-orange-50/50 border border-orange-200 rounded-xl p-4 space-y-2">
      <div className="flex items-center gap-2 text-sm font-semibold text-orange-800">
        <span>🔄</span> Superseded Existing Behavior
      </div>
      {matchedText && (
        <div className="text-xs">
          <span className="text-gray-500">Old behavior: </span>
          <span className="line-through text-gray-400">"{matchedText}"</span>
          {matchedId && <span className="text-gray-400 ml-1">({matchedId.slice(0, 8)}…)</span>}
        </div>
      )}
      {info?.resolution && (
        <div className="text-xs">
          <span className="text-gray-500">Resolution: </span>
          <span className="font-medium text-orange-700">{info.resolution}</span>
        </div>
      )}
      {info?.explanation && (
        <div className="text-xs text-gray-600">{info.explanation}</div>
      )}
    </div>
  );
}

/* ──────────────────── Single behavior flow card ──────────────────────── */

function BehaviorFlowCard({ flow, index }) {
  const style = getActionStyle(flow.action);

  return (
    <div className={`${style.bg} border ${style.border} rounded-xl p-4 animate-slide-in`} style={{ animationDelay: `${index * 60}ms` }}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">{style.icon}</span>
            <Badge className={`${style.bg} ${style.text} border ${style.border}`}>
              {style.label}
            </Badge>
            {flow.stored_behavior_id && (
              <span className="text-[10px] text-gray-400 font-mono">{flow.stored_behavior_id.slice(0, 8)}…</span>
            )}
          </div>
          <p className="text-sm font-medium text-gray-900 leading-snug">"{flow.behavior_description}"</p>
          <CanonicalTags canonical={flow.canonical} />
        </div>

        {/* Credibility badge */}
        <div className="text-right shrink-0">
          <div className="text-[10px] text-gray-400 uppercase tracking-wide">Credibility</div>
          <div className="text-lg font-bold text-gray-900">{flow.credibility.toFixed(4)}</div>
        </div>
      </div>

      {/* Matched behavior info */}
      {flow.matched_behavior_id && flow.action === 'DUPLICATE_REINFORCED' && (
        <div className="mt-2 text-xs text-gray-500">
          <span>Matched: </span>
          <span className="font-medium text-gray-700">"{flow.matched_behavior_text}"</span>
          <span className="text-gray-400 ml-1">(distance: {flow.distance?.toFixed(4)})</span>
        </div>
      )}

      {/* Details string */}
      {flow.details && (
        <div className="mt-2 text-xs text-gray-500 italic">{flow.details}</div>
      )}

      {/* Reinforcement panel */}
      {flow.action === 'DUPLICATE_REINFORCED' && flow.reinforcement_info && (
        <ReinforcementPanel info={flow.reinforcement_info} />
      )}

      {/* Conflict panel */}
      {(flow.action === 'CONFLICT_DETECTED' || flow.action === 'CONFLICT_AUTO_RESOLVED') && (
        <ConflictPanel info={flow.conflict_info} matchedText={flow.matched_behavior_text} matchedId={flow.matched_behavior_id} />
      )}

      {/* Superseded panel */}
      {flow.action === 'SUPERSEDED_EXISTING' && (
        <SupersededPanel info={flow.conflict_info} matchedText={flow.matched_behavior_text} matchedId={flow.matched_behavior_id} />
      )}

      {/* Ignored panel */}
      {flow.action === 'IGNORED_NEW' && flow.conflict_info && (
        <ConflictPanel info={flow.conflict_info} matchedText={flow.matched_behavior_text} matchedId={flow.matched_behavior_id} />
      )}
    </div>
  );
}

/* ──────────────── Summary banner for an extraction round ──────────────── */

function ExtractionSummary({ summary }) {
  const items = [
    { label: 'Extracted', value: summary.total_extracted, color: 'text-gray-700' },
    { label: 'Stored',     value: summary.total_stored,     color: 'text-emerald-600' },
    { label: 'Reinforced', value: summary.total_reinforced,  color: 'text-blue-600' },
    { label: 'Conflicts',  value: summary.total_conflicts,   color: 'text-red-600' },
    { label: 'Pruned',     value: summary.total_pruned,      color: 'text-gray-400' },
  ];

  return (
    <div className="flex flex-wrap gap-4">
      {items.map(({ label, value, color }) => (
        <div key={label} className="flex items-center gap-1.5">
          <span className={`text-xl font-bold ${color}`}>{value}</span>
          <span className="text-xs text-gray-400">{label}</span>
        </div>
      ))}
    </div>
  );
}

/* ───────────────── Collapsible history entry ─────────────────────────── */

function HistoryEntry({ entry, index, totalEntries }) {
  const [open, setOpen] = useState(index === 0); // most recent is open by default
  const isLatest = index === 0;

  return (
    <div className={`border ${isLatest ? 'border-brand-200 bg-brand-50/30' : 'border-gray-200 bg-white'} rounded-2xl overflow-hidden shadow-sm`}>
      {/* Header — always visible */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3 hover:bg-gray-50 transition"
      >
        <div className="flex items-center gap-3">
          <span className={`w-7 h-7 flex items-center justify-center rounded-full text-xs font-bold ${isLatest ? 'bg-brand-500 text-white' : 'bg-gray-200 text-gray-600'}`}>
            {totalEntries - index}
          </span>
          <div className="text-left">
            <div className="text-sm font-medium text-gray-900 truncate max-w-md">
              {entry.prompt.length > 80 ? entry.prompt.slice(0, 80) + '…' : entry.prompt}
            </div>
            <div className="text-[10px] text-gray-400">{entry.timestamp}</div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <ExtractionSummary summary={entry.summary} />
          <svg className={`w-4 h-4 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Body — collapsible */}
      {open && (
        <div className="px-5 pb-5 space-y-3 border-t border-gray-100">
          {/* Prompt */}
          <div className="mt-3 bg-gray-50 rounded-xl p-3 text-sm text-gray-700 border border-gray-100">
            <span className="text-xs text-gray-400 block mb-1">Prompt</span>
            {entry.prompt}
          </div>

          {/* Extracted segments */}
          {entry.segments && entry.segments.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Extracted Segments</div>
              {entry.segments.map((seg, si) => (
                <div key={si} className="bg-white border border-gray-100 rounded-lg p-3 mb-2">
                  <div className="text-xs text-gray-400 mb-1">Segment {si + 1}</div>
                  <div className="text-sm text-gray-700 italic mb-2">"{seg.text}"</div>
                  <div className="flex flex-wrap gap-1.5">
                    {seg.behaviors.map((b, bi) => (
                      <span key={bi} className="px-2 py-0.5 bg-purple-50 border border-purple-200 rounded-full text-[10px] text-purple-700 font-medium">
                        {b.description} <span className="text-purple-400">(conf: {b.confidence?.toFixed(2)})</span>
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Storage flow */}
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Storage Flow</div>
            <div className="space-y-2">
              {entry.flow.map((f, fi) => (
                <BehaviorFlowCard key={fi} flow={f} index={fi} />
              ))}
              {entry.flow.length === 0 && (
                <div className="text-sm text-gray-400 italic">No behaviors processed in storage flow.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════ MAIN PAGE COMPONENT ═══════════════════════════ */

export default function BehaviorExtractionPage() {
  const { userId, sessionId } = useSession();

  const [prompt, setPrompt]       = useState('');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [history, setHistory]     = useState([]); // newest first
  const textRef                   = useRef(null);

  /* ─── Submit extraction ─── */
  const handleExtract = async () => {
    if (!prompt.trim() || !userId.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await api.extractBehaviors({
        user_id: userId,
        session_id: sessionId || 'default',
        prompt: prompt.trim(),
      });

      if (!res.success) throw new Error(res.error || 'Extraction failed');

      const pipeline = res.pipeline;
      const entry = {
        prompt: prompt.trim(),
        timestamp: new Date().toLocaleString(),
        segments: pipeline.extraction.segments || [],
        flow: pipeline.storage.flow || [],
        summary: pipeline.storage.summary || {
          total_extracted: 0,
          total_stored: 0,
          total_reinforced: 0,
          total_conflicts: 0,
          total_pruned: 0,
        },
      };

      setHistory(prev => [entry, ...prev]);
      setPrompt('');

      // Auto-focus back to textarea
      setTimeout(() => textRef.current?.focus(), 100);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleExtract();
  };

  /* ─── Derive a behavior history map (to show credibility over time) ─── */
  const behaviorTimeline = React.useMemo(() => {
    // Walk history in chronological order (oldest first) and build a map
    // key: matched_behavior_id or stored_behavior_id → array of snapshots
    const map = {};
    const chronological = [...history].reverse();
    chronological.forEach((entry, roundIdx) => {
      (entry.flow || []).forEach(f => {
        const id = f.matched_behavior_id || f.stored_behavior_id;
        if (!id) return;
        if (!map[id]) map[id] = [];
        map[id].push({
          round: roundIdx + 1,
          action: f.action,
          credibility: f.credibility,
          reinforcement_info: f.reinforcement_info,
          conflict_info: f.conflict_info,
          description: f.behavior_description,
          prompt: entry.prompt,
          timestamp: entry.timestamp,
        });
      });
    });
    return map;
  }, [history]);

  /* Count unique behaviors tracked */
  const trackedBehaviorIds = Object.keys(behaviorTimeline);

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Behavior Extraction</h2>
        <p className="text-sm text-gray-400 mt-1">
          Extract behaviors from prompts and observe the full storage lifecycle —
          new saves, reinforcements, conflicts & auto-resolution.
        </p>
      </div>

      {/* ──── Input section ──── */}
      <Card>
        <div className="space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">💬</span>
            <span className="text-sm font-semibold text-gray-700">User Prompt</span>
          </div>

          <textarea
            ref={textRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            className="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-800 placeholder-gray-400 focus:ring-2 focus:ring-brand-100 focus:border-brand-400 outline-none transition resize-none"
            placeholder="e.g. I really enjoy drinking black coffee every morning…"
          />

          <div className="flex items-center justify-between">
            <span className="text-[10px] text-gray-400">Press ⌘+Enter / Ctrl+Enter to submit</span>
            <button
              onClick={handleExtract}
              disabled={loading || !prompt.trim() || !userId.trim()}
              className="px-5 py-2 bg-brand-600 text-white text-sm font-semibold rounded-xl shadow hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.37 0 0 5.37 0 12h4z" /></svg>
                  Extracting…
                </>
              ) : (
                <>🧬 Extract Behaviors</>
              )}
            </button>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>
      </Card>

      {/* ──── Behavior Lifecycle Tracker ──── */}
      {trackedBehaviorIds.length > 0 && (
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <span className="text-lg">🔍</span>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Behavior Lifecycle Tracker</h3>
              <p className="text-xs text-gray-400">Credibility changes across extraction rounds for tracked behaviors</p>
            </div>
            <Badge className="bg-brand-50 text-brand-700 border border-brand-200 ml-auto">
              {trackedBehaviorIds.length} tracked
            </Badge>
          </div>

          <div className="space-y-3">
            {trackedBehaviorIds.map(id => {
              const snapshots = behaviorTimeline[id];
              const latest = snapshots[snapshots.length - 1];
              return (
                <LifecycleBehaviorRow key={id} behaviorId={id} snapshots={snapshots} latest={latest} />
              );
            })}
          </div>
        </Card>
      )}

      {/* ──── Extraction history (collapsible) ──── */}
      {history.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">📜</span>
            <h3 className="text-lg font-semibold text-gray-900">Extraction History</h3>
            <Badge className="bg-gray-100 text-gray-600 border border-gray-200">
              {history.length} round{history.length !== 1 ? 's' : ''}
            </Badge>
          </div>

          <div className="space-y-3">
            {history.map((entry, i) => (
              <HistoryEntry key={i} entry={entry} index={i} totalEntries={history.length} />
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {history.length === 0 && !loading && (
        <div className="text-center py-16">
          <div className="text-5xl mb-4">🧬</div>
          <h3 className="text-lg font-semibold text-gray-700 mb-1">No extractions yet</h3>
          <p className="text-sm text-gray-400 max-w-md mx-auto">
            Enter a prompt above to extract behaviors. Send multiple prompts with overlapping behaviors 
            to see reinforcement, conflict detection, and auto-resolution in action.
          </p>
        </div>
      )}
    </div>
  );
}

/* ──────────── Lifecycle row — shows a behavior across rounds ──────────── */

function LifecycleBehaviorRow({ behaviorId, snapshots, latest }) {
  const [expanded, setExpanded] = useState(false);

  // Color based on latest action
  const latestStyle = getActionStyle(latest.action);

  return (
    <div className={`border ${latestStyle.border} rounded-xl overflow-hidden`}>
      {/* Summary row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={`w-full flex items-center justify-between px-4 py-3 ${latestStyle.bg} hover:brightness-95 transition`}
      >
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-lg">{latestStyle.icon}</span>
          <div className="text-left min-w-0">
            <div className="text-sm font-medium text-gray-900 truncate max-w-sm">
              "{latest.description}"
            </div>
            <div className="text-[10px] text-gray-400 font-mono">{behaviorId.slice(0, 12)}…</div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <Badge className={`${latestStyle.bg} ${latestStyle.text} border ${latestStyle.border}`}>
            {latestStyle.label}
          </Badge>
          <span className="text-xs text-gray-500">{snapshots.length} interaction{snapshots.length !== 1 ? 's' : ''}</span>

          {/* Show credibility change if we have reinforcement data */}
          {snapshots.length > 1 && latest.reinforcement_info?.credibility_after != null && (
            <div className="text-right">
              <div className="text-xs font-bold text-emerald-600">
                {latest.reinforcement_info.credibility_before?.toFixed(4)} → {latest.reinforcement_info.credibility_after?.toFixed(4)}
              </div>
              <div className="text-[10px] text-emerald-500">+{latest.reinforcement_info.credibility_boost?.toFixed(4)}</div>
            </div>
          )}

          <svg className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Timeline detail */}
      {expanded && (
        <div className="px-4 py-3 border-t border-gray-100 bg-white">
          <div className="relative pl-6">
            {/* Vertical timeline line */}
            <div className="absolute left-2 top-2 bottom-2 w-px bg-gray-200" />

            {snapshots.map((snap, si) => {
              const sStyle = getActionStyle(snap.action);
              return (
                <div key={si} className="relative mb-4 last:mb-0">
                  {/* Dot */}
                  <div className={`absolute -left-4 top-1 w-3 h-3 rounded-full border-2 ${sStyle.border} ${sStyle.bg}`} />

                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm">{sStyle.icon}</span>
                        <Badge className={`${sStyle.bg} ${sStyle.text} border ${sStyle.border}`}>
                          {sStyle.label}
                        </Badge>
                        <span className="text-[10px] text-gray-400">Round {snap.round}</span>
                      </div>
                      <div className="text-xs text-gray-500 mt-0.5 italic truncate max-w-md">
                        Prompt: "{snap.prompt.slice(0, 60)}{snap.prompt.length > 60 ? '…' : ''}"
                      </div>
                      <div className="text-[10px] text-gray-400">{snap.timestamp}</div>

                      {/* Reinforcement details inline */}
                      {snap.action === 'DUPLICATE_REINFORCED' && snap.reinforcement_info && (
                        <div className="flex items-center gap-2 mt-1 text-xs">
                          <span className="text-gray-500">Credibility:</span>
                          <span className="font-medium text-gray-700">{snap.reinforcement_info.credibility_before?.toFixed(4)}</span>
                          <span className="text-gray-400">→</span>
                          <span className="font-bold text-emerald-600">{snap.reinforcement_info.credibility_after?.toFixed(4)}</span>
                          <span className="text-emerald-500">(+{snap.reinforcement_info.credibility_boost?.toFixed(4)})</span>
                        </div>
                      )}

                      {/* Conflict info inline */}
                      {snap.conflict_info && (
                        <div className="text-xs text-red-600 mt-1">
                          {snap.conflict_info.conflict_type} — {snap.conflict_info.resolution || 'pending'}
                        </div>
                      )}
                    </div>

                    <div className="text-right shrink-0">
                      <div className="text-xs font-bold text-gray-800">{snap.credibility.toFixed(4)}</div>
                      <div className="text-[10px] text-gray-400">credibility</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
