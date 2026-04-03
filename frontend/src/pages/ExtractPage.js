import React, { useState, useRef } from 'react';
import { useSession } from '../SessionContext';
import { api } from '../api';

/* ──────────────────────────────── helpers ──────────────────────────────── */

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

function SectionHeader({ icon, title, subtitle, count }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <span className="text-xl">{icon}</span>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          {subtitle && <p className="text-xs text-gray-400">{subtitle}</p>}
        </div>
      </div>
      {count !== undefined && (
        <Badge className="bg-brand-50 text-brand-700 border border-brand-200">
          {count} item{count !== 1 ? 's' : ''}
        </Badge>
      )}
    </div>
  );
}

function ProgressBar({ value, max, color = 'bg-brand-500' }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden mt-2">
      <div
        className={`h-full rounded-full transition-all duration-1000 ease-out ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function StatBox({ label, value, color = 'text-gray-900', icon }) {
  return (
    <div className="bg-gray-50 rounded-xl p-3 text-center border border-gray-100">
      <div className="text-2xl mb-1">{icon}</div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="text-xs text-gray-400 mt-0.5">{label}</div>
    </div>
  );
}

/* ─────────────────────── Pipeline Step Components ─────────────────────── */

function ExtractionPhase({ data }) {
  return (
    <Card delay={100}>
      <SectionHeader
        icon="🔬"
        title="Extraction Phase"
        subtitle={`${data.extraction_time_ms?.toFixed(0)}ms • ${data.total_segments} segment(s)`}
        count={data.total_behaviors_extracted}
      />

      {data.standalone_query && (
        <div className="mb-4 bg-brand-50 border border-brand-100 rounded-xl p-3">
          <p className="text-xs text-brand-600 font-medium mb-1">📝 Standalone Query (Enriched)</p>
          <p className="text-sm text-gray-700 italic">"{data.standalone_query}"</p>
        </div>
      )}

      {data.required_intents && data.required_intents.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
          <span className="text-xs text-gray-400 mr-1">Required Intents:</span>
          {data.required_intents.map((intent, i) => (
            <Badge key={i} className="bg-purple-50 text-purple-700 border border-purple-200">
              {intent}
            </Badge>
          ))}
        </div>
      )}

      <div className="space-y-3">
        {data.segments?.map((seg, segIdx) => (
          <div key={segIdx} className="border border-gray-100 rounded-xl p-4 bg-gray-50/50">
            <p className="text-xs text-gray-400 mb-2">Segment {segIdx + 1}: <span className="text-gray-600">"{seg.text}"</span></p>
            <div className="space-y-2">
              {seg.behaviors?.map((beh, behIdx) => (
                <div key={behIdx} className="bg-white rounded-lg p-3 border border-gray-100 shadow-sm">
                  <p className="text-sm text-gray-800 mb-2">{beh.description}</p>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <Badge className="bg-emerald-50 text-emerald-700 border border-emerald-200">
                      Confidence: {(beh.confidence * 100).toFixed(0)}%
                    </Badge>
                    <Badge className="bg-blue-50 text-blue-700 border border-blue-200">
                      Clarity: {(beh.clarity * 100).toFixed(0)}%
                    </Badge>
                    <Badge className="bg-purple-50 text-purple-700 border border-purple-200">
                      Strength: {(beh.linguistic_strength * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  {beh.canonical && (
                    <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
                      <span className="px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                        intent: {beh.canonical.intent}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-cyan-50 text-cyan-700 border border-cyan-200">
                        target: {beh.canonical.target}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">
                        context: {beh.canonical.context}
                      </span>
                      <span className={`px-2 py-0.5 rounded border ${
                        beh.canonical.polarity === 'POSITIVE'
                          ? 'bg-green-50 text-green-700 border-green-200'
                          : 'bg-red-50 text-red-700 border-red-200'
                      }`}>
                        polarity: {beh.canonical.polarity}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function RetrievalPhase({ data }) {
  return (
    <Card delay={200}>
      <SectionHeader
        icon="🔍"
        title="3D Hybrid Retrieval"
        subtitle={`Threshold: ${data.distance_threshold}`}
        count={data.total_candidates}
      />

      {data.query_used && (
        <div className="mb-4 bg-cyan-50 border border-cyan-100 rounded-xl p-3">
          <p className="text-xs text-cyan-700 font-medium mb-1">🎯 Search Query</p>
          <p className="text-sm text-gray-700 italic">"{data.query_used}"</p>
        </div>
      )}

      <div className="mb-3 flex gap-2 text-sm">
        <Badge className="bg-emerald-50 text-emerald-700 border border-emerald-200">
          ✅ Within threshold: {data.within_threshold}
        </Badge>
        <Badge className="bg-gray-100 text-gray-600 border border-gray-200">
          Total candidates: {data.total_candidates}
        </Badge>
      </div>

      {data.results?.length > 0 ? (
        <div className="space-y-2">
          {data.results.map((r, i) => (
            <div
              key={i}
              className={`rounded-xl p-3 border transition-all ${
                r.within_threshold
                  ? 'bg-emerald-50/60 border-emerald-200'
                  : 'bg-gray-50 border-gray-200 opacity-60'
              }`}
            >
              <div className="flex items-start justify-between mb-1">
                <p className="text-sm text-gray-800 flex-1">{r.behavior_text}</p>
                <div className="flex gap-2 ml-3 shrink-0">
                  <Badge className={r.within_threshold ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-600'}>
                    dist: {r.distance}
                  </Badge>
                  <Badge className="bg-blue-50 text-blue-700">
                    cred: {r.credibility}
                  </Badge>
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5 text-xs mt-1">
                <span className="px-2 py-0.5 rounded bg-indigo-50 text-indigo-600">{r.intent}</span>
                <span className="px-2 py-0.5 rounded bg-cyan-50 text-cyan-600">{r.target}</span>
                <span className={`px-2 py-0.5 rounded ${
                  r.polarity === 'POSITIVE' ? 'bg-green-50 text-green-600' : 'bg-red-50 text-red-600'
                }`}>{r.polarity}</span>
                <span className="px-2 py-0.5 rounded bg-gray-100 text-gray-500">×{r.reinforcement_count}</span>
              </div>
              <ProgressBar
                value={1 - r.distance}
                max={1}
                color={r.within_threshold ? 'bg-emerald-500' : 'bg-gray-300'}
              />
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-400 italic">No similar behaviors found in database</p>
      )}
    </Card>
  );
}

function GraphExpansionPhase({ data }) {
  return (
    <Card delay={300}>
      <SectionHeader
        icon="🕸️"
        title="Knowledge Graph Expansion"
        subtitle="1-hop co-occurrence walk"
        count={data.total_expanded}
      />

      {data.seed_behavior_ids?.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-gray-400 mb-1">Seed behavior IDs:</p>
          <div className="flex flex-wrap gap-1">
            {data.seed_behavior_ids.map((id, i) => (
              <Badge key={i} className="bg-purple-50 text-purple-600 border border-purple-200 font-mono text-[10px]">
                {id.substring(0, 8)}…
              </Badge>
            ))}
          </div>
        </div>
      )}

      {data.expanded_behaviors?.length > 0 ? (
        <div className="space-y-2">
          {data.expanded_behaviors.map((b, i) => (
            <div key={i} className="bg-purple-50/60 border border-purple-200 rounded-xl p-3">
              <p className="text-sm text-gray-800">{b.behavior_text}</p>
              <div className="flex gap-2 mt-1 text-xs">
                {b.edge_weight && (
                  <Badge className="bg-purple-100 text-purple-700">
                    weight: {typeof b.edge_weight === 'number' ? b.edge_weight.toFixed(3) : b.edge_weight}
                  </Badge>
                )}
                {b.credibility && (
                  <Badge className="bg-blue-50 text-blue-700">
                    cred: {typeof b.credibility === 'number' ? b.credibility.toFixed(3) : b.credibility}
                  </Badge>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-400 italic">No graph neighbors discovered</p>
      )}
    </Card>
  );
}

function StoragePhase({ data }) {
  const { flow, summary } = data;

  return (
    <Card delay={400}>
      <SectionHeader
        icon="💾"
        title="Storage & Decision Phase"
        subtitle="Per-behavior flow tracking"
        count={summary.total_extracted}
      />

      <div className="grid grid-cols-5 gap-2 mb-4">
        <StatBox label="Extracted" value={summary.total_extracted} icon="📄" color="text-gray-900" />
        <StatBox label="Stored" value={summary.total_stored} icon="💾" color="text-emerald-600" />
        <StatBox label="Reinforced" value={summary.total_reinforced} icon="🔁" color="text-blue-600" />
        <StatBox label="Conflicts" value={summary.total_conflicts} icon="⚡" color="text-red-600" />
        <StatBox label="Pruned" value={summary.total_pruned} icon="✂️" color="text-gray-400" />
      </div>

      <div className="space-y-3">
        {flow?.map((item, i) => {
          const style = getActionStyle(item.action);
          return (
            <div
              key={i}
              className={`${style.bg} border ${style.border} rounded-xl p-4 animate-slide-in`}
              style={{ animationDelay: `${500 + i * 80}ms` }}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{style.icon}</span>
                  <Badge className={`${style.bg} ${style.text} border ${style.border}`}>
                    {style.label}
                  </Badge>
                </div>
                <Badge className="bg-gray-100 text-gray-500">
                  cred: {item.credibility}
                </Badge>
              </div>

              <p className="text-sm text-gray-800 mb-2">"{item.behavior_description}"</p>

              {item.canonical && (
                <div className="flex flex-wrap gap-1.5 text-xs mb-2">
                  <span className="px-2 py-0.5 rounded bg-indigo-50 text-indigo-600 border border-indigo-100">
                    intent: {item.canonical.intent}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-cyan-50 text-cyan-600 border border-cyan-100">
                    target: {item.canonical.target}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-100">
                    context: {item.canonical.context}
                  </span>
                  <span className={`px-2 py-0.5 rounded border ${
                    item.canonical.polarity === 'POSITIVE'
                      ? 'bg-green-50 text-green-600 border-green-100'
                      : 'bg-red-50 text-red-600 border-red-100'
                  }`}>
                    {item.canonical.polarity}
                  </span>
                </div>
              )}

              {item.matched_behavior_text && (
                <div className="bg-gray-50 rounded-lg p-2 mb-2 border border-gray-100">
                  <p className="text-xs text-gray-400 mb-0.5">Matched existing behavior:</p>
                  <p className="text-xs text-gray-700">"{item.matched_behavior_text}"</p>
                  <div className="flex gap-2 mt-1">
                    {item.distance !== null && (
                      <Badge className="bg-gray-100 text-gray-500 text-[10px]">
                        distance: {item.distance}
                      </Badge>
                    )}
                    <Badge className="bg-gray-100 text-gray-500 text-[10px] font-mono">
                      ID: {item.matched_behavior_id?.substring(0, 8)}…
                    </Badge>
                  </div>
                </div>
              )}

              {item.conflict_info && (
                <div className="bg-red-50 border border-red-100 rounded-lg p-2 mb-2">
                  <p className="text-xs text-red-600 font-medium mb-1">⚡ Conflict Details</p>
                  <div className="grid grid-cols-2 gap-1 text-xs text-gray-500">
                    <span>Type: <span className="text-red-600 font-medium">{item.conflict_info.conflict_type}</span></span>
                    <span>Resolution: <span className="text-amber-600 font-medium">{item.conflict_info.resolution}</span></span>
                    {item.conflict_info.existing_polarity && (
                      <span>Old polarity: <span className="text-gray-700">{item.conflict_info.existing_polarity}</span></span>
                    )}
                    {item.conflict_info.new_polarity && (
                      <span>New polarity: <span className="text-gray-700">{item.conflict_info.new_polarity}</span></span>
                    )}
                  </div>
                  {item.conflict_info.explanation && (
                    <p className="text-xs text-gray-500 mt-1 italic">{item.conflict_info.explanation}</p>
                  )}
                  {item.conflict_info.llm_analysis && (
                    <div className="mt-1 bg-white rounded p-1.5 border border-gray-100">
                      <p className="text-[10px] text-gray-400">LLM Analysis:</p>
                      <p className="text-xs text-gray-700">{item.conflict_info.llm_analysis}</p>
                    </div>
                  )}
                </div>
              )}

              {item.details && (
                <p className="text-xs text-gray-400 italic">💡 {item.details}</p>
              )}

              {item.stored_behavior_id && (
                <p className="text-[10px] text-gray-400 mt-1 font-mono">
                  Stored as: {item.stored_behavior_id}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function RelatedBehaviors({ behaviors }) {
  return (
    <Card delay={500}>
      <SectionHeader
        icon="🎯"
        title="Final Related Behaviors"
        subtitle="Returned to consumer service"
        count={behaviors?.length || 0}
      />
      {behaviors?.length > 0 ? (
        <div className="space-y-2">
          {behaviors.map((text, i) => (
            <div key={i} className="flex items-start gap-2 bg-brand-50 border border-brand-100 rounded-xl p-3">
              <span className="text-brand-500 font-mono text-xs mt-0.5">{i + 1}</span>
              <p className="text-sm text-gray-700">{text}</p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-400 italic">No related behaviors found</p>
      )}
    </Card>
  );
}

/* ─────────────────────────── Main Page ─────────────────────────── */

export default function ExtractPage() {
  const { userId, sessionId } = useSession();
  const [prompt, setPrompt] = useState('');
  const [history, setHistory] = useState([]);
  const [historyInput, setHistoryInput] = useState('');
  const [historyRole, setHistoryRole] = useState('user');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activePhase, setActivePhase] = useState(null);
  const resultRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setActivePhase('extraction');

    try {
      const data = await api.extractDetailed({
        user_id: userId,
        session_id: sessionId,
        prompt: prompt.trim(),
        recent_history: history.length > 0 ? history : undefined,
      });

      if (data.success) {
        setResult(data);
        setTimeout(() => setActivePhase('retrieval'), 600);
        setTimeout(() => setActivePhase('graph'), 1200);
        setTimeout(() => setActivePhase('storage'), 1800);
        setTimeout(() => setActivePhase('done'), 2400);
      } else {
        setError(data.error || 'Extraction failed');
      }
    } catch (err) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
      if (resultRef.current) {
        resultRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  const addHistoryMessage = () => {
    if (!historyInput.trim()) return;
    setHistory(prev => [...prev, { role: historyRole, text: historyInput.trim() }]);
    setHistoryInput('');
  };

  const removeHistoryMessage = (idx) => {
    setHistory(prev => prev.filter((_, i) => i !== idx));
  };

  const pipeline = result?.pipeline;

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-1">
          🔬 Behavior Extraction Pipeline
        </h1>
        <p className="text-sm text-gray-500">
          Extract behaviors, search related behaviors, expand knowledge graph, and store with conflict detection — all visualized.
        </p>
      </div>

      {/* Input Form */}
      <Card className="mb-6">
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-xs font-medium text-gray-500 mb-1">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              className="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-800 focus:border-brand-500 focus:ring-2 focus:ring-brand-100 outline-none transition resize-none"
              placeholder="Enter a natural language prompt to analyze... e.g. 'I always prefer dark mode when coding, and I use VS Code with Vim keybindings'"
            />
          </div>

          {/* Conversation History */}
          <div className="mb-4">
            <label className="block text-xs font-medium text-gray-500 mb-2">
              Conversation History (optional)
            </label>
            {history.length > 0 && (
              <div className="space-y-1 mb-2 max-h-32 overflow-y-auto">
                {history.map((msg, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <Badge className={msg.role === 'user' ? 'bg-blue-50 text-blue-700' : 'bg-green-50 text-green-700'}>
                      {msg.role}
                    </Badge>
                    <span className="text-gray-600 flex-1 truncate">{msg.text}</span>
                    <button
                      type="button"
                      onClick={() => removeHistoryMessage(i)}
                      className="text-red-400 hover:text-red-600 text-xs"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <select
                value={historyRole}
                onChange={(e) => setHistoryRole(e.target.value)}
                className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 outline-none"
              >
                <option value="user">user</option>
                <option value="assistant">assistant</option>
              </select>
              <input
                type="text"
                value={historyInput}
                onChange={(e) => setHistoryInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addHistoryMessage())}
                className="flex-1 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 outline-none focus:border-brand-400"
                placeholder="Add a history message..."
              />
              <button
                type="button"
                onClick={addHistoryMessage}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm text-gray-600 transition border border-gray-200"
              >
                Add
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !prompt.trim()}
            className={`w-full py-3 rounded-xl font-semibold text-sm transition-all duration-300 ${
              loading
                ? 'bg-brand-100 text-brand-400 cursor-wait'
                : 'bg-brand-600 hover:bg-brand-700 text-white shadow-md shadow-brand-200 hover:shadow-brand-300'
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                Processing Pipeline...
              </span>
            ) : (
              '🚀 Run Extraction Pipeline'
            )}
          </button>
        </form>
      </Card>

      {/* Error */}
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-xl p-4 animate-slide-in">
          <p className="text-red-700 font-medium">❌ Error</p>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Pipeline Progress Indicator */}
      {loading && (
        <div className="mb-6">
          <div className="flex items-center gap-3">
            {['extraction', 'retrieval', 'graph', 'storage'].map((phase, i) => (
              <React.Fragment key={phase}>
                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-500 ${
                  activePhase === phase
                    ? 'bg-brand-100 text-brand-700 animate-pulse'
                    : 'bg-gray-100 text-gray-400'
                }`}>
                  <span>{['🔬', '🔍', '🕸️', '💾'][i]}</span>
                  {phase.charAt(0).toUpperCase() + phase.slice(1)}
                </div>
                {i < 3 && <div className="w-8 h-px bg-gray-200" />}
              </React.Fragment>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {pipeline && (
        <div ref={resultRef} className="space-y-6 animate-fade-in">
          {/* Pipeline visualization header */}
          <div className="flex items-center gap-3 text-sm flex-wrap">
            {['🔬 Extract', '🔍 Retrieve', '🕸️ Expand', '💾 Store', '🎯 Output'].map((phase, i) => (
              <React.Fragment key={i}>
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-brand-50 text-brand-700 border border-brand-200 text-xs font-medium">
                  {phase}
                </div>
                {i < 4 && (
                  <svg className="w-4 h-4 text-brand-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                )}
              </React.Fragment>
            ))}
          </div>

          <ExtractionPhase data={pipeline.extraction} />
          <RetrievalPhase data={pipeline.retrieval} />
          <GraphExpansionPhase data={pipeline.graph_expansion} />
          <StoragePhase data={pipeline.storage} />
          <RelatedBehaviors behaviors={pipeline.related_behaviors} />
        </div>
      )}
    </div>
  );
}
