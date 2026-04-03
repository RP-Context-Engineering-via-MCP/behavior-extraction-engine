import React, { useState } from 'react';
import { useSession } from '../SessionContext';
import { api } from '../api';

/* ──────────────────────── shared helpers ──────────────────────── */

function Badge({ children, className = '' }) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${className}`}>
      {children}
    </span>
  );
}

function Card({ children, className = '' }) {
  return (
    <div className={`bg-white border border-gray-200 rounded-2xl p-5 shadow-sm ${className}`}>
      {children}
    </div>
  );
}

function formatTs(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString();
}

/* ──────────────────────── status / type maps ──────────────────────── */

const STATUS_STYLES = {
  PENDING:        { bg: 'bg-amber-50',    border: 'border-amber-200',   text: 'text-amber-700',   icon: '⏳', label: 'Pending' },
  USER_RESOLVED:  { bg: 'bg-emerald-50',  border: 'border-emerald-200', text: 'text-emerald-700', icon: '✅', label: 'User Resolved' },
  AUTO_RESOLVED:  { bg: 'bg-blue-50',     border: 'border-blue-200',    text: 'text-blue-700',    icon: '🤖', label: 'Auto-Resolved' },
};

const CHOICE_LABELS = {
  OLD_WINS:     { icon: '🛡️', label: 'Old Wins', desc: 'Existing behavior kept, new invalidated' },
  NEW_WINS:     { icon: '🔄', label: 'New Wins', desc: 'New behavior kept, old superseded' },
  BOTH_CORRECT: { icon: '🤝', label: 'Both Valid', desc: 'Both behaviors kept as valid' },
};

/* ──────────────── Auto-Resolution Info Banner ──────────────── */

function AutoResolutionBanner({ conflict }) {
  const gap = Math.abs(
    (conflict.behavior_1_reinforcement_count || 0) -
    (conflict.behavior_2_reinforcement_count || 0)
  );

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">🤖</span>
        <h4 className="text-sm font-semibold text-blue-800">Auto-Resolution Applied</h4>
      </div>
      <p className="text-xs text-blue-700 mb-3">
        This conflict was automatically resolved because the reinforcement gap between the two
        behaviors reached the threshold (≥3). When one behavior is reinforced significantly more
        than its conflicting counterpart, the system auto-resolves in favor of the stronger behavior.
      </p>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-white rounded-lg p-2 border border-blue-100">
          <div className="text-lg font-bold text-blue-700">{gap}</div>
          <div className="text-[10px] text-blue-500">Reinforcement Gap</div>
        </div>
        <div className="bg-white rounded-lg p-2 border border-blue-100">
          <div className="text-lg font-bold text-blue-700">
            {CHOICE_LABELS[conflict.resolution_choice]?.icon || '—'} {conflict.resolution_choice?.replace(/_/g, ' ')}
          </div>
          <div className="text-[10px] text-blue-500">Auto-Chosen Resolution</div>
        </div>
        <div className="bg-white rounded-lg p-2 border border-blue-100">
          <div className="text-lg font-bold text-blue-700">{formatTs(conflict.resolved_at)}</div>
          <div className="text-[10px] text-blue-500">Resolved At</div>
        </div>
      </div>
      <div className="mt-3 bg-blue-100/50 rounded-lg p-2">
        <p className="text-[10px] text-blue-600 font-medium mb-1">How Auto-Resolution Works:</p>
        <ul className="text-[10px] text-blue-600 space-y-0.5 list-disc list-inside">
          <li>When a <strong>CONFLICT_DETECTED</strong> behavior gets reinforced, the system checks the gap</li>
          <li>If <code className="bg-blue-200/50 px-1 rounded">reinforcement_gap ≥ 3</code> OR <code className="bg-blue-200/50 px-1 rounded">credibility_gap ≥ 0.15</code>, auto-resolve triggers</li>
          <li>The behavior with higher credibility wins; loser is set to SUPERSEDED with credibility → 0.0</li>
          <li>If the conflict expires (30 days), it auto-resolves as BOTH_CORRECT</li>
        </ul>
      </div>
    </div>
  );
}

/* ──────────────── Behavior Side Card ──────────────── */

function BehaviorCard({ label, labelColor, behavior_text, polarity, credibility, state, reinforcement_count, behavior_id, intent, isWinner, isLoser }) {
  return (
    <div className={`rounded-xl p-4 border transition-all ${
      isWinner ? 'bg-emerald-50/80 border-emerald-200 ring-2 ring-emerald-200' :
      isLoser  ? 'bg-red-50/50 border-red-200 opacity-75' :
                 'bg-gray-50 border-gray-200'
    }`}>
      <div className="flex items-center gap-1.5 mb-2">
        {isWinner && <span className="text-xs">🏆</span>}
        {isLoser && <span className="text-xs">❌</span>}
        <p className={`text-[10px] uppercase tracking-wider font-bold ${labelColor}`}>{label}</p>
      </div>
      <p className="text-sm text-gray-800 mb-2 font-medium">{behavior_text || '—'}</p>
      <div className="flex flex-wrap gap-1.5 text-xs">
        {polarity && (
          <Badge className={polarity === 'POSITIVE' ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}>
            {polarity}
          </Badge>
        )}
        {intent && (
          <Badge className="bg-indigo-50 text-indigo-700 border border-indigo-200">
            {intent}
          </Badge>
        )}
        <Badge className={`border ${
          state === 'ACTIVE'     ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
          state === 'SUPERSEDED' ? 'bg-red-50 text-red-600 border-red-200' :
          state === 'FLAGGED'    ? 'bg-amber-50 text-amber-700 border-amber-200' :
                                   'bg-gray-100 text-gray-500 border-gray-200'
        }`}>
          {state || 'UNKNOWN'}
        </Badge>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <div className="bg-white rounded-lg p-1.5 border border-gray-100 text-center">
          <div className={`text-sm font-bold ${credibility >= 0.5 ? 'text-emerald-600' : 'text-red-500'}`}>
            {typeof credibility === 'number' ? credibility.toFixed(3) : credibility ?? '—'}
          </div>
          <div className="text-[9px] text-gray-400">Credibility</div>
        </div>
        <div className="bg-white rounded-lg p-1.5 border border-gray-100 text-center">
          <div className="text-sm font-bold text-blue-600">{reinforcement_count ?? '—'}</div>
          <div className="text-[9px] text-gray-400">Reinforcements</div>
        </div>
      </div>
      {behavior_id && (
        <p className="text-[9px] text-gray-400 mt-2 font-mono">ID: {behavior_id}</p>
      )}
    </div>
  );
}

/* ──────────────── Resolution Options (inline per conflict) ──────────────── */

const RESOLUTION_OPTIONS = [
  {
    value: 'OLD_WINS',
    icon: '🛡️',
    title: 'Keep Existing',
    description: 'The original behavior is correct. Discard the new conflicting behavior. Old behavior gets reinforced, new gets credibility → 0.0.',
    color: 'border-blue-200 bg-blue-50 hover:bg-blue-100',
    activeColor: 'border-blue-500 bg-blue-100 ring-2 ring-blue-300',
    textColor: 'text-blue-700',
  },
  {
    value: 'NEW_WINS',
    icon: '🔄',
    title: 'Accept New',
    description: 'The new behavior is correct. Supersede the existing one. New behavior becomes ACTIVE, old gets superseded.',
    color: 'border-emerald-200 bg-emerald-50 hover:bg-emerald-100',
    activeColor: 'border-emerald-500 bg-emerald-100 ring-2 ring-emerald-300',
    textColor: 'text-emerald-700',
  },
  {
    value: 'BOTH_CORRECT',
    icon: '🤝',
    title: 'Both Are Valid',
    description: 'Both behaviors are valid in different contexts. Keep both as ACTIVE and reinforce both.',
    color: 'border-purple-200 bg-purple-50 hover:bg-purple-100',
    activeColor: 'border-purple-500 bg-purple-100 ring-2 ring-purple-300',
    textColor: 'text-purple-700',
  },
];

function ResolutionControls({ conflict, onResolved }) {
  const { userId } = useSession();
  const [selected, setSelected] = useState(null);
  const [resolving, setResolving] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleResolve = async () => {
    if (!selected) return;
    setResolving(true);
    setError(null);
    try {
      const data = await api.resolveConflict({
        conflict_id: conflict.conflict_id,
        user_id: userId.trim(),
        resolution_choice: selected,
      });
      if (data.success) {
        setResult(data.data);
        if (onResolved) onResolved(conflict.conflict_id, data.data);
      } else {
        setError(data.error || 'Resolution failed');
      }
    } catch (err) {
      setError(err.message || 'Network error');
    } finally {
      setResolving(false);
    }
  };

  /* ── Already resolved? Show outcome ── */
  if (result) {
    return (
      <div className="mt-4 animate-slide-in">
        <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🎉</span>
            <h4 className="text-sm font-semibold text-emerald-800">Resolution Applied Successfully</h4>
          </div>
          <div className="grid grid-cols-3 gap-2 mb-3">
            <div className="bg-white rounded-lg p-2 border border-emerald-100 text-center">
              <div className="text-sm font-bold text-emerald-700">{result.resolution_choice?.replace(/_/g, ' ')}</div>
              <div className="text-[10px] text-emerald-500">Choice</div>
            </div>
            <div className="bg-white rounded-lg p-2 border border-emerald-100 text-center">
              <div className="text-sm font-bold text-emerald-700">{result.resolution_status}</div>
              <div className="text-[10px] text-emerald-500">Status</div>
            </div>
            <div className="bg-white rounded-lg p-2 border border-emerald-100 text-center">
              <div className="text-sm font-bold text-emerald-700">{formatTs(result.resolved_at)}</div>
              <div className="text-[10px] text-emerald-500">Resolved At</div>
            </div>
          </div>

          {/* Updated behavior states after resolution */}
          {result.behaviors && result.behaviors.length > 0 && (
            <div>
              <p className="text-xs text-emerald-600 font-medium mb-2">Updated Behavior States:</p>
              <div className="space-y-2">
                {result.behaviors.map((b, i) => (
                  <div key={i} className="bg-white rounded-lg p-3 border border-emerald-100">
                    <div className="flex items-start justify-between mb-1">
                      <p className="text-sm text-gray-800">{b.behavior_text}</p>
                      <Badge className={`ml-2 ${
                        b.behavior_state === 'ACTIVE' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' :
                        b.behavior_state === 'SUPERSEDED' ? 'bg-red-50 text-red-600 border border-red-200' :
                        'bg-gray-100 text-gray-500 border border-gray-200'
                      }`}>
                        {b.behavior_state}
                      </Badge>
                    </div>
                    <div className="flex gap-3 text-xs text-gray-500 mt-1">
                      <span>Credibility: <strong className={b.credibility >= 0.5 ? 'text-emerald-600' : 'text-red-500'}>{typeof b.credibility === 'number' ? b.credibility.toFixed(3) : b.credibility}</strong></span>
                      <span>Last accessed: {formatTs(b.last_accessed_at)}</span>
                    </div>
                    <p className="text-[9px] text-gray-400 mt-1 font-mono">ID: {b.behavior_id}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="mt-4 border-t border-gray-100 pt-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-base">⚖️</span>
        <h4 className="text-sm font-semibold text-gray-800">Choose Resolution</h4>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-3">
        {RESOLUTION_OPTIONS.map(option => (
          <button
            key={option.value}
            onClick={() => setSelected(option.value)}
            className={`border-2 rounded-xl p-3 text-left transition-all duration-200 ${
              selected === option.value ? option.activeColor : option.color
            }`}
          >
            <div className="text-xl mb-1">{option.icon}</div>
            <div className={`font-semibold text-xs mb-0.5 ${option.textColor}`}>{option.title}</div>
            <p className="text-[10px] text-gray-500 leading-tight">{option.description}</p>
          </button>
        ))}
      </div>

      <button
        onClick={handleResolve}
        disabled={!selected || resolving}
        className={`w-full py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 ${
          !selected
            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
            : resolving
            ? 'bg-brand-100 text-brand-400 cursor-wait'
            : 'bg-brand-600 hover:bg-brand-700 text-white shadow-md shadow-brand-200'
        }`}
      >
        {resolving ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
            Resolving...
          </span>
        ) : selected ? (
          `✅ Apply: ${selected.replace(/_/g, ' ')}`
        ) : (
          'Select a resolution above'
        )}
      </button>

      {error && (
        <div className="mt-2 bg-red-50 border border-red-200 rounded-lg p-2">
          <p className="text-xs text-red-600">❌ {error}</p>
        </div>
      )}
    </div>
  );
}

/* ──────────────── Single Conflict Card ──────────────── */

function ConflictCard({ conflict, onResolved }) {
  const statusStyle = STATUS_STYLES[conflict.resolution_status] || STATUS_STYLES.PENDING;
  const isAutoResolved = conflict.resolution_status === 'AUTO_RESOLVED';
  const isResolved = conflict.resolution_status !== 'PENDING';

  /* Determine winner/loser for resolved conflicts */
  let b1Winner = false, b1Loser = false, b2Winner = false, b2Loser = false;
  if (isResolved && conflict.resolution_choice) {
    if (conflict.resolution_choice === 'OLD_WINS') {
      b1Winner = true; b2Loser = true;
    } else if (conflict.resolution_choice === 'NEW_WINS') {
      b2Winner = true; b1Loser = true;
    }
    // BOTH_CORRECT → neither winner nor loser styling
  }

  return (
    <Card className="animate-slide-in">
      {/* Header row */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge className={`${statusStyle.bg} ${statusStyle.text} border ${statusStyle.border}`}>
            {statusStyle.icon} {statusStyle.label}
          </Badge>
          {conflict.conflict_type && (
            <Badge className="bg-red-50 text-red-700 border border-red-200">
              {conflict.conflict_type.replace(/_/g, ' ')}
            </Badge>
          )}
          {conflict.resolution_choice && (
            <Badge className="bg-purple-50 text-purple-700 border border-purple-200">
              {CHOICE_LABELS[conflict.resolution_choice]?.icon} {conflict.resolution_choice.replace(/_/g, ' ')}
            </Badge>
          )}
        </div>
        <div className="text-right shrink-0 ml-3">
          <p className="text-[9px] text-gray-400 font-mono">{conflict.conflict_id}</p>
          <p className="text-[10px] text-gray-400">Created: {formatTs(conflict.created_at)}</p>
        </div>
      </div>

      {/* LLM Analysis */}
      {conflict.llm_analysis && (
        <div className="bg-amber-50 border border-amber-100 rounded-xl p-3 mb-4">
          <p className="text-[10px] text-amber-600 font-bold uppercase tracking-wider mb-1">🤖 LLM Conflict Analysis</p>
          <p className="text-xs text-gray-700">{conflict.llm_analysis}</p>
        </div>
      )}

      {/* Side-by-side behaviors */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <BehaviorCard
          label="Existing Behavior (B1)"
          labelColor="text-red-400"
          behavior_text={conflict.behavior_1_text}
          polarity={conflict.behavior_1_polarity}
          credibility={conflict.behavior_1_credibility}
          state={conflict.behavior_1_state}
          reinforcement_count={conflict.behavior_1_reinforcement_count}
          behavior_id={conflict.behavior_id_1}
          intent={conflict.behavior_1_intent}
          isWinner={b1Winner}
          isLoser={b1Loser}
        />
        <BehaviorCard
          label="New Behavior (B2)"
          labelColor="text-blue-400"
          behavior_text={conflict.behavior_2_text}
          polarity={conflict.behavior_2_polarity}
          credibility={conflict.behavior_2_credibility}
          state={conflict.behavior_2_state}
          reinforcement_count={conflict.behavior_2_reinforcement_count}
          behavior_id={conflict.behavior_id_2}
          intent={conflict.behavior_2_intent}
          isWinner={b2Winner}
          isLoser={b2Loser}
        />
      </div>

      {/* Similarity distance */}
      {conflict.similarity_distance != null && (
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[10px] text-gray-400">Similarity Distance:</span>
          <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-brand-500 rounded-full transition-all duration-500"
              style={{ width: `${Math.min((1 - conflict.similarity_distance) * 100, 100)}%` }}
            />
          </div>
          <span className="text-xs font-mono text-gray-500">{conflict.similarity_distance.toFixed(4)}</span>
        </div>
      )}

      {/* Auto-Resolution banner for AUTO_RESOLVED conflicts */}
      {isAutoResolved && <AutoResolutionBanner conflict={conflict} />}

      {/* Resolved summary for USER_RESOLVED */}
      {conflict.resolution_status === 'USER_RESOLVED' && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-3 mb-3">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm">✅</span>
            <h4 className="text-xs font-semibold text-emerald-800">User-Resolved</h4>
          </div>
          <p className="text-xs text-emerald-700">
            Resolved as <strong>{conflict.resolution_choice?.replace(/_/g, ' ')}</strong> on {formatTs(conflict.resolved_at)}
          </p>
        </div>
      )}

      {/* Resolution controls for PENDING conflicts */}
      {!isResolved && <ResolutionControls conflict={conflict} onResolved={onResolved} />}
    </Card>
  );
}

/* ──────────────── Main Page ──────────────── */

export default function ConflictsPage() {
  const { userId } = useSession();
  const [conflicts, setConflicts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('ALL');

  const fetchConflicts = async () => {
    if (!userId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getConflicts(userId.trim());
      if (data.success) {
        setConflicts(data.data?.conflicts || []);
      } else {
        setError(data.error || 'Failed to fetch conflicts');
      }
    } catch (err) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  /* After a conflict is resolved, update local state to reflect the new data */
  const handleResolved = (conflictId, resolutionData) => {
    setConflicts(prev => prev.map(c => {
      if (c.conflict_id !== conflictId) return c;

      const updatedBehaviors = resolutionData.behaviors || [];
      const b1 = updatedBehaviors.find(b => b.behavior_id === c.behavior_id_1);
      const b2 = updatedBehaviors.find(b => b.behavior_id === c.behavior_id_2);

      return {
        ...c,
        resolution_status: resolutionData.resolution_status || 'USER_RESOLVED',
        resolution_choice: resolutionData.resolution_choice,
        resolved_at: resolutionData.resolved_at,
        behavior_1_state: b1?.behavior_state || c.behavior_1_state,
        behavior_1_credibility: b1?.credibility ?? c.behavior_1_credibility,
        behavior_2_state: b2?.behavior_state || c.behavior_2_state,
        behavior_2_credibility: b2?.credibility ?? c.behavior_2_credibility,
      };
    }));
  };

  const filtered = statusFilter === 'ALL'
    ? conflicts
    : conflicts.filter(c => c.resolution_status === statusFilter);

  const statusCounts = conflicts.reduce((acc, c) => {
    acc[c.resolution_status] = (acc[c.resolution_status] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-1">
          ⚡ Conflicts & Resolution
        </h1>
        <p className="text-sm text-gray-500">
          View all detected behavior conflicts for a user. Pending conflicts can be resolved inline.
          Auto-resolved conflicts show the system's reasoning.
        </p>
      </div>

      {/* Fetch */}
      <Card className="mb-6">
        <div className="flex items-end gap-4">
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-500 mb-1">User ID</label>
            <p className="text-sm text-gray-700 bg-gray-50 border border-gray-200 rounded-xl px-4 py-2.5 font-mono">
              {userId || <span className="text-gray-400 italic font-sans">Set User ID in the header above</span>}
            </p>
          </div>
          <button
            onClick={fetchConflicts}
            disabled={loading || !userId.trim()}
            className={`px-6 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 shrink-0 ${
              loading
                ? 'bg-brand-100 text-brand-400 cursor-wait'
                : 'bg-brand-600 hover:bg-brand-700 text-white shadow-md shadow-brand-200'
            }`}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>
                Loading...
              </span>
            ) : '🔍 Fetch Conflicts'}
          </button>
        </div>
      </Card>

      {/* Error */}
      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-xl p-4 animate-slide-in">
          <p className="text-red-700 font-medium">❌ Error</p>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Results */}
      {conflicts.length > 0 && (
        <>
          {/* Summary row */}
          <div className="grid grid-cols-4 gap-3 mb-6">
            <Card className="text-center !p-4">
              <div className="text-3xl font-bold text-gray-900">{conflicts.length}</div>
              <div className="text-xs text-gray-400 mt-1">Total Conflicts</div>
            </Card>
            {Object.entries(statusCounts).map(([st, count]) => {
              const style = STATUS_STYLES[st] || STATUS_STYLES.PENDING;
              return (
                <Card key={st} className={`text-center !p-4 ${style.bg} ${style.border}`}>
                  <div className={`text-3xl font-bold ${style.text}`}>{count}</div>
                  <div className="text-xs text-gray-500 mt-1">{style.icon} {style.label}</div>
                </Card>
              );
            })}
          </div>

          {/* Filters */}
          <div className="flex gap-1 mb-5">
            {['ALL', 'PENDING', 'AUTO_RESOLVED', 'USER_RESOLVED'].map(s => (
              <button
                key={s}
                onClick={() => setStatusFilter(s)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  statusFilter === s
                    ? 'bg-brand-600 text-white shadow-sm'
                    : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                }`}
              >
                {s === 'ALL' ? `All (${conflicts.length})` : `${s.replace(/_/g, ' ')} (${statusCounts[s] || 0})`}
              </button>
            ))}
          </div>

          {/* Conflict list */}
          <div className="space-y-5">
            {filtered.map((conflict, i) => (
              <ConflictCard
                key={conflict.conflict_id || i}
                conflict={conflict}
                onResolved={handleResolved}
              />
            ))}

            {filtered.length === 0 && (
              <div className="text-center py-10 text-gray-400">
                <p className="text-4xl mb-2">🔍</p>
                <p className="text-sm">No conflicts match the selected filter</p>
              </div>
            )}
          </div>
        </>
      )}

      {/* Empty state */}
      {!loading && !error && conflicts.length === 0 && (
        <div className="text-center py-16 text-gray-400">
          <p className="text-5xl mb-3">⚡</p>
          <p className="text-sm">Set your User ID in the header and click "Fetch Conflicts" to see results</p>
          <p className="text-xs mt-1 text-gray-300">
            Conflicts are detected when a new behavior contradicts an existing one
          </p>
        </div>
      )}
    </div>
  );
}
