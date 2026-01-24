import React from 'react';
import { StoredBehavior } from '../types';
import { Badge } from './ui/Badge';
import { CredibilityBar } from './ui/CredibilityBar';
import { Database, Zap, Activity, CheckCircle2 } from 'lucide-react';
import { POLARITY_COLORS } from '../constants';

interface StoredBehaviorCardProps {
  behavior: StoredBehavior;
}

export const StoredBehaviorCard: React.FC<StoredBehaviorCardProps> = ({ behavior }) => {
  const stateColor = 
    behavior.behavior_state === 'ACTIVE' ? 'bg-emerald-100 text-emerald-800 border-emerald-200' :
    behavior.behavior_state === 'SUPERSEDED' ? 'bg-indigo-100 text-indigo-800 border-indigo-200' :
    'bg-slate-100 text-slate-800 border-slate-200';

  return (
    <div className={`bg-white rounded-lg border p-4 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden ${
      behavior.behavior_state === 'ACTIVE' ? 'border-emerald-200' : 'border-slate-200 opacity-90'
    }`}>
      {/* State Badge */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
            <Badge colorClass={`${stateColor} font-bold border`}>
                {behavior.behavior_state}
            </Badge>
            <span className="text-xs font-mono text-slate-400 bg-slate-50 px-1.5 py-0.5 rounded">
                {behavior.behavior_id.substring(0, 8)}...
            </span>
        </div>
      </div>

      {/* Main Text */}
      <p className="text-md font-medium text-slate-800 mb-4 leading-relaxed">
        "{behavior.behavior_text}"
      </p>

      {/* Canonical Grid */}
      <div className="grid grid-cols-2 gap-2 text-xs mb-4 bg-slate-50 p-3 rounded border border-slate-100">
        <div>
            <span className="text-slate-400 uppercase font-semibold">Intent</span>
            <div className="text-slate-700 font-medium">{behavior.intent}</div>
        </div>
        <div>
            <span className="text-slate-400 uppercase font-semibold">Polarity</span>
            <div><Badge colorClass={POLARITY_COLORS[behavior.polarity]}>{behavior.polarity}</Badge></div>
        </div>
        <div className="col-span-2">
            <span className="text-slate-400 uppercase font-semibold">Context</span>
            <div className="text-slate-700">{behavior.context}</div>
        </div>
      </div>

      {/* Metrics */}
      <div className="space-y-3 pt-2 border-t border-slate-100">
        <CredibilityBar score={behavior.credibility} label="Credibility" />
        
        <div className="flex items-center justify-between text-xs text-slate-500">
            <div className="flex items-center gap-1">
                <Zap className="w-3 h-3 text-amber-500" />
                <span>Reinforcements: <span className="font-bold text-slate-700">{behavior.reinforcement_count}</span></span>
            </div>
            <div className="flex items-center gap-1">
                <Activity className="w-3 h-3 text-blue-500" />
                <span>Clarity: <span className="font-bold text-slate-700">{behavior.clarity_score.toFixed(2)}</span></span>
            </div>
            <div className="flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                <span>Conf: <span className="font-bold text-slate-700">{behavior.extraction_confidence.toFixed(2)}</span></span>
            </div>
        </div>
      </div>
    </div>
  );
};
