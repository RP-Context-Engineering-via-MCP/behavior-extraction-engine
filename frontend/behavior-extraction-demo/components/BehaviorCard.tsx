import React, { useState } from 'react';
import { 
  BehaviorFlow, 
  CanonicalFields, 
  ConflictInfo, 
  ReinforcementInfo 
} from '../types';
import { Badge } from './ui/Badge';
import { CredibilityBar } from './ui/CredibilityBar';
import { ACTION_COLORS, POLARITY_COLORS } from '../constants';
import { 
  AlertTriangle, 
  ArrowRight, 
  Brain, 
  CheckCircle, 
  ChevronDown, 
  ChevronUp, 
  GitMerge, 
  ShieldCheck, 
  Database,
  ArrowUpRight
} from 'lucide-react';

interface BehaviorCardProps {
  flow: BehaviorFlow;
}

// --- Sub-components for Cleaner Code ---

const CanonicalDisplay: React.FC<{ fields: CanonicalFields | null }> = ({ fields }) => {
  if (!fields) return <div className="text-sm text-gray-400 italic">No structured data</div>;
  
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm mt-3 bg-slate-50 p-3 rounded-md border border-slate-100">
      <div>
        <span className="text-xs text-slate-400 uppercase tracking-wider block">Intent</span>
        <span className="font-semibold text-slate-700">{fields.intent}</span>
      </div>
      <div>
        <span className="text-xs text-slate-400 uppercase tracking-wider block">Polarity</span>
        <Badge colorClass={POLARITY_COLORS[fields.polarity] || 'bg-gray-100'}>{fields.polarity}</Badge>
      </div>
      <div>
        <span className="text-xs text-slate-400 uppercase tracking-wider block">Target</span>
        <span className="text-slate-700 font-medium">{fields.target}</span>
      </div>
      <div>
        <span className="text-xs text-slate-400 uppercase tracking-wider block">Context</span>
        <span className="text-slate-700">{fields.context}</span>
      </div>
    </div>
  );
};

const ConflictSection: React.FC<{ conflict: ConflictInfo; matchedText: string | null }> = ({ conflict, matchedText }) => {
  const isLLM = conflict.resolution_method === 'LLM';
  
  return (
    <div className="mt-4 p-4 bg-orange-50 rounded-lg border border-orange-100">
      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle className="w-5 h-5 text-orange-500" />
        <h4 className="font-semibold text-orange-900">Conflict Detected</h4>
        <Badge colorClass="bg-orange-200 text-orange-800">{conflict.conflict_type}</Badge>
      </div>

      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div className="bg-white p-3 rounded border border-orange-200">
          <div className="text-xs text-orange-600 font-bold mb-1 uppercase">Conflicting With (Existing)</div>
          <p className="text-sm text-slate-700 italic">"{matchedText || conflict.conflicting_behavior_text}"</p>
          <div className="mt-2 text-xs text-slate-500">
             Polarity: <span className="font-bold">{conflict.existing_polarity}</span>
          </div>
        </div>
        <div className="bg-white p-3 rounded border border-orange-200">
          <div className="text-xs text-orange-600 font-bold mb-1 uppercase">New Extraction</div>
          <p className="text-sm text-slate-700 italic">Target: {conflict.new_polarity ? "Has polarity mismatch" : "Incompatible intent"}</p>
          <div className="mt-2 text-xs text-slate-500">
             Polarity: <span className="font-bold">{conflict.new_polarity}</span>
          </div>
        </div>
      </div>

      {isLLM && conflict.llm_analysis && (
        <div className="mb-4 bg-purple-50 p-3 rounded border border-purple-100">
            <div className="flex items-center gap-2 mb-2 text-purple-800">
                <Brain className="w-4 h-4" />
                <span className="font-semibold text-sm">LLM Analysis</span>
            </div>
            <p className="text-sm text-purple-900 leading-relaxed">{conflict.llm_analysis}</p>
            {conflict.llm_confidence && (
                 <div className="mt-2 flex items-center gap-2">
                    <span className="text-xs text-purple-600 font-medium">Confidence:</span>
                    <div className="w-24 bg-purple-200 rounded-full h-1.5">
                        <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: `${conflict.llm_confidence * 100}%`}}></div>
                    </div>
                    <span className="text-xs text-purple-600">{Math.round(conflict.llm_confidence * 100)}%</span>
                 </div>
            )}
        </div>
      )}

      <div className="flex flex-col sm:flex-row justify-between items-center bg-white p-3 rounded border border-slate-200 shadow-sm">
        <div className="flex items-center gap-2">
            <span className="text-sm text-slate-500">Resolution:</span>
            <Badge colorClass="bg-slate-100 text-slate-800 font-mono">{conflict.resolution}</Badge>
        </div>
        <div className="flex items-center gap-2 mt-2 sm:mt-0">
            {conflict.winner_behavior_id && (
                <span className="text-xs text-emerald-600 font-bold flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> Auto-Resolved
                </span>
            )}
            {conflict.requires_user_resolution && (
                <span className="text-xs text-amber-600 font-bold flex items-center gap-1">
                    Requires Manual Review
                </span>
            )}
        </div>
      </div>
      
      {conflict.explanation && (
          <p className="mt-2 text-xs text-slate-500">{conflict.explanation}</p>
      )}
    </div>
  );
};

const ReinforcementSection: React.FC<{ info: ReinforcementInfo }> = ({ info }) => (
  <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
    <div className="flex items-center gap-2 mb-2">
      <ArrowUpRight className="w-5 h-5 text-blue-500" />
      <h4 className="font-semibold text-blue-900">Behavior Reinforced</h4>
    </div>
    <p className="text-sm text-blue-800 mb-3">{info.message}</p>
    <div className="grid grid-cols-3 gap-2">
      <div className="bg-white p-2 rounded text-center">
        <div className="text-xs text-slate-400">Previous</div>
        <div className="font-mono text-slate-700 font-bold">{info.previous_credibility.toFixed(3)}</div>
      </div>
      <div className="flex items-center justify-center text-blue-400">
        <ArrowRight className="w-4 h-4" />
      </div>
      <div className="bg-white p-2 rounded text-center border-b-2 border-emerald-400">
        <div className="text-xs text-slate-400">New</div>
        <div className="font-mono text-emerald-700 font-bold">{info.new_credibility.toFixed(3)}</div>
      </div>
    </div>
  </div>
);

// --- Main Component ---

export const BehaviorCard: React.FC<BehaviorCardProps> = ({ flow }) => {
  const [expanded, setExpanded] = useState(false);
  
  const actionColor = ACTION_COLORS[flow.action] || 'bg-gray-100 border-gray-200 text-gray-800';
  
  // Icons mapping based on action
  const ActionIcon = () => {
    switch (flow.action) {
      case 'CONFLICT_DETECTED':
      case 'CONFLICT_LLM_RESOLVED':
      case 'CONFLICT_AUTO_RESOLVED': return <AlertTriangle className="w-4 h-4" />;
      case 'DUPLICATE_REINFORCED': return <ArrowUpRight className="w-4 h-4" />;
      case 'NEW_BEHAVIOR': return <Database className="w-4 h-4" />;
      case 'SUPERSEDED_EXISTING': return <ShieldCheck className="w-4 h-4" />;
      case 'COMPATIBLE': return <GitMerge className="w-4 h-4" />;
      default: return <Brain className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden transition-all hover:shadow-md mb-4">
      {/* Header Bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-50 border-b border-slate-100">
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-bold uppercase tracking-wide ${actionColor}`}>
          <ActionIcon />
          {flow.action.replace(/_/g, ' ')}
        </div>
        <div className="text-xs text-slate-400 font-mono">
            Confidence: {Math.round(flow.extraction_confidence * 100)}%
        </div>
      </div>

      {/* Main Content */}
      <div className="p-5">
        <div className="flex flex-col md:flex-row gap-6">
          
          {/* Left: Description & Canonical */}
          <div className="flex-1">
             <h3 className="text-lg font-medium text-slate-800 leading-snug">
               "{flow.behavior_description}"
             </h3>
             <CanonicalDisplay fields={flow.canonical} />
             
             {/* Stored State Indicator */}
             {flow.stored_behavior_id && (
                 <div className="mt-3 flex items-center gap-1.5 text-xs text-emerald-600 font-medium">
                     <Database className="w-3 h-3" />
                     <span>Saved to Database ID: {flow.stored_behavior_id.slice(0,8)}...</span>
                 </div>
             )}
          </div>

          {/* Right: Metrics */}
          <div className="w-full md:w-48 flex flex-col gap-4 shrink-0">
             <CredibilityBar 
                score={flow.credibility} 
                delta={flow.credibility_delta} 
                label="Current Credibility"
             />
             <div className="space-y-1">
                <div className="flex justify-between text-xs text-slate-500">
                    <span>Clarity</span>
                    <span className="font-mono font-medium">{flow.clarity_score}</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1">
                    <div className="bg-blue-400 h-1 rounded-full" style={{ width: `${flow.clarity_score * 100}%`}}></div>
                </div>
             </div>
             {flow.distance !== null && (
                 <div className="text-xs text-slate-400 text-right">
                     Matched Distance: {flow.distance.toFixed(3)}
                 </div>
             )}
          </div>
        </div>

        {/* Dynamic Details Area */}
        {flow.conflict_info && (
            <ConflictSection conflict={flow.conflict_info} matchedText={flow.matched_behavior_text} />
        )}

        {flow.reinforcement_info && (
            <ReinforcementSection info={flow.reinforcement_info} />
        )}
        
        {/* Toggle Details Footer */}
        {flow.details && (
            <div className="mt-4 pt-3 border-t border-slate-100">
                 <button 
                    onClick={() => setExpanded(!expanded)}
                    className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-800 transition-colors"
                 >
                    {expanded ? "Hide Processing Details" : "Show Processing Details"}
                    {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                 </button>
                 
                 {expanded && (
                     <div className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded">
                         <p>{flow.details}</p>
                         {flow.matched_behavior_id && <p className="mt-1 font-mono">Ref ID: {flow.matched_behavior_id}</p>}
                     </div>
                 )}
            </div>
        )}
      </div>
    </div>
  );
};
