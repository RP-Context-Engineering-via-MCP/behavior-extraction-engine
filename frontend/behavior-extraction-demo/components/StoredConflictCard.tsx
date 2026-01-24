import React from 'react';
import { StoredConflict } from '../types';
import { Badge } from './ui/Badge';
import { AlertTriangle, Swords, Brain } from 'lucide-react';

interface StoredConflictCardProps {
  conflict: StoredConflict;
}

export const StoredConflictCard: React.FC<StoredConflictCardProps> = ({ conflict }) => {
  const statusColor = 
    conflict.resolution_status === 'RESOLVED' || conflict.resolution_status === 'AUTO_RESOLVED' ? 'bg-emerald-100 text-emerald-800' :
    conflict.resolution_status === 'PENDING' ? 'bg-amber-100 text-amber-800' :
    'bg-slate-100 text-slate-800';

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-5 shadow-sm hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-500" />
            <span className="font-mono text-xs text-slate-400">{conflict.conflict_id.substring(0, 8)}...</span>
            <Badge colorClass={statusColor}>{conflict.resolution_status}</Badge>
        </div>
        <div className="text-xs text-slate-400">
            {new Date(conflict.created_at * 1000).toLocaleString()}
        </div>
      </div>

      <div className="mb-4">
        <Badge colorClass="bg-orange-50 text-orange-700 border border-orange-100">{conflict.conflict_type}</Badge>
      </div>

      {/* VS Section */}
      <div className="relative grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
         {/* Behavior 1 */}
         <div className="bg-slate-50 p-3 rounded border border-slate-200">
            <div className="text-xs font-bold text-slate-500 mb-1">Behavior 1 ({conflict.behavior_1_state})</div>
            <p className="text-sm text-slate-800 mb-2 italic">"{conflict.behavior_1_text}"</p>
            <div className="text-xs text-slate-400 font-mono">ID: {conflict.behavior_id_1.substring(0, 6)}...</div>
            <div className="text-xs text-slate-500 mt-1">Credibility: <strong>{conflict.behavior_1_credibility.toFixed(3)}</strong></div>
         </div>

         {/* VS Icon Absolute Center (Desktop) */}
         <div className="hidden md:flex absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white border border-slate-200 rounded-full items-center justify-center text-slate-400 z-10 shadow-sm">
            <Swords className="w-4 h-4" />
         </div>

         {/* Behavior 2 */}
         <div className="bg-slate-50 p-3 rounded border border-slate-200">
            <div className="text-xs font-bold text-slate-500 mb-1">Behavior 2 ({conflict.behavior_2_state})</div>
            <p className="text-sm text-slate-800 mb-2 italic">"{conflict.behavior_2_text}"</p>
            <div className="text-xs text-slate-400 font-mono">ID: {conflict.behavior_id_2.substring(0, 6)}...</div>
            <div className="text-xs text-slate-500 mt-1">Credibility: <strong>{conflict.behavior_2_credibility.toFixed(3)}</strong></div>
         </div>
      </div>

      {/* LLM Analysis */}
      {conflict.llm_analysis && (
          <div className="bg-purple-50 p-3 rounded border border-purple-100 mt-3">
              <div className="flex items-center gap-1.5 mb-2 text-purple-800 font-semibold text-xs uppercase tracking-wide">
                  <Brain className="w-3 h-3" /> LLM Analysis
              </div>
              <p className="text-sm text-purple-900 leading-relaxed">{conflict.llm_analysis}</p>
          </div>
      )}
      
      <div className="mt-3 text-xs text-right text-slate-400">
        Similarity Distance: {conflict.similarity_distance.toFixed(4)}
      </div>
    </div>
  );
};
