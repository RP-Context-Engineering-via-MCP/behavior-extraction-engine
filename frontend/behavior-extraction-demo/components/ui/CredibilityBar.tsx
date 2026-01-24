import React from 'react';

interface CredibilityBarProps {
  score: number; // 0 to 1
  label?: string;
  showValue?: boolean;
  delta?: number;
  className?: string;
}

export const CredibilityBar: React.FC<CredibilityBarProps> = ({ 
  score, 
  label = "Credibility", 
  showValue = true, 
  delta,
  className = "" 
}) => {
  const percentage = Math.min(Math.max(score * 100, 0), 100);
  
  // Color based on score
  let barColor = 'bg-red-500';
  if (score >= 0.7) barColor = 'bg-emerald-500';
  else if (score >= 0.4) barColor = 'bg-yellow-500';

  return (
    <div className={`w-full ${className}`}>
      <div className="flex justify-between items-end mb-1">
        <span className="text-xs font-medium text-slate-500">{label}</span>
        {showValue && (
          <div className="flex items-center gap-1">
             <span className="text-xs font-bold text-slate-700">{score.toFixed(3)}</span>
             {delta !== undefined && delta !== 0 && (
                <span className={`text-xs ${delta > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ({delta > 0 ? '+' : ''}{delta.toFixed(3)})
                </span>
             )}
          </div>
        )}
      </div>
      <div className="w-full bg-slate-200 rounded-full h-2">
        <div 
          className={`${barColor} h-2 rounded-full transition-all duration-500`} 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
};
