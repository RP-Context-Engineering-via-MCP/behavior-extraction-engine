export const API_BASE_URL = 'http://localhost:8000';

// Colors for Actions
export const ACTION_COLORS: Record<string, string> = {
  NEW_BEHAVIOR: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  DUPLICATE_REINFORCED: 'bg-blue-100 text-blue-800 border-blue-200',
  CONFLICT_DETECTED: 'bg-orange-100 text-orange-800 border-orange-200',
  CONFLICT_AUTO_RESOLVED: 'bg-amber-100 text-amber-800 border-amber-200',
  CONFLICT_LLM_RESOLVED: 'bg-purple-100 text-purple-800 border-purple-200',
  SUPERSEDED_EXISTING: 'bg-indigo-100 text-indigo-800 border-indigo-200',
  IGNORED_NEW: 'bg-gray-100 text-gray-800 border-gray-200',
  COMPATIBLE: 'bg-teal-100 text-teal-800 border-teal-200',
  PRUNED: 'bg-red-50 text-red-800 border-red-200',
};

// Colors for Polarity
export const POLARITY_COLORS: Record<string, string> = {
  POSITIVE: 'bg-green-100 text-green-700',
  NEGATIVE: 'bg-red-100 text-red-700',
  NEUTRAL: 'bg-gray-100 text-gray-700'
};
