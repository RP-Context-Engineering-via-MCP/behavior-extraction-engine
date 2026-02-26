import React, { useState } from 'react';
import { extractDetailed, getBehaviors, getConflicts } from './services/api';
import { ExtractResponseData, StoredBehavior, StoredConflict } from './types';
import { BehaviorCard } from './components/BehaviorCard';
import { StoredBehaviorCard } from './components/StoredBehaviorCard';
import { StoredConflictCard } from './components/StoredConflictCard';
import { Modal } from './components/ui/Modal';
import { Activity, Layers, AlertCircle, CheckCircle2, Search, Loader2, Database, Swords, Edit2, History, ChevronDown, ChevronUp, Clock, FileText } from 'lucide-react';

const DEFAULT_PROMPT = "I prefer dark mode in my IDE. I like using Python for backend development. I don't like verbose code.";

// Simple Stat Card Component
const StatCard = ({ label, value, icon: Icon, color }: { label: string, value: number, icon: any, color: string }) => (
  <div className="bg-white p-4 rounded-lg border border-slate-100 shadow-sm flex items-center justify-between">
    <div>
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-slate-800 mt-1">{value}</p>
    </div>
    <div className={`p-3 rounded-full ${color}`}>
      <Icon className="w-5 h-5 text-white" />
    </div>
  </div>
);

interface HistoryItem {
  id: string;
  timestamp: number;
  prompt: string;
  result: ExtractResponseData;
}

function App() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [sessionId, setSessionId] = useState(`demo_user_${Math.floor(Math.random() * 1000)}`);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractResponseData | null>(null);

  // History State
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Modal State
  const [showBehaviorsModal, setShowBehaviorsModal] = useState(false);
  const [showConflictsModal, setShowConflictsModal] = useState(false);
  const [storedBehaviors, setStoredBehaviors] = useState<StoredBehavior[]>([]);
  const [storedConflicts, setStoredConflicts] = useState<StoredConflict[]>([]);
  const [loadingModal, setLoadingModal] = useState(false);

  const handleExtract = async () => {
    if (!prompt.trim() || !sessionId.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await extractDetailed({
        prompt: prompt,
        user_id: sessionId,
        enable_llm_conflict_resolution: true
      });

      if (response.success && response.data) {
        setResult(response.data);
        // Add to history
        const newItem: HistoryItem = {
          id: crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(),
          timestamp: Date.now(),
          prompt: prompt,
          result: response.data
        };
        setHistory(prev => [newItem, ...prev]);
      } else {
        setError(response.error || response.message || "Unknown error occurred");
      }
    } catch (err) {
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  const handleFetchBehaviors = async () => {
    setLoadingModal(true);
    setShowBehaviorsModal(true);
    try {
      const response = await getBehaviors(sessionId);
      if (response.success && response.data) {
        setStoredBehaviors(response.data.behaviors);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingModal(false);
    }
  };

  const handleFetchConflicts = async () => {
    setLoadingModal(true);
    setShowConflictsModal(true);
    try {
      const response = await getConflicts(sessionId);
      if (response.success && response.data) {
        setStoredConflicts(response.data.conflicts);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingModal(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 pb-20 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20 shadow-sm/50 backdrop-blur-md bg-white/90">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-6 h-6 text-indigo-600" />
            <h1 className="font-bold text-xl tracking-tight text-slate-800">Behavior<span className="text-indigo-600">Extract</span></h1>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-medium hidden sm:inline">User ID:</span>
            <div className="relative group">
              <input
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                className="bg-slate-100 border-none text-sm font-mono text-slate-600 px-3 py-1.5 rounded-md focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all w-32 sm:w-48 text-right sm:text-left"
              />
              <Edit2 className="w-3 h-3 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none group-hover:text-indigo-500 transition-colors" />
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        
        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-8 transition-all hover:shadow-md">
          <label htmlFor="prompt" className="block text-sm font-medium text-slate-700 mb-2">
            User Behavior Prompt
          </label>
          <div className="relative">
            <textarea
              id="prompt"
              rows={4}
              className="w-full rounded-lg border-slate-300 border p-4 text-slate-800 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all resize-y shadow-inner bg-slate-50/50"
              placeholder="Describe user preferences, habits, or feedback..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onFocus={() => {
                if (prompt === DEFAULT_PROMPT) {
                  setPrompt('');
                }
              }}
            />
          </div>
          <div className="mt-4 flex flex-col sm:flex-row justify-between items-center gap-4">
             <div className="flex gap-2 w-full sm:w-auto">
               <span className="hidden sm:inline text-xs text-slate-400 self-center">
                 Target: <span className="font-mono bg-slate-100 px-1 py-0.5 rounded text-slate-500">/extract-detailed</span>
               </span>
             </div>
             
             <div className="flex gap-3 w-full sm:w-auto">
                {/* Secondary Actions */}
                <button 
                  onClick={handleFetchBehaviors}
                  className="flex-1 sm:flex-none px-4 py-2.5 rounded-lg text-sm font-medium text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:text-indigo-600 hover:border-indigo-200 transition-all flex items-center justify-center gap-2"
                >
                  <Database className="w-4 h-4" /> <span className="hidden sm:inline">Stored</span> Behaviors
                </button>
                <button 
                  onClick={handleFetchConflicts}
                  className="flex-1 sm:flex-none px-4 py-2.5 rounded-lg text-sm font-medium text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 hover:text-orange-600 hover:border-orange-200 transition-all flex items-center justify-center gap-2"
                >
                  <Swords className="w-4 h-4" /> <span className="hidden sm:inline">All</span> Conflicts
                </button>

                {/* Primary Action */}
                <button
                  onClick={handleExtract}
                  disabled={loading || !prompt.trim()}
                  className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg font-medium text-white transition-all
                    ${loading || !prompt.trim() 
                      ? 'bg-slate-300 cursor-not-allowed' 
                      : 'bg-indigo-600 hover:bg-indigo-700 shadow-lg shadow-indigo-200 hover:shadow-indigo-300 hover:-translate-y-0.5'}`}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" /> Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4" /> Extract
                    </>
                  )}
                </button>
             </div>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-8 flex items-center gap-3 animate-in shake">
            <AlertCircle className="w-5 h-5" />
            <div>
              <p className="font-bold">Extraction Failed</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 mb-12">
            <div className="flex items-center gap-2 mb-4">
                <div className="p-1.5 bg-indigo-100 rounded-md">
                    <Activity className="w-5 h-5 text-indigo-600" />
                </div>
                <h2 className="text-xl font-bold text-slate-800">Current Analysis</h2>
            </div>
            
            {/* Statistics Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <StatCard 
                label="Extracted" 
                value={result.processing.total_extracted} 
                icon={Layers} 
                color="bg-slate-400" 
              />
              <StatCard 
                label="New / Stored" 
                value={result.processing.total_stored} 
                icon={CheckCircle2} 
                color="bg-emerald-500" 
              />
              <StatCard 
                label="Reinforced" 
                value={result.processing.total_reinforced} 
                icon={Activity} 
                color="bg-blue-500" 
              />
              <StatCard 
                label="Conflicts" 
                value={result.processing.total_conflicts} 
                icon={AlertCircle} 
                color="bg-orange-500" 
              />
            </div>

            {/* Detailed Flow List */}
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-slate-800">Processing Flow</h3>
                <span className="text-sm text-slate-500 bg-slate-100 px-2 py-1 rounded-full">
                  {result.flow_info.length} item{result.flow_info.length !== 1 ? 's' : ''}
                </span>
              </div>
              
              <div className="space-y-6">
                {result.flow_info.map((flow, idx) => (
                  <BehaviorCard key={idx} flow={flow} />
                ))}
              </div>
            </div>
          </div>
        )}
        
        {/* Empty State / Initial Instructions */}
        {!result && !loading && !error && (
          <div className="text-center py-20">
             <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-6 text-slate-300">
                <Activity className="w-10 h-10" />
             </div>
             <h3 className="text-lg font-medium text-slate-700 mb-2">Ready to Extract</h3>
             <p className="text-slate-500 max-w-md mx-auto">Enter a natural language prompt above to visualize the behavior extraction pipeline, or inspect existing data using the buttons.</p>
          </div>
        )}

        {/* --- Session History Section --- */}
        {history.length > 1 && (
            <div className="border-t border-slate-200 pt-8 mt-8 animate-in fade-in">
                <button 
                    onClick={() => setShowHistory(!showHistory)}
                    className="flex items-center justify-between w-full p-4 bg-white rounded-xl border border-slate-200 shadow-sm hover:bg-slate-50 hover:border-slate-300 transition-all group"
                >
                    <div className="flex items-center gap-4">
                        <div className="p-2 bg-slate-100 rounded-lg text-slate-500 group-hover:bg-indigo-50 group-hover:text-indigo-600 transition-colors">
                            <History className="w-6 h-6" />
                        </div>
                        <div className="text-left">
                            <h3 className="font-bold text-slate-800 text-lg">Activity History</h3>
                            <p className="text-sm text-slate-500">
                                {history.length - 1} previous interaction{history.length - 1 !== 1 ? 's' : ''} in this session
                            </p>
                        </div>
                    </div>
                    {showHistory ? (
                        <ChevronUp className="w-5 h-5 text-slate-400 group-hover:text-slate-600" />
                    ) : (
                        <ChevronDown className="w-5 h-5 text-slate-400 group-hover:text-slate-600" />
                    )}
                </button>

                {showHistory && (
                    <div className="mt-8 space-y-12 relative">
                        {/* Vertical line connecting items */}
                        <div className="absolute left-8 top-4 bottom-0 w-0.5 bg-slate-200 -z-10"></div>

                        {history.slice(1).map((item) => (
                            <div key={item.id} className="relative pl-20 animate-in slide-in-from-left-4 duration-300">
                                {/* Timeline Node */}
                                <div className="absolute left-6 top-0 w-4 h-4 rounded-full bg-white border-4 border-indigo-200 shadow-sm"></div>
                                <div className="absolute left-[30px] top-4 w-0.5 h-full bg-slate-200 -z-10"></div>
                                
                                {/* Header Info */}
                                <div className="flex flex-col sm:flex-row sm:items-center gap-2 mb-4">
                                    <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-100 text-xs font-medium text-slate-600 border border-slate-200 w-fit">
                                        <Clock className="w-3 h-3" />
                                        {new Date(item.timestamp).toLocaleTimeString()}
                                    </div>
                                    <div className="text-xs text-slate-400 font-mono hidden sm:block">
                                        ID: {item.id.slice(0,8)}
                                    </div>
                                </div>

                                {/* Prompt Bubble */}
                                <div className="bg-slate-50 rounded-lg border border-slate-200 p-4 mb-6 relative">
                                    <div className="absolute top-4 -left-3 w-3 h-3 bg-slate-50 border-l border-b border-slate-200 transform rotate-45"></div>
                                    <div className="flex items-start gap-2">
                                        <FileText className="w-4 h-4 text-slate-400 mt-1 shrink-0" />
                                        <div>
                                            <span className="text-xs font-bold text-slate-400 uppercase tracking-wide block mb-1">Input Prompt</span>
                                            <p className="text-slate-700 italic">"{item.prompt}"</p>
                                        </div>
                                    </div>
                                </div>

                                {/* Result Cards */}
                                <div className="space-y-4 opacity-90 hover:opacity-100 transition-opacity">
                                    {item.result.flow_info.map((flow, idx) => (
                                         <BehaviorCard key={idx} flow={flow} />
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        )}

      </main>

      {/* --- MODALS --- */}

      {/* Stored Behaviors Modal */}
      <Modal 
        isOpen={showBehaviorsModal} 
        onClose={() => setShowBehaviorsModal(false)}
        title="Stored Behaviors"
      >
        {loadingModal ? (
           <div className="flex justify-center py-12 text-slate-500 gap-2 items-center">
             <Loader2 className="w-5 h-5 animate-spin" /> Loading data...
           </div>
        ) : storedBehaviors.length === 0 ? (
           <div className="text-center py-12 text-slate-400">No stored behaviors found for this session.</div>
        ) : (
           <div className="grid grid-cols-1 gap-4">
             {storedBehaviors.map((b) => (
               <StoredBehaviorCard key={b.behavior_id} behavior={b} />
             ))}
           </div>
        )}
      </Modal>

      {/* Conflicts Modal */}
      <Modal 
        isOpen={showConflictsModal} 
        onClose={() => setShowConflictsModal(false)}
        title="Conflict History"
      >
        {loadingModal ? (
           <div className="flex justify-center py-12 text-slate-500 gap-2 items-center">
             <Loader2 className="w-5 h-5 animate-spin" /> Loading data...
           </div>
        ) : storedConflicts.length === 0 ? (
           <div className="text-center py-12 text-slate-400">No conflicts recorded for this session.</div>
        ) : (
           <div className="grid grid-cols-1 gap-4">
             {storedConflicts.map((c) => (
               <StoredConflictCard key={c.conflict_id} conflict={c} />
             ))}
           </div>
        )}
      </Modal>
    </div>
  );
}

export default App;
