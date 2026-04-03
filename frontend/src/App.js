import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { SessionProvider, useSession } from './SessionContext';
import BehaviorExtractionPage from './pages/BehaviorExtractionPage';
import ExtractPage from './pages/ExtractPage';
import ConflictsPage from './pages/ConflictsPage';
import DecayPage from './pages/DecayPage';

const NAV_ITEMS = [
  { to: '/', label: 'Behavior Extraction', icon: '🧬', end: true },
  { to: '/extract', label: 'Extract & Analyze', icon: '🔬' },
  { to: '/conflicts', label: 'Conflicts & Resolution', icon: '⚡' },
  { to: '/decay', label: 'Decay & Credibility', icon: '📉' },
];

function TopBar() {
  const { userId, setUserId, sessionId, setSessionId } = useSession();

  return (
    <header className="h-16 bg-white border-b border-gray-200 px-6 flex items-center justify-between shrink-0 shadow-sm z-10">
      <div className="flex items-center gap-3">
        <span className="text-2xl">🧠</span>
        <div>
          <h1 className="text-base font-bold text-gray-900 leading-tight">Behavior Engine</h1>
          <p className="text-[10px] text-gray-400 leading-tight">Demo Dashboard</p>
        </div>
      </div>

      {/* Centralized User / Session fields */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-500">User ID</label>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            className="w-44 bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5 text-sm text-gray-800 focus:border-brand-500 focus:ring-2 focus:ring-brand-100 outline-none transition"
            placeholder="user-demo-001"
          />
        </div>
        <div className="w-px h-6 bg-gray-200" />
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-500">Session ID</label>
          <input
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            className="w-36 bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5 text-sm text-gray-800 focus:border-brand-500 focus:ring-2 focus:ring-brand-100 outline-none transition"
            placeholder="session-001"
          />
        </div>
      </div>
    </header>
  );
}

function Sidebar() {
  return (
    <aside className="w-60 min-h-screen bg-white border-r border-gray-200 flex flex-col shrink-0">
      <nav className="flex-1 p-3 pt-4 space-y-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-brand-50 text-brand-700 border border-brand-200 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`
            }
          >
            <span className="text-base">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-gray-100">
        <p className="text-[10px] text-gray-400">Behavior Detection & Management v2.0</p>
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <SessionProvider>
        <div className="flex min-h-screen bg-gray-50">
          <Sidebar />
          <div className="flex-1 flex flex-col min-h-screen overflow-hidden">
            <TopBar />
            <main className="flex-1 overflow-auto">
              <Routes>
                <Route path="/" element={<BehaviorExtractionPage />} />
                <Route path="/extract" element={<ExtractPage />} />
                <Route path="/conflicts" element={<ConflictsPage />} />
                <Route path="/decay" element={<DecayPage />} />
              </Routes>
            </main>
          </div>
        </div>
      </SessionProvider>
    </BrowserRouter>
  );
}
