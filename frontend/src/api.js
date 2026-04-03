const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:7890';

export const api = {
  /**
   * POST /v2/extract/detailed — Full pipeline extraction with tracking
   */
  extractDetailed: async ({ user_id, session_id, prompt, recent_history }) => {
    const res = await fetch(`${API_BASE}/v2/extract/detailed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id, session_id, prompt, recent_history }),
    });
    return res.json();
  },

  /**
   * GET /conflicts/{user_id} — All conflicts for a user
   */
  getConflicts: async (userId) => {
    const res = await fetch(`${API_BASE}/conflicts/${userId}`);
    return res.json();
  },

  /**
   * POST /resolve-conflict — Resolve a behavior conflict
   */
  resolveConflict: async ({ conflict_id, user_id, resolution_choice }) => {
    const res = await fetch(`${API_BASE}/resolve-conflict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conflict_id, user_id, resolution_choice }),
    });
    return res.json();
  },

  /**
   * GET /behaviors/{user_id} — All behaviors for a user
   */
  getBehaviors: async (userId, sessionId) => {
    let url = `${API_BASE}/behaviors/${userId}`;
    if (sessionId) url += `?session_id=${sessionId}`;
    const res = await fetch(url);
    return res.json();
  },

  /**
   * GET /health
   */
  health: async () => {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },

  // ── Decay Demo endpoints ──

  /**
   * GET /decay/config — Decay algorithm configuration
   */
  getDecayConfig: async () => {
    const res = await fetch(`${API_BASE}/decay/config`);
    return res.json();
  },

  /**
   * GET /decay/preview/{user_id} — Preview decay state (no persist)
   */
  previewDecay: async (userId) => {
    const res = await fetch(`${API_BASE}/decay/preview/${userId}`);
    return res.json();
  },

  /**
   * POST /decay/apply-lazy/{user_id} — Trigger lazy decay
   */
  applyLazyDecay: async (userId) => {
    const res = await fetch(`${API_BASE}/decay/apply-lazy/${userId}`, { method: 'POST' });
    return res.json();
  },

  /**
   * POST /decay/apply-cron/{user_id} — Trigger cron batch decay
   */
  applyCronDecay: async (userId) => {
    const res = await fetch(`${API_BASE}/decay/apply-cron/${userId}`, { method: 'POST' });
    return res.json();
  },

  /**
   * GET /decay/simulate — Simulate decay curves
   */
  simulateDecay: async (days = 90, initialCredibility = 0.85) => {
    const res = await fetch(
      `${API_BASE}/decay/simulate?days=${days}&initial_credibility=${initialCredibility}`
    );
    return res.json();
  },

  /**
   * POST /v2/extract/detailed — Behavior extraction with flow tracking (for extraction tab)
   * Reuses the detailed endpoint but the UI only surfaces the storage/flow section.
   */
  extractBehaviors: async ({ user_id, session_id, prompt }) => {
    const res = await fetch(`${API_BASE}/v2/extract/detailed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id, session_id, prompt, recent_history: [] }),
    });
    return res.json();
  },
};
