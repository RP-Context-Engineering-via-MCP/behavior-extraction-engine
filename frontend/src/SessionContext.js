import React, { createContext, useContext, useState } from 'react';

const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const [userId, setUserId] = useState('5ca4d3ee-a139-44f9-9f9a-84655025a8f2');
  const [sessionId, setSessionId] = useState('45388bbc-d11c-4a34-bf2b-65d69a3cfa6f');

  return (
    <SessionContext.Provider value={{ userId, setUserId, sessionId, setSessionId }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSession must be used inside SessionProvider');
  return ctx;
}
