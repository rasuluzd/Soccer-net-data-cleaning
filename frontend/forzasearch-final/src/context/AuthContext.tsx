"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from "react";

interface User {
  id: string;
  email: string;
  username: string;
  firstName: string;
  lastName: string;
  role: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (identifier: string, password: string) => Promise<{ error?: string }>;
  register: (data: { email: string; username: string; firstName: string; lastName: string; password: string }) => Promise<{ error?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthState>({
  user: null, token: null, loading: true,
  login: async () => ({}), register: async () => ({}), logout: () => {},
});

function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("forzasearch-token");
}

function storeToken(token: string) {
  localStorage.setItem("forzasearch-token", token);
}

function clearToken() {
  localStorage.removeItem("forzasearch-token");
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const saved = getStoredToken();
    if (!saved) { setLoading(false); return; }
    fetch("/api/auth/me", { headers: { Authorization: `Bearer ${saved}` } })
      .then((r) => r.ok ? r.json() : Promise.reject())
      .then((data) => { if (data?.user) { setUser(data.user); setToken(saved); } else clearToken(); })
      .catch(() => clearToken())
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (identifier: string, password: string) => {
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier, password }),
      });
      const data = await res.json();
      if (data.error) return { error: data.error };
      storeToken(data.token);
      setUser(data.user);
      setToken(data.token);
      return {};
    } catch { return { error: "Network error" }; }
  }, []);

  const register = useCallback(async (d: { email: string; username: string; firstName: string; lastName: string; password: string }) => {
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(d),
      });
      const data = await res.json();
      if (data.error) return { error: data.error };
      storeToken(data.token);
      setUser(data.user);
      setToken(data.token);
      return {};
    } catch { return { error: "Network error" }; }
  }, []);

  const logout = useCallback(() => { setUser(null); setToken(null); clearToken(); }, []);

  return <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
