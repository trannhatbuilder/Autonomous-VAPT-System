'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { apiLogin, getMe, tokenStorage, ApiError } from "../../lib/api";

interface AuthUser {
  id: string;
  email: string;
  role: string;
  display_name: string | null;
}

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;  // initial auth check
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  // On mount: check if there's a valid access token + fetch user
  useEffect(() => {
    const accessToken = tokenStorage.getAccessToken();
    if (!accessToken) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLoading(false);
      return;
    }
    getMe()
      .then((me) => setUser(me))
      .catch(() => {
        tokenStorage.clearTokens();
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = async (email: string, password: string) => {
    const result = await apiLogin(email, password);
    tokenStorage.setTokens(result.access_token, result.refresh_token);
    setUser(result.user);
  };

  const logout = () => {
    tokenStorage.clearTokens();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        logout,
        isAuthenticated: user !== null,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
