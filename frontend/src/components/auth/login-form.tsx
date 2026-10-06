'use client';

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { useAuth } from "./auth-provider";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Alert, AlertDescription } from "../ui/alert";
import { Loader2, ShieldCheck, Sun, Moon } from "lucide-react";

export function LoginForm() {
  const { login } = useAuth();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => setMounted(true), []);

  const isDark = theme === "dark";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
    } catch (err: any) {
      setError(err.message || "Login failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 p-4 transition-colors">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-between mb-8">
          <div className="w-12 h-12 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 flex items-center justify-center shadow-sm">
            <ShieldCheck className="w-6 h-6 text-emerald-400" />
          </div>
          {mounted && (
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => setTheme(isDark ? "light" : "dark")}
              aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
              title={isDark ? "Switch to light theme" : "Switch to dark theme"}
              className="border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
            >
              {isDark ? <Sun /> : <Moon />}
            </Button>
          )}
        </div>
        <div className="text-center mb-6">
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">VAPT-AI</h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 mt-1">Vulnerability Assessment &amp; Penetration Testing AI</p>
        </div>

        <Card className="bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 backdrop-blur shadow-sm dark:shadow-none">
          <CardHeader>
            <CardTitle className="text-zinc-900 dark:text-zinc-50">Sign in</CardTitle>
            <CardDescription className="text-zinc-600 dark:text-zinc-400">
              Enter your credentials to access the VAPT-AI console.
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit}>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-zinc-700 dark:text-zinc-200">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="admin@vapt-ai.local"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={submitting}
                  className="bg-white dark:bg-zinc-950 border-zinc-300 dark:border-zinc-800 text-zinc-900 dark:text-zinc-50 placeholder-zinc-400 dark:placeholder-zinc-600"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-zinc-700 dark:text-zinc-200">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={submitting}
                  className="bg-white dark:bg-zinc-950 border-zinc-300 dark:border-zinc-800 text-zinc-900 dark:text-zinc-50 placeholder-zinc-400 dark:placeholder-zinc-600"
                />
              </div>
              {error && (
                <Alert className="bg-red-50 dark:bg-red-950/40 border-red-200 dark:border-red-900">
                  <AlertDescription className="text-red-700 dark:text-red-200">{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
            <CardFooter>
              <Button
                type="submit"
                disabled={submitting || !email || !password}
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white"
              >
                {submitting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Signing in...
                  </>
                ) : (
                  "Sign in"
                )}
              </Button>
            </CardFooter>
          </form>
        </Card>

        <p className="text-center text-xs text-zinc-500 dark:text-zinc-600 mt-6">
          VAPT-AI v3.2 — single-user console. Default admin: admin@vapt-ai.local
        </p>
      </div>
    </div>
  );
}
