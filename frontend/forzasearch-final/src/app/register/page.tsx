"use client";

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { useAuth } from "@/context/AuthContext";
import Logo from "@/components/Logo";

function getStrength(pw: string) {
  let s = 0;
  if (pw.length >= 6) s++;
  if (pw.length >= 10) s++;
  if (/[A-Z]/.test(pw)) s++;
  if (/[0-9]/.test(pw)) s++;
  if (/[^A-Za-z0-9]/.test(pw)) s++;
  if (s <= 1) return { score: s, label: "Weak", color: "bg-red-500" };
  if (s <= 3) return { score: s, label: "Medium", color: "bg-yellow-500" };
  return { score: s, label: "Strong", color: "bg-green-500" };
}

export default function RegisterPage() {
  const { register } = useAuth();
  const router = useRouter();

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [showPw, setShowPw] = useState(false);

  const strength = getStrength(password);

  const validate = () => {
    const errs: Record<string, string> = {};
    if (!firstName.trim()) errs.firstName = "Required";
    if (!lastName.trim()) errs.lastName = "Required";
    if (!username.trim()) errs.username = "Required";
    else if (username.length < 3) errs.username = "Min 3 characters";
    if (!email.trim()) errs.email = "Required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) errs.email = "Invalid email";
    if (!password) errs.password = "Required";
    else if (password.length < 6) errs.password = "Min 6 characters";
    if (password !== confirm) errs.confirm = "Passwords don't match";
    setFieldErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    if (!validate()) return;
    if (loading) return;

    setLoading(true);
    const result = await register({ email, username, firstName, lastName, password });
    setLoading(false);

    if (result.error) { setError(result.error); return; }
    router.push("/app");
  };

  const clearFieldError = (field: string) => {
    if (fieldErrors[field]) setFieldErrors((prev) => { const n = { ...prev }; delete n[field]; return n; });
  };

  return (
    <div className="min-h-screen bg-black flex flex-col">
      <nav className="bg-brand-navy/95 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/">
            <Logo />
          </Link>
          <div className="flex items-center gap-6">
            <Link href="/login" className="text-white/80 hover:text-white font-semibold text-sm">Login</Link>
            <Link href="/register" className="text-brand-gold font-semibold text-sm">Register</Link>
          </div>
        </div>
      </nav>

      <div className="flex-1 flex items-center justify-center p-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-lg bg-gray-900 border border-white/10 rounded-2xl p-8 shadow-2xl">
          <h1 className="font-display text-3xl font-bold text-brand-gold text-center mb-8">Create Account</h1>

          {error && <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-4 py-3 rounded-lg mb-6">{error}</div>}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-white/70 text-sm mb-1.5">First Name</label>
                <input type="text" value={firstName} onChange={(e) => { setFirstName(e.target.value); clearFieldError("firstName"); }}
                  className={`w-full bg-white/10 border rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none transition-colors ${fieldErrors.firstName ? "border-red-500/50" : "border-white/10 focus:border-brand-gold/50"}`} placeholder="John" />
                {fieldErrors.firstName && <p className="text-red-400 text-xs mt-1">{fieldErrors.firstName}</p>}
              </div>
              <div>
                <label className="block text-white/70 text-sm mb-1.5">Last Name</label>
                <input type="text" value={lastName} onChange={(e) => { setLastName(e.target.value); clearFieldError("lastName"); }}
                  className={`w-full bg-white/10 border rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none transition-colors ${fieldErrors.lastName ? "border-red-500/50" : "border-white/10 focus:border-brand-gold/50"}`} placeholder="Doe" />
                {fieldErrors.lastName && <p className="text-red-400 text-xs mt-1">{fieldErrors.lastName}</p>}
              </div>
            </div>

            <div>
              <label className="block text-white/70 text-sm mb-1.5">Username</label>
              <input type="text" value={username} onChange={(e) => { setUsername(e.target.value); clearFieldError("username"); }}
                className={`w-full bg-white/10 border rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none transition-colors ${fieldErrors.username ? "border-red-500/50" : "border-white/10 focus:border-brand-gold/50"}`} placeholder="johndoe" />
              {fieldErrors.username && <p className="text-red-400 text-xs mt-1">{fieldErrors.username}</p>}
            </div>

            <div>
              <label className="block text-white/70 text-sm mb-1.5">Email</label>
              <input type="email" value={email} onChange={(e) => { setEmail(e.target.value); clearFieldError("email"); }}
                className={`w-full bg-white/10 border rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none transition-colors ${fieldErrors.email ? "border-red-500/50" : "border-white/10 focus:border-brand-gold/50"}`} placeholder="john@example.com" />
              {fieldErrors.email && <p className="text-red-400 text-xs mt-1">{fieldErrors.email}</p>}
            </div>

            <div>
              <label className="block text-white/70 text-sm mb-1.5">Password</label>
              <input type={showPw ? "text" : "password"} value={password} onChange={(e) => { setPassword(e.target.value); clearFieldError("password"); }}
                className={`w-full bg-white/10 border rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none transition-colors ${fieldErrors.password ? "border-red-500/50" : "border-white/10 focus:border-brand-gold/50"}`} placeholder="••••••••" />
              {fieldErrors.password && <p className="text-red-400 text-xs mt-1">{fieldErrors.password}</p>}
              {password && (
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div className={`h-full ${strength.color} transition-all rounded-full`} style={{ width: `${(strength.score / 5) * 100}%` }} />
                  </div>
                  <span className={`text-xs ${strength.score <= 1 ? "text-red-400" : strength.score <= 3 ? "text-yellow-400" : "text-green-400"}`}>{strength.label}</span>
                </div>
              )}
            </div>

            <div>
              <label className="block text-white/70 text-sm mb-1.5">Confirm Password</label>
              <input type={showPw ? "text" : "password"} value={confirm} onChange={(e) => { setConfirm(e.target.value); clearFieldError("confirm"); }}
                className={`w-full bg-white/10 border rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none transition-colors ${fieldErrors.confirm ? "border-red-500/50" : "border-white/10 focus:border-brand-gold/50"}`} placeholder="••••••••" />
              {fieldErrors.confirm && <p className="text-red-400 text-xs mt-1">{fieldErrors.confirm}</p>}
            </div>

            <label className="flex items-center gap-2 cursor-pointer pt-1">
              <input type="checkbox" checked={showPw} onChange={() => setShowPw(!showPw)} className="w-3.5 h-3.5 rounded bg-white/10 border-white/20" />
              <span className="text-white/50 text-sm">Show passwords</span>
            </label>

            <button type="submit" disabled={loading}
              className="w-full bg-brand-gold hover:bg-brand-gold-light text-white font-semibold py-3 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2">
              {loading && <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg>}
              Create Account
            </button>
          </form>

          <p className="text-center text-white/40 text-sm mt-6">Already have an account? <Link href="/login" className="text-brand-gold hover:underline">Login</Link></p>
        </motion.div>
      </div>
    </div>
  );
}
