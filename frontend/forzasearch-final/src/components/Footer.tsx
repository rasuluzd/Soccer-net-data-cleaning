"use client";

import Logo from "./Logo";

export default function Footer() {
  return (
    <footer className="bg-brand-navy text-white/70 border-t border-white/5">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          {/* Logo */}
          <div>
            <div className="mb-3">
              <Logo />
            </div>
            <p className="text-sm text-white/40 leading-relaxed">
              AI-powered sports video search platform.
            </p>
          </div>

          {/* Contact */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-3">Contact</h4>
            <p className="text-sm leading-relaxed">contact@forzasearch.com</p>
            <p className="text-sm leading-relaxed mt-1">+47 970 80 007</p>
          </div>

          {/* Address */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-3">Address</h4>
            <p className="text-sm leading-relaxed">
              ForzaSearch<br />
              Stensberggata 27, 0170<br />
              Oslo, Norway
            </p>
          </div>
        </div>

        <div className="border-t border-white/10 mt-10 pt-6 text-center text-xs text-white/30">
          © {new Date().getFullYear()} ForzaSearch. All rights reserved.
        </div>
      </div>
    </footer>
  );
}
