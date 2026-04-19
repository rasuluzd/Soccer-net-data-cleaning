"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Logo from "@/components/Logo";
import TypewriterText from "@/components/TypewriterText";
import TeamCarousel from "@/components/TeamCarousel";
import VideoPlayer from "@/components/VideoPlayer";
import Image from "next/image";

const fadeUp = { hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0, transition: { duration: 0.7 } } };

export default function LandingPage() {
  return (
    <>
      <Navbar />

      {/* ═══ HERO ═══ */}
      <section className="relative min-h-screen flex items-center pt-16 overflow-hidden bg-gradient-to-br from-brand-gold via-brand-gold-dark to-brand-navy">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,rgba(255,255,255,0.08)_0%,transparent_60%)]" />
        <div className="max-w-7xl mx-auto px-6 py-20 relative z-10 grid md:grid-cols-2 gap-12 items-center">
          {/* Hero Image */}
          <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8 }} className="flex justify-center md:justify-start">
            <div className="w-48 h-48 md:w-64 md:h-64 rounded-3xl overflow-hidden shadow-2xl">
              <Image
                src="/landing/826118.png"
                alt="Sports highlights search"
                width={256}
                height={256}
                className="w-full h-full object-cover"
                priority
              />
            </div>
          </motion.div>

          {/* Text */}
          <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.15 } } }}>
            <motion.div variants={fadeUp} className="inline-block bg-brand-navy/60 backdrop-blur-sm text-white text-xs uppercase tracking-[0.2em] px-5 py-2.5 rounded-full mb-6 font-semibold">
              Sports Technology Leader
            </motion.div>
            <motion.h1 variants={fadeUp} className="font-display text-4xl md:text-5xl lg:text-6xl font-bold text-white leading-tight mb-6">
              <TypewriterText />
            </motion.h1>
            <motion.p variants={fadeUp} className="text-white/70 text-lg leading-relaxed mb-8 max-w-lg">
              Transforming Sports Video Analysis With Intelligent Search — Instantly Find The Moments That Matter.
            </motion.p>
            <motion.div variants={fadeUp}>
              <Link href="/register" className="inline-block bg-brand-navy/80 hover:bg-brand-navy text-white font-semibold px-8 py-3.5 rounded-full transition-colors text-sm">
                Get Started
              </Link>
            </motion.div>
          </motion.div>
        </div>

        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 text-center text-white/50">
          <p className="text-xs mb-2">About Us</p>
          <motion.div animate={{ y: [0, 6, 0] }} transition={{ repeat: Infinity, duration: 1.5 }}>
            <svg className="w-8 h-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 14l-7 7m0 0l-7-7m7 7V3"/></svg>
          </motion.div>
        </div>
      </section>

      {/* ═══ ABOUT ═══ */}
      <section className="py-24 bg-brand-cream dark:bg-[#1a1a2e]">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={{ visible: { transition: { staggerChildren: 0.15 } } }}>
            <motion.p variants={fadeUp} className="text-brand-gold font-bold uppercase tracking-[0.15em] text-sm mb-4">Welcome To</motion.p>
            <motion.div variants={fadeUp} className="mb-10">
              <Logo className="font-mono text-3xl md:text-4xl text-gray-900 dark:text-white tracking-widest font-bold" />
            </motion.div>
            <div className="grid md:grid-cols-2 gap-16 items-start">
              <motion.div variants={fadeUp}>
                <div className="w-16 h-1 bg-brand-gold mb-8" />
                <p className="text-gray-600 dark:text-white/50 italic text-lg leading-relaxed">
                  Whether It&apos;s A Decisive Goal, A Tactical Formation, Or A Game-Changing Play, Our Solution Allows Analysts, Coaches, And Media Teams To Locate Critical Events Instantly.
                </p>
              </motion.div>
              <motion.div variants={fadeUp}>
                <p className="text-gray-700 dark:text-white/70 leading-relaxed text-base">
                  We Are Developing An <span className="text-brand-gold font-semibold">Advanced Artificial Intelligence</span> System Designed To Revolutionize How Sports Organizations Analyze Video Content. Our Platform Enables Users To Search For Specific <span className="text-brand-gold font-semibold">Highlights</span> Within Seconds, Eliminating Hours Of Manual Review And Unlocking Faster, Smarter Insights.
                </p>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ═══ FEATURES + VIDEO ═══ */}
      <section className="py-24 bg-brand-navy">
        <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-2 gap-12 items-center">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={{ visible: { transition: { staggerChildren: 0.1 } } }}>
            <motion.p variants={fadeUp} className="text-white/50 leading-relaxed mb-4">
              Sports Organizations Generate Massive Amounts Of Video Data Every Day. Traditionally, Identifying Specific Highlights Requires Manual Tagging Or Time-Consuming Review — Slowing Down Workflows And Limiting Analytical Potential.
            </motion.p>
            <motion.p variants={fadeUp} className="text-white/50 leading-relaxed">
              Our AI System Automates This Process By Understanding Video Context And Enabling Natural Search Queries. Users Can Describe What They Are Looking For, And The Platform Intelligently Retrieves Matching Moments With Speed And Precision.
            </motion.p>
          </motion.div>

          <motion.div initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 0.7 }}>
            <div className="bg-gray-900 rounded-2xl overflow-hidden shadow-2xl border border-white/5">
              <div className="px-4 py-3 flex items-center gap-3 border-b border-white/5">
                <div className="bg-white/10 rounded-full px-4 py-1.5 text-white/40 text-sm flex-1">IF Brommapojkarna 0–1 Degerfors mål</div>
                <svg className="w-4 h-4 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
              </div>
              {/* Replace src with your promo video: src="/video/promo.mp4" type="direct" */}
              <VideoPlayer src="/video/Screen Recording 2026-04-19 150404.mp4" loop className="rounded-none" />
            </div>
          </motion.div>
        </div>
      </section>

      <TeamCarousel />
      <Footer />
    </>
  );
}
