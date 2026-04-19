"use client";

import { useRef, useState, useEffect } from "react";
import Hls from "hls.js";

interface Props {
  src: string;
  type?: "hls" | "direct";
  startSec?: number;
  endSec?: number;
  className?: string;
  autoPlay?: boolean;
  loop?: boolean;
}

export default function VideoPlayer({ src, type, startSec = 0, endSec, className = "", autoPlay = false, loop = false }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  const isHls = type === "hls" || src.includes(".m3u8");

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) { setStatus("error"); return; }

    if (hlsRef.current) { hlsRef.current.destroy(); hlsRef.current = null; }

    if (isHls && Hls.isSupported()) {
      const hls = new Hls({ startPosition: startSec, maxBufferLength: 60 });
      hlsRef.current = hls;
      hls.loadSource(src);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        setStatus("ready");
        video.currentTime = startSec;
        if (autoPlay) video.play().catch(() => {});
      });
      hls.on(Hls.Events.ERROR, (_, d) => { if (d.fatal) setStatus("error"); });
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = src;
      video.addEventListener("loadedmetadata", () => { setStatus("ready"); video.currentTime = startSec; });
    } else if (!isHls) {
      video.src = src;
      video.addEventListener("loadedmetadata", () => { setStatus("ready"); video.currentTime = startSec; });
    } else {
      setStatus("error");
    }

    return () => { if (hlsRef.current) { hlsRef.current.destroy(); hlsRef.current = null; } };
  }, [src, startSec, isHls, autoPlay]);

  // Pause at clip end
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !endSec) return;
    const handler = () => { if (v.currentTime >= endSec) v.pause(); };
    v.addEventListener("timeupdate", handler);
    return () => v.removeEventListener("timeupdate", handler);
  }, [endSec]);

  if (!src) {
    return (
      <div className={`bg-gray-900 rounded-xl flex items-center justify-center aspect-video text-gray-500 text-sm ${className}`}>
        No video source — add your video file to /public/video/
      </div>
    );
  }

  return (
    <div className={`relative rounded-xl overflow-hidden bg-black ${className}`}>
      {status === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-white/50 text-sm z-10">Loading…</div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 text-red-400 text-sm z-10 gap-1">
          <span>Could not load video</span>
          <span className="text-white/30 text-xs">Check the source URL</span>
        </div>
      )}
      <video ref={videoRef} controls playsInline loop={loop} muted className="w-full aspect-video" />
    </div>
  );
}
