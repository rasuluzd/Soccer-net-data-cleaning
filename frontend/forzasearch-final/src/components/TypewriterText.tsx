"use client";

import { useState, useEffect } from "react";

const WORDS = ["Highlights", "Moments", "Goals", "Clips", "Saves"];
const TYPING_SPEED = 100;
const DELETE_SPEED = 60;
const PAUSE = 1800;

export default function TypewriterText() {
  const [wordIndex, setWordIndex] = useState(0);
  const [text, setText] = useState("");
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const word = WORDS[wordIndex];

    const timeout = setTimeout(
      () => {
        if (!deleting) {
          setText(word.slice(0, text.length + 1));
          if (text.length + 1 === word.length) {
            setTimeout(() => setDeleting(true), PAUSE);
          }
        } else {
          setText(word.slice(0, text.length - 1));
          if (text.length === 0) {
            setDeleting(false);
            setWordIndex((i) => (i + 1) % WORDS.length);
          }
        }
      },
      deleting ? DELETE_SPEED : TYPING_SPEED
    );

    return () => clearTimeout(timeout);
  }, [text, deleting, wordIndex]);

  return (
    <span>
      Find The Exact Sports{" "}
      <span className="text-brand-gold">{text}</span>
      <span className="typewriter-cursor text-brand-gold">|</span>
      {" "}You&apos;re Looking For, Instantly!
    </span>
  );
}
