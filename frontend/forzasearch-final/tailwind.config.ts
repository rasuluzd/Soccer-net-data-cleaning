import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          gold: "#81681C",
          "gold-light": "#C4A93D",
          "gold-dark": "#5A4913",
          navy: "#141E2F",
          "navy-light": "#1E2D42",
          cream: "#FDF6E3",
        },
      },
      fontFamily: {
        display: ['"Cormorant Garamond"', "Georgia", "serif"],
        body: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"Courier New"', "Courier", "monospace"],
      },
      animation: {
        "carousel-scroll": "carousel-scroll 30s linear infinite",
      },
      keyframes: {
        "carousel-scroll": {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
