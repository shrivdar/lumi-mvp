import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        protein: "#4A90D9",
        gene: "#6B5CE7",
        disease: "#E74C3C",
        pathway: "#2ECC71",
        drug: "#F39C12",
        clinical: "#1ABC9C",
        mechanism: "#95A5A6",
        experiment: "#E67E22",
      },
    },
  },
  plugins: [],
};
export default config;
