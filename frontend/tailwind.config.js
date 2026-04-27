/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        muted: "#9ca3af",
        tertiary: "#6b7280",
        info: "#0ea5e9",
        success: "#16a34a",
        secondary: "#f3f4f6",
      },
    },
  },
  plugins: [],
};
