/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        chatBg: "#343541",  // ChatGPT-like background
        codeBlock: "#282C34", // Dark theme for code blocks
        codeText: "#ABB2BF",  // ChatGPT-style code text color
        inlineCodeBg: "#E8E8E8", // Light inline code background
        inlineCodeText: "#2D2D2D" // Dark inline code text
      },
    },
  },
  plugins: [],
};
