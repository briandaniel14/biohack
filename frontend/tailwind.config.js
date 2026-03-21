/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          50:  '#f0f4fa',
          100: '#d5deea',
          200: '#aebfd0',
          300: '#7b8faa',
          400: '#4f637d',
          500: '#3b4b5e',
          600: '#2d3948',
          700: '#1e2834',
          800: '#141c23',
          900: '#0c1216',
          950: '#06090d',
        },
      },
    },
  },
  plugins: [],
}
