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
          50:  '#f0faf4',
          100: '#d5ead9',
          200: '#aed0b5',
          300: '#7baa84',
          400: '#4f7d58',
          500: '#3b5e43',
          600: '#2d4833',
          700: '#1e3425',
          800: '#142319',
          900: '#0c1610',
          950: '#060d08',
        },
      },
    },
  },
  plugins: [],
}
