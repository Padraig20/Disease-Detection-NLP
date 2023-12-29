/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{html,ts}',
  ],
  theme: {
    extend: {
      colors:{
        'cool-blue': {
          DEFAULT: '#308BC8',
          50: '#C0DDF0',
          100: '#AFD4EC',
          200: '#8EC2E4',
          300: '#6EB0DC',
          400: '#4D9ED4',
          500: '#308BC8',
          600: '#256C9B',
          700: '#1A4C6E',
          800: '#0F2D40',
          900: '#050D13',
          950: '#000000'
        },
      },
      spacing: {
        '100': '28rem',
        '128': '32rem',
        '130': '50rem'
      }
    },
  },
  variants: {
    extend: {},
  },
  plugins: [
    require("@tailwindcss/forms"),
    require('@tailwindcss/typography'),
  ],
};

