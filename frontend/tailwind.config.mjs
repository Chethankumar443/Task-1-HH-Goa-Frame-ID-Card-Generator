/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        hh: {
          DEFAULT: '#026636',
          dark: '#014424',
          darker: '#002e18',
          card: '#02522c',
          yellow: '#FEE001',
          yellowHover: '#e6ca00',
          accent: '#059653',
        },
      },
      fontFamily: {
        heading: ['Imbue', 'serif'],
        mono: ['Victor Mono', 'monospace'],
        sans: ['Victor Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};

