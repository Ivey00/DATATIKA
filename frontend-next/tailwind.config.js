/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#ffbd7f',    // light orange
        'primary-foreground': '#FFFFFC', // Off-white for text on primary
        secondary: '#FF9149',  // Orange
        'secondary-foreground': '#000000', // Black for text on secondary
        tertiary: '#BEB7A4',   // Light gray/beige
        background: '#FFFFFC', // Off-white
        foreground: '#000000', // Black
        destructive: '#ffbd7f', // (same as primary)
        'destructive-foreground': '#FFFFFC', // Off-white for text on destructive
        muted: '#BEB7A4', // Light gray/beige (same as tertiary)
        'muted-foreground': '#000000', // Black for text on muted
        accent: '#FF9149', // Orange (same as secondary)
        'accent-foreground': '#000000', // Black for text on accent
        card: '#FFFFFC', // Off-white (same as background)
        'card-foreground': '#000000', // Black for text on card
        popover: '#FFFFFC', // Off-white (same as background)
        'popover-foreground': '#000000', // Black for text on popover
        border: '#BEB7A4', // Light gray/beige (same as tertiary)
        input: '#BEB7A4', // Light gray/beige (same as tertiary)
        ring: '#FF9149', // Orange (same as secondary)
      },
    },
  },
  plugins: [],
}