import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from 'kimi-plugin-inspect-react'

const inspectAttr = process.env.NODE_ENV !== 'production' 
  ? (await import('kimi-plugin-inspect-react')).inspectAttr 
  : () => null


// https://vite.dev/config/
export default defineConfig({
  base: '/TAAC2026/',
  plugins: [inspectAttr(), react()],
  server: {
    port: 3000,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
