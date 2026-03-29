import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/comfy': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/comfy/, ''),
        headers: {
          'Origin': 'http://127.0.0.1:8000'
        }
      }
    }
  }
})
