import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import Inspector from 'unplugin-vue-dev-locator/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_PROXY_TARGET || env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

  return {
    build: {
      sourcemap: 'hidden',
      assetsDir: 'ui-assets',
    },
    server: {
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
        },
        '/assets': {
          target,
          changeOrigin: true,
        },
        '/health': {
          target,
          changeOrigin: true,
        },
      },
    },
    plugins: [
      vue(),
      Inspector(),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
  }
})
