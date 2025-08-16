import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // 设置为相对路径，这样在任何服务器上都能正常工作
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: '',
      },
    },
  },
  server: {
    port: parseInt(process.env.VITE_FRONTEND_PORT || '3001'),
    open: true,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    // 优化构建
    rollupOptions: {
      output: {
        manualChunks: {
          // 将大的依赖分离成单独的 chunk
          vendor: ['react', 'react-dom'],
          antd: ['antd'],
          router: ['react-router-dom'],
        }
      }
    },
    // 调整 chunk 大小警告阈值
    chunkSizeWarningLimit: 1000,
  },
})
