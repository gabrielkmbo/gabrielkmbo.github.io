import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/gabrielkmbo/github.io/', // Replace with your repository name
});
