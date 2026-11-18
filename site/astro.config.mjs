// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

// Served from GitHub Pages at https://sublimeye.github.io/rl-lab/
export default defineConfig({
  site: 'https://sublimeye.github.io',
  base: '/rl-lab',
  integrations: [mdx()],
});
