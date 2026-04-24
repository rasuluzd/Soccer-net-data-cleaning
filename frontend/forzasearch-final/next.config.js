/** @type {import('next').NextConfig} */
const nextConfig = {
  // Next.js 14: serverComponentsExternalPackages lives under `experimental`.
  // (Renamed to top-level `serverExternalPackages` in Next.js 15.)
  experimental: {
    serverComponentsExternalPackages: [
      '@elastic/elasticsearch',
      'bcryptjs',
      '@xenova/transformers',
      'onnxruntime-node',
    ],
  },
};

module.exports = nextConfig;
