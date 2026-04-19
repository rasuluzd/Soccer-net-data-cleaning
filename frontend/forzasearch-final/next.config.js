/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: ['@elastic/elasticsearch', 'bcryptjs'],
};

module.exports = nextConfig;
