/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
  // Add proxy configuration
  webProxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      timeout: 300000, // 5 minutes timeout
      proxyTimeout: 300000,
      // Handle proxy errors
      onError: (err, req, res) => {
        console.error('Proxy Error:', err);
        res.writeHead(500, {
          'Content-Type': 'application/json',
        });
        res.end(JSON.stringify({ error: 'Proxy connection failed' }));
      },
    },
  },
  webpack: (config) => {
    config.externals = [...(config.externals || []), { sharp: 'commonjs sharp' }];
    return config;
  },
}

module.exports = nextConfig