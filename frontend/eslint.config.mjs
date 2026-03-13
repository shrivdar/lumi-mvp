import { dirname } from "path";
import { fileURLToPath } from "url";
import nextConfig from "eslint-config-next";

const __dirname = dirname(fileURLToPath(import.meta.url));

// eslint-config-next v16 exports flat config natively
const configs = Array.isArray(nextConfig)
  ? nextConfig
  : nextConfig.configs?.["core-web-vitals"] ?? [nextConfig];

const mapped = configs.map((c) => ({
  ...c,
  settings: { ...c.settings, next: { rootDir: __dirname } },
}));

export default [...mapped, { ignores: [".next/", "node_modules/"] }];
