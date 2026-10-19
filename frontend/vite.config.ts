import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import { defineConfig, loadEnv } from "vite";
import { fileURLToPath } from "node:url";

export default defineConfig(({ mode }) => {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const frontend = fileURLToPath(new URL(".", import.meta.url));
  const env = { ...loadEnv(mode, root, "OPENROUTER_"), ...loadEnv(mode, frontend, "OPENROUTER_"), ...process.env };

  return {
    plugins: [react(), babel({ presets: [reactCompilerPreset()] })],
    // This local, browser-run prototype intentionally exposes only these values.
    define: {
      "import.meta.env.OPENROUTER_API_KEY": JSON.stringify(env.OPENROUTER_API_KEY ?? ""),
      "import.meta.env.OPENROUTER_MODEL": JSON.stringify(env.OPENROUTER_MODEL ?? ""),
    },
  };
});
