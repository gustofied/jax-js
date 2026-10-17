import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: [
      "src/backend/webgpu-batch.test.ts",
      "src/backend/backend.test.ts",
      "src/frontend/array.test.ts",
    ],
  },
});
