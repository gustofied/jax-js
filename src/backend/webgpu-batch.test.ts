import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { Executable } from "../backend";
import { startTrace, stopTrace } from "../tracing";
import { WebGPUBackend } from "./webgpu";

// Exercise command ordering independently of hardware. Numerical parity is
// covered by the existing browser/backend tests.
function setup() {
  const events: string[] = [];
  const device = {
    limits: { maxStorageBuffersPerShaderStage: 10 },
    features: new Set(),
    addEventListener() {},
    createBuffer: () => ({ destroy: () => events.push("destroy") }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => {
      let active = false;
      return {
        beginComputePass: () => {
          expect(active).toBe(false);
          active = true;
          events.push("begin");
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups: () => events.push("dispatch"),
            end: () => {
              expect(active).toBe(true);
              active = false;
              events.push("end");
            },
          };
        },
        clearBuffer: () => {
          expect(active).toBe(false);
          events.push("clear");
        },
        finish: () => {
          expect(active).toBe(false);
          return {};
        },
      };
    },
    queue: {
      submit: () => events.push("submit"),
      writeBuffer: () => events.push("write"),
    },
  } as unknown as GPUDevice;
  const backend = new WebGPUBackend(device);
  const input = backend.malloc(16),
    output = backend.malloc(16);
  const execute = (clearOutputs = false, target = output) => {
    const exe = new Executable(null!, [
      {
        pipeline: { getBindGroupLayout() {} },
        code: "",
        numInputs: 1,
        numOutputs: 1,
        hasUniform: false,
        clearOutputs,
        passes: [{ grid: [1, 1] }],
      },
    ]) as unknown as Parameters<WebGPUBackend["dispatch"]>[0];
    backend.dispatch(exe, [input], [target]);
  };
  return { backend, events, execute, input, output };
}

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 });
});
afterEach(() => {
  stopTrace();
  vi.unstubAllGlobals();
});

test("bounds batches and shares compute passes across nested scopes", () => {
  const { backend, events, execute } = setup();
  backend.batch(() => {
    for (let i = 0; i < 130; i++) backend.batch(execute);
  });
  expect(events.filter((e) => e === "dispatch")).toHaveLength(130);
  expect(events.filter((e) => e === "submit")).toHaveLength(3);
  expect(events.filter((e) => e === "begin")).toHaveLength(3);
});

test("unbatched callers still submit immediately", () => {
  const { events, execute } = setup();
  execute();
  execute();
  expect(events.filter((e) => e === "submit")).toHaveLength(2);
});

test("flushes before writing host data to a reused buffer", () => {
  const { backend, events, execute, input } = setup();
  backend.batch(() => {
    execute();
    backend.decRef(input);
    const reused = backend.malloc(16, new Uint8Array(16));
    backend.decRef(reused);
  });
  expect(events).toEqual(["begin", "dispatch", "end", "submit", "write"]);
});

test("defers destruction until encoded commands have been submitted", () => {
  const { backend, events, execute } = setup();
  const large = backend.malloc(64 * 1024 * 1024 + 4);
  backend.batch(() => {
    execute(false, large);
    backend.decRef(large);
    expect(events).not.toContain("destroy");
  });
  expect(events).toEqual(["begin", "dispatch", "end", "submit", "destroy"]);
});

test("pool overflow does not split a batch or destroy live commands", () => {
  const { backend, events, execute } = setup();
  const slots = Array.from({ length: 513 }, () => backend.malloc(16));
  backend.batch(() => {
    execute(false, slots[512]);
    for (const slot of slots) backend.decRef(slot);
    expect(events).not.toContain("destroy");
  });
  expect(events).toEqual(["begin", "dispatch", "end", "submit", "destroy"]);
});

test("bounds total small-buffer retention and releases budget on reuse", () => {
  const { backend, events } = setup();
  const create = vi.spyOn(backend.device, "createBuffer");
  const slots = [
    ...Array.from({ length: 512 }, () => backend.malloc(64 * 1024)),
    ...Array.from({ length: 512 }, () => backend.malloc(64 * 1024 - 4)),
  ];
  const extra = backend.malloc(4096);
  for (const slot of slots) backend.decRef(slot);
  backend.decRef(extra);
  expect(events.filter((e) => e === "destroy")).toHaveLength(1);

  const allocations = create.mock.calls.length;
  const reused = backend.malloc(64 * 1024);
  expect(create.mock.calls.length).toBe(allocations);
  backend.decRef(reused);
  expect(events.filter((e) => e === "destroy")).toHaveLength(1);
});

test("retains the original pool limit for large buffers", () => {
  const { backend, events } = setup();
  const slots = Array.from({ length: 65 }, () => backend.malloc(64 * 1024 + 4));
  for (const slot of slots) backend.decRef(slot);
  expect(events.filter((e) => e === "destroy")).toHaveLength(1);
});

test.each([128, 256, 512, 1024, 2048])(
  "reuses a bounded family of %i KiB tiles",
  (kib) => {
    const { backend, events, execute } = setup();
    const create = vi.spyOn(backend.device, "createBuffer");
    const slots = Array.from({ length: 97 }, () => backend.malloc(kib * 1024));
    backend.batch(() => {
      execute(false, slots[96]);
      for (const slot of slots) backend.decRef(slot);
      expect(events).not.toContain("destroy");
    });
    expect(events).toEqual(["begin", "dispatch", "end", "submit", "destroy"]);
    const allocations = create.mock.calls.length;
    const reused = Array.from({ length: 96 }, () => backend.malloc(kib * 1024));
    expect(create.mock.calls.length).toBe(allocations);
    for (const slot of reused) backend.decRef(slot);
  },
);

test("does not expand retention beyond the tile size range", () => {
  const { backend, events } = setup();
  const slots = Array.from({ length: 65 }, () =>
    backend.malloc(4 * 1024 * 1024),
  );
  for (const slot of slots) backend.decRef(slot);
  expect(events.filter((event) => event === "destroy")).toHaveLength(1);
});

test("ends the compute pass before clearing routine outputs", () => {
  const { backend, events, execute } = setup();
  backend.batch(() => {
    execute();
    execute(true);
  });
  expect(events).toEqual([
    "begin",
    "dispatch",
    "end",
    "clear",
    "begin",
    "dispatch",
    "end",
    "submit",
  ]);
});

test("submits work on exceptions and restores normal dispatch", () => {
  const { backend, events, execute } = setup();
  expect(() =>
    backend.batch(() => {
      execute();
      throw Error("test");
    }),
  ).toThrow("test");
  execute();
  expect(events.filter((e) => e === "submit")).toHaveLength(2);
});

test("flushes before a synchronous read inside a batch", () => {
  const { backend, events, execute, output } = setup();
  vi.spyOn(backend.syncReader, "read").mockImplementation(() => {
    events.push("read");
    return new Uint8Array(16);
  });
  backend.batch(() => {
    execute();
    backend.readSync(output);
  });
  expect(events).toEqual(["begin", "dispatch", "end", "submit", "read"]);
});

test("per-operation tracing retains immediate submission", () => {
  const { backend, events, execute } = setup();
  startTrace();
  backend.batch(() => {
    execute();
    execute();
  });
  expect(events.filter((e) => e === "submit")).toHaveLength(2);
});
