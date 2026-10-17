import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { AluExp, AluOp, AluVar, DType, Kernel, Reduction } from "../alu";
import { ShapeTracker, unravelAlu } from "../shape";
import { tuneWebgpu } from "../tuner";
import { WebGPUBackend } from "./webgpu";

function source(
  subgroups: boolean,
  rows: number,
  columns: number,
  op = AluOp.Add,
  packed = false,
) {
  const device = {
    limits: {
      maxStorageBuffersPerShaderStage: 10,
      maxComputeWorkgroupSizeX: 256,
    },
    features: new Set(subgroups ? ["subgroups"] : []),
    addEventListener() {},
    createBuffer: () => ({}),
  } as unknown as GPUDevice;
  const backend = new WebGPUBackend(device);
  vi.spyOn(backend.pipelines, "prepareSync").mockReturnValue(
    {} as GPUComputePipeline,
  );
  const values = AluExp.globalView(
    packed ? DType.Uint32 : DType.Float32,
    0,
    ShapeTracker.fromShape([rows, columns]),
    [...unravelAlu([rows], AluVar.gidx), AluVar.ridx],
  );
  return backend.prepareKernelSync(
    new Kernel(
      1,
      rows,
      packed
        ? AluExp.bitShift(values, AluExp.const(DType.Uint32, 4), "shr")
        : values,
      new Reduction(DType.Float32, op, columns),
    ),
  ).data[0].code;
}

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 });
  vi.stubGlobal("navigator", { userAgent: "test" });
});
afterEach(() => vi.unstubAllGlobals());

test("groups large-output reductions and uses portable subgroup leaders", () => {
  const code = source(true, 70001, 1280);
  expect(code).toContain("enable subgroups;");
  expect(code).toContain("subgroupAdd(acc0)");
  expect(code).toContain("subgroupElect()");
  expect(code).toContain("partial0[sg_id]");
  expect(code).toContain("sg < sg_count");
  expect(code).toContain("wg_id.y");
  expect(code).toContain("@workgroup_size(32)");
});

test("handles reduction lengths with smaller workgroups", () => {
  const code = source(true, 37, 520);
  expect(code).toContain("@workgroup_size(8)");
  expect(code).toContain("subgroupAdd(acc0)");
});

test("retains workgroup reduction when subgroups are unavailable", () => {
  const code = source(false, 8193, 1280);
  expect(code).not.toContain("subgroup");
  expect(code).toContain("@workgroup_size(32)");
  expect(code).toContain("partial0[lid.x]");
  expect(code).toContain("workgroupBarrier()");
});

test("does not use subgroup addition for short or max reductions", () => {
  expect(source(true, 37, 32)).not.toContain("subgroup");
  expect(source(true, 37, 1280, AluOp.Max)).not.toContain("subgroup");
});

test.each([false, true])(
  "compiles packed reductions with subgroups %s",
  (subgroups) => {
    const code = source(subgroups, 37, 1280, AluOp.Add, true);
    expect(code).toContain("@workgroup_size(32)");
    expect(code).toContain(">>");
    if (subgroups) expect(code).toContain("subgroupAdd(acc0)");
    else expect(code).toContain("workgroupBarrier()");
  },
);

test("preserves the mobile unrolling policy", () => {
  vi.stubGlobal("navigator", { userAgent: "Android Mobile" });
  const values = AluExp.globalView(
    DType.Uint32,
    0,
    ShapeTracker.fromShape([37, 1280]),
    [...unravelAlu([37], AluVar.gidx), AluVar.ridx],
  );
  const shifted = AluExp.bitShift(values, AluExp.const(DType.Uint32, 4), "shr");
  const tuned = tuneWebgpu(
    new Kernel(1, 37, shifted, new Reduction(DType.Float32, AluOp.Add, 1280)),
  );
  expect(tuned.size.unroll).toBe(1);
  expect(tuned.size.groups).toBe(32);
});

test.each([512, 520, 1280, 3328])(
  "unrolls adjacent packed values before grouping %i columns",
  (columns) => {
    const values = AluExp.globalView(
      DType.Uint32,
      0,
      ShapeTracker.fromShape([37, columns]),
      [...unravelAlu([37], AluVar.gidx), AluVar.ridx],
    );
    const shifted = AluExp.bitShift(
      values,
      AluExp.const(DType.Uint32, 4),
      "shr",
    );
    const kernel = new Kernel(
      1,
      37,
      shifted,
      new Reduction(DType.Float32, AluOp.Add, columns),
    );
    const tuned = tuneWebgpu(kernel);
    expect(tuned.size.unroll! % 4).toBe(0);
    expect(tuned.size.unroll).toBeGreaterThanOrEqual(4);
    expect(
      (tuned.size.groups ?? 1) * tuned.size.reduce * tuned.size.unroll!,
    ).toBe(columns);
    expect(tuned.size.groups).toBe(columns === 520 ? 2 : 32);
  },
);

test("does not force packed unrolling for incompatible lengths or max reductions", () => {
  for (const [columns, op] of [
    [514, AluOp.Add],
    [1280, AluOp.Max],
  ] as const) {
    const values = AluExp.globalView(
      DType.Uint32,
      0,
      ShapeTracker.fromShape([37, columns]),
      [...unravelAlu([37], AluVar.gidx), AluVar.ridx],
    );
    const shifted = AluExp.bitShift(
      values,
      AluExp.const(DType.Uint32, 4),
      "shr",
    );
    const tuned = tuneWebgpu(
      new Kernel(1, 37, shifted, new Reduction(DType.Float32, op, columns)),
    );
    if (op === AluOp.Max) expect(tuned.size.groups).toBe(1);
    else expect(tuned.size.unroll).not.toBe(4);
  }
});
