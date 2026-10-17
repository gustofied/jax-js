import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { AluExp, AluOp, AluVar, DType, Kernel, Reduction } from "../alu";
import { ShapeTracker, unravelAlu } from "../shape";
import { WebGPUBackend } from "./webgpu";

function source(
  subgroups: boolean,
  rows: number,
  columns: number,
  op = AluOp.Add,
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
    DType.Float32,
    0,
    ShapeTracker.fromShape([rows, columns]),
    [...unravelAlu([rows], AluVar.gidx), AluVar.ridx],
  );
  return backend.prepareKernelSync(
    new Kernel(1, rows, values, new Reduction(DType.Float32, op, columns)),
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
