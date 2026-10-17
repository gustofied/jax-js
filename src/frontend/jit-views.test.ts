import { beforeEach, expect, suite, test } from "vitest";

import { defaultDevice, init, jit, numpy as np } from "../index";

const available = await init();
suite.each(available)("shared JIT views on %s", (device) => {
  beforeEach(() => defaultDevice(device));

  test("shares a computed producer through differently reduced reshape branches", async () => {
    const f = jit((x: np.Array) => {
      const view = x.mul(2).add(3).reshape([4, 8]);
      return [view.ref.sum(0), view.sum(1)];
    });
    const [columns, rows] = f(np.arange(32).astype(np.float32));
    expect(Array.from(await columns.data())).toEqual([
      108, 116, 124, 132, 140, 148, 156, 164,
    ]);
    expect(Array.from(await rows.data())).toEqual([80, 208, 336, 464]);
  });

  test("preserves transpose, flip and sliced indexing on shared views", async () => {
    const f = jit((x: np.Array) => {
      const view = np
        .flip(x.mul(2).reshape([4, 8]).transpose(), 0)
        .slice([1, 7]);
      return [view.ref.add(1).sum(1), view.mul(2).sum(0)];
    });
    const [rows, columns] = f(np.arange(32).astype(np.float32));
    expect(Array.from(await rows.data())).toEqual([
      148, 140, 132, 124, 116, 108,
    ]);
    expect(Array.from(await columns.data())).toEqual([84, 276, 468, 660]);
  });

  test("keeps padding outside a computed producer", async () => {
    const f = jit((x: np.Array) => {
      const view = x.mul(2).add(3).reshape([2, 4]);
      return [view.ref.sum(0), np.pad(view, { 1: [1, 1] }).sum(1)];
    });
    const [columns, rows] = f(np.arange(8).astype(np.float32));
    expect(Array.from(await columns.data())).toEqual([14, 18, 22, 26]);
    expect(Array.from(await rows.data())).toEqual([24, 56]);
  });

  test("preserves a view that is itself returned and consumed", async () => {
    const f = jit((x: np.Array) => {
      const view = x.add(1).reshape([2, 4]);
      return [view.ref, view.sum(0)];
    });
    const [view, sums] = f(np.arange(8).astype(np.float32));
    expect(Array.from(await view.data())).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(await sums.data())).toEqual([6, 8, 10, 12]);
  });
});
