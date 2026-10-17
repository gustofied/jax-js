import { beforeEach, expect, suite, test } from "vitest";

import { defaultDevice, init, jit, numpy as np, tree } from "../index";

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

  test("fuses reshape and broadcast arithmetic after a reduction", async () => {
    const f = jit((x: np.Array, scale: np.Array) =>
      x.reshape([8, 16]).sum(1).reshape([2, 4]).mul(scale).add(1),
    );
    const output = f(np.arange(128).astype(np.float32), np.array([[2], [3]]));
    expect(output.shape).toEqual([2, 4]);
    expect(Array.from(await output.data())).toEqual(
      Array.from(
        { length: 8 },
        (_, i) => (256 * i + 120) * (i < 4 ? 2 : 3) + 1,
      ),
    );
  });

  test("keeps a second reduction separate after a reshaped epilogue", async () => {
    const f = jit((x: np.Array) =>
      x.reshape([8, 16]).sum(1).reshape([2, 4]).mul(2).sum(0),
    );
    expect(
      Array.from(await f(np.arange(128).astype(np.float32)).data()),
    ).toEqual([2528, 3552, 4576, 5600]);
  });

  test("retains transpose indexing after a reshaped reduction", async () => {
    const f = jit((x: np.Array) =>
      x.reshape([8, 16]).sum(1).reshape([2, 4]).transpose().add(1),
    );
    expect(
      Array.from(await f(np.arange(128).astype(np.float32)).data()),
    ).toEqual([121, 1145, 377, 1401, 633, 1657, 889, 1913]);
  });

  test("keeps pending JIT work alive when sibling outputs are disposed", async () => {
    const f = jit((x: np.Array) => [x.ref.mul(2), x.add(7)]);
    const [discarded, kept] = f(np.arange(1024).astype(np.float32));
    discarded.dispose();
    expect(Array.from(await kept.data())).toEqual(
      Array.from({ length: 1024 }, (_, i) => i + 7),
    );
  });

  test("submits shared outputs once when read concurrently", async () => {
    const f = jit((x: np.Array) => [x.ref.mul(2), x.add(7)]);
    const values = await Promise.all(
      f(np.arange(1024).astype(np.float32)).map((x) => x.data()),
    );
    expect(Array.from(values[0])).toEqual(
      Array.from({ length: 1024 }, (_, i) => i * 2),
    );
    expect(Array.from(values[1])).toEqual(
      Array.from({ length: 1024 }, (_, i) => i + 7),
    );
  });

  test("preserves differently shaped shared outputs", async () => {
    const f = jit((x: np.Array) => [
      x.ref.reshape([32, 32]).transpose(),
      x.add(7),
    ]);
    const [view, sibling] = f(np.arange(1024).astype(np.float32));
    const [a, b] = await Promise.all([view.data(), sibling.data()]);
    expect(Array.from(a)).toEqual(
      Array.from(
        { length: 1024 },
        (_, i) => (i % 32) * 32 + Math.floor(i / 32),
      ),
    );
    expect(Array.from(b)).toEqual(
      Array.from({ length: 1024 }, (_, i) => i + 7),
    );
  });

  test("cancels a complete output family without affecting the next call", async () => {
    const f = jit((x: np.Array) => [x.ref.mul(2), x.add(7)]);
    tree.dispose(f(np.arange(1024).astype(np.float32)));
    const [a, b] = f(np.arange(1024).astype(np.float32));
    a.dispose();
    expect(Array.from(await b.data())).toEqual(
      Array.from({ length: 1024 }, (_, i) => i + 7),
    );
  });
});
