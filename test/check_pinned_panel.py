# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

"""A panel pinned whole is estimated as the complete panel, by every algorithm
that completes one.

    python check_pinned_panel.py <bayests> <fixture directory> <scratch directory>

An algorithm that reads /data/train/constraints starts every sweep by drawing
what the panel did not observe. Pinned whole -- a hard row of one entry for
every entry of /data/train/y, at the value it holds -- there is nothing to draw:
the completion step consumes no random number and rebuilds the regressors the
file already carries. So the chain has to be the complete panel's, bit for bit,
and the log likelihood, which a panel with constraints computes as the density
of what was observed by a filter, has to agree with the complete panel's to
rounding.

For every such algorithm, a copy of its first fixture without constraints or a
score is run through `bayests coefficients` and `bayests loglik` twice: as it is
and pinned whole. Every /posterior dataset the first writes has to match in the
second -- exactly, the log likelihood to 1e-9 relative -- and the second's
completed panel, /posterior/y/coeffs, has to be the data in every draw. That
runs each model's reader and writer, which the unit tests building inputs in
memory do not.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

# The algorithms that complete a panel not observed whole.
COMPLETES = ("VarNormalWishart", "VarNormalGamma", "VarNormalStochvol", "VarTvpWishart",
             "VarTvpGamma", "VarTvpStochvol")


def algorithm_of(f):
    value = f["model"].attrs["algorithm"]
    return value.decode() if isinstance(value, bytes) else str(value)


def run(bayests, command, path):
    done = subprocess.run([bayests, command, path], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{command} {os.path.basename(path)} exited {done.returncode}: "
                           f"{(done.stdout + done.stderr).strip()[-300:]}")


def posterior(path):
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = np.array(obj[()])
        if "posterior" in f:
            f["posterior"].visititems(visit)
    return out


def pin_whole(path):
    with h5py.File(path, "a") as f:
        k = int(f["model"].attrs["k"])
        y = np.asarray(f["data/train/y"][()], dtype=float).flatten()  # stacked by period
        n = y.size
        group = f.create_group("data/train/constraints")
        group["value"] = y
        group["group"] = np.zeros(n)
        group["row"] = np.arange(1, n + 1, dtype=float)
        group["period"] = np.arange(n) // k + 1.0
        group["variable"] = np.arange(n) % k + 1.0
        group["weight"] = np.ones(n)
    return y


def main():
    bayests, fixtures, scratch = sys.argv[1:4]
    os.makedirs(scratch, exist_ok=True)

    first = {}
    for name in sorted(os.listdir(fixtures)):
        if not name.endswith(".h5"):
            continue
        path = os.path.join(fixtures, name)
        try:
            with h5py.File(path, "r") as f:
                if "model" not in f or "algorithm" not in f["model"].attrs:
                    continue
                if any(f"data/{s}/constraints" in f for s in ("train", "test", "forecast")):
                    continue
                if "data/test/y" in f:
                    continue
                algorithm = algorithm_of(f)
                if algorithm in COMPLETES:
                    first.setdefault(algorithm, path)
        except OSError:
            continue

    missing = [a for a in COMPLETES if a not in first]
    if missing:
        print("no plain fixture for", ", ".join(missing))
        return 1

    failures = []
    for algorithm, source in sorted(first.items()):
        plain = os.path.join(scratch, f"{algorithm}-plain.h5")
        pinned = os.path.join(scratch, f"{algorithm}-pinned.h5")
        for target in (plain, pinned):
            shutil.copyfile(source, target)
            with h5py.File(target, "a") as f:
                if "posterior" in f:
                    del f["posterior"]
        y = pin_whole(pinned)

        try:
            for target in (plain, pinned):
                run(bayests, "coefficients", target)
                run(bayests, "loglik", target)
        except RuntimeError as e:
            failures.append(f"{algorithm}: {e}")
            continue

        a, b = posterior(plain), posterior(pinned)
        for name, values in a.items():
            if name not in b:
                failures.append(f"{algorithm}: /posterior/{name} not written when pinned")
            elif name == "loglik":
                scale = max(1.0, float(np.abs(values).max()))
                if values.shape != b[name].shape or float(np.abs(values - b[name]).max()) > 1e-9 * scale:
                    failures.append(f"{algorithm}: the log likelihood moved when pinned")
            elif values.shape != b[name].shape or not np.array_equal(values, b[name]):
                failures.append(f"{algorithm}: /posterior/{name} moved when pinned")
        completed = b.get("y/coeffs")
        if completed is None:
            failures.append(f"{algorithm}: no /posterior/y/coeffs when pinned")
        else:
            draws = completed.reshape(completed.shape[0], -1) if completed.ndim > 1 else completed
            # One row per quantity and one column per draw, in dataspace terms.
            if not np.array_equal(draws, np.repeat(y[:, None], draws.shape[1], axis=1)):
                failures.append(f"{algorithm}: the completed panel is not the data")

    print(f"{len(first)} algorithms, {len(failures)} failures")
    for failure in failures:
        print("  FAILED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
