# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

"""A constraint set is refused by `bayests check`, by name, for every algorithm.

    python check_constraints.py <bayests> <fixture directory> <scratch directory>

A constraint set says what was observed of a panel that is not observed whole
-- /data/train/constraints for the sample, /data/test/constraints for the
horizon. No algorithm reads one yet, and each has to refuse a file that carries
one: the entries a panel does not observe hold placeholders, and a sampler that
reads /data/train/y whole would estimate from them and report a fit.

For every algorithm the generated fixtures cover, a copy of its first fixture
gets a one-row set, first under /data/train and then under /data/test, and
`bayests check` must exit 1 naming the algorithm and the datasets. That runs
each model's own reader, which unit.constraints does not: it builds the inputs
in memory, so a reader that forgot the datasets would pass there and here
would not.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

# The one row: a noisy reading of the first variable in the first period. Soft,
# so that nothing about it depends on what the fixture's y holds.
ROW = {
    "value": [0.0],
    "group": [1.0],
    "row": [1.0],
    "period": [1.0],
    "variable": [1.0],
    "weight": [1.0],
}


def algorithm_of(f):
    value = f["model"].attrs["algorithm"]
    return value.decode() if isinstance(value, bytes) else str(value)


def main():
    bayests, fixtures, scratch = sys.argv[1:4]
    os.makedirs(scratch, exist_ok=True)

    # algorithm -> its first fixture with the model at the root
    first = {}
    for name in sorted(os.listdir(fixtures)):
        if not name.endswith(".h5"):
            continue
        path = os.path.join(fixtures, name)
        try:
            with h5py.File(path, "r") as f:
                if "model" not in f or "algorithm" not in f["model"].attrs:
                    continue  # a multi-model file
                first.setdefault(algorithm_of(f), path)
        except OSError:
            continue  # one another test is writing

    if not first:
        print("no fixtures found in", fixtures)
        return 1

    failures = []
    for algorithm, source in sorted(first.items()):
        for section in ("train", "test"):
            target = os.path.join(scratch, "constraints.h5")
            shutil.copyfile(source, target)
            with h5py.File(target, "a") as f:
                if "posterior" in f:
                    del f["posterior"]
                for member, values in ROW.items():
                    f[f"data/{section}/constraints/{member}"] = np.array(values, dtype=float)

            run = subprocess.run([bayests, "check", target], capture_output=True, text=True)
            said = run.stdout + run.stderr
            named = (f"{algorithm} does not read /data/train/constraints" in said)
            if run.returncode != 1 or not named:
                line = next((l for l in said.splitlines() if "rror" in l), "")
                failures.append(f"{algorithm} /data/{section}/constraints "
                                f"(from {os.path.basename(source)}): exit {run.returncode}; "
                                f"{line.strip()[:160]}")

    print(f"{len(first)} algorithms, {2 * len(first)} files, {len(failures)} not refused")
    for failure in failures:
        print("  NOT REFUSED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
