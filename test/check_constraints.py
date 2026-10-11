# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

"""A constraint set is refused by `bayests check`, by name, where it is not read.

    python check_constraints.py <bayests> <fixture directory> <scratch directory>

A constraint set says what was observed of a panel that is not observed whole
-- /data/train/constraints for the sample, /data/test/constraints for the
horizon. An algorithm that does not read one has to refuse a file that carries
one: the entries a panel does not observe hold placeholders, and a sampler that
reads /data/train/y whole would estimate from them and report a fit.

For every algorithm the generated fixtures cover, a copy of its first fixture
gets a one-row set under /data/train, /data/test and /data/forecast in turn, and
`bayests check` must exit 1 naming the algorithm and the group -- for a factor
model, which has no forecast data, its reader is what refuses the scenario. That runs
each model's own reader, which unit.constraints does not: it builds the inputs
in memory, so a reader that forgot the datasets would pass there and here
would not.

The algorithms in READS read all three: for them the row is a hard pin, of the
first entry of /data/train/y or /data/test/y at the value the file holds there,
or of the first period of the horizon, and `check` must pass. The test set goes
on a fixture that is scored, the other two on one that is not.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

# The algorithms that read /data/train/constraints.
READS = {"VarNormalWishart", "VarNormalGamma"}

# The algorithms that read a training set but not yet a test set or a scenario:
# their forecasts carry states forward. A training pin passes; the other two are
# refused, by a message naming the algorithm and the group.
READS_TRAIN = {"VarNormalStochvol", "VarTvpWishart", "VarTvpGamma", "VarTvpStochvol"}

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
    scored = {}
    for name in sorted(os.listdir(fixtures)):
        if not name.endswith(".h5"):
            continue
        path = os.path.join(fixtures, name)
        try:
            with h5py.File(path, "r") as f:
                if "model" not in f or "algorithm" not in f["model"].attrs:
                    continue  # a multi-model file
                if any(f"data/{s}/constraints" in f for s in ("train", "test", "forecast")):
                    continue  # constraints of its own already
                if bool(f["model"].attrs.get("structural", False)):
                    continue  # a structural form is refused beside constraints by design
                if "data/test/y" in f:
                    scored.setdefault(algorithm_of(f), path)
                    continue
                first.setdefault(algorithm_of(f), path)
        except OSError:
            continue  # one another test is writing

    if not first:
        print("no fixtures found in", fixtures)
        return 1

    failures = []
    for algorithm, plain in sorted(first.items()):
        for section in ("train", "test", "forecast"):
            reads = algorithm in READS or (algorithm in READS_TRAIN and section == "train")
            source = scored.get(algorithm, plain) if reads and section == "test" else plain
            target = os.path.join(scratch, "constraints.h5")
            shutil.copyfile(source, target)
            with h5py.File(target, "a") as f:
                if "posterior" in f:
                    del f["posterior"]
                row = dict(ROW)
                if reads and section == "test" and "data/test/y" not in f:
                    # No scored fixture of this algorithm to copy: realise its
                    # horizon here, as zeros -- (k, h) in dataspace terms.
                    k = int(f["model"].attrs["k"])
                    h = int(f["model"].attrs.get("h", 0))
                    f["data/test/y"] = np.zeros((k, max(h, 1)))
                if reads:
                    # A hard pin: of the first entry of y, stacked by period for
                    # the sample and one row per period for the horizon, at the
                    # value the file holds; or of the horizon's first period.
                    held = {"train": "data/train/y", "test": "data/test/y"}.get(section)
                    row["value"] = ([float(np.asarray(f[held][()]).flatten()[0])]
                                    if held else [0.5])
                    row["group"] = [0.0]
                for member, values in row.items():
                    f[f"data/{section}/constraints/{member}"] = np.array(values, dtype=float)

            run = subprocess.run([bayests, "check", target], capture_output=True, text=True)
            said = run.stdout + run.stderr
            if reads:
                if run.returncode != 0:
                    line = next((l for l in said.splitlines() if "rror" in l), "")
                    failures.append(f"{algorithm} /data/{section}/constraints "
                                    f"(from {os.path.basename(source)}): refused, exit "
                                    f"{run.returncode}; {line.strip()[:160]}")
                continue
            named = ((f"{algorithm} does not read" in said
                      or f"{algorithm} does not score or condition" in said)
                     and f"/data/{section}" in said)
            if run.returncode != 1 or not named:
                line = next((l for l in said.splitlines() if "rror" in l), "")
                failures.append(f"{algorithm} /data/{section}/constraints "
                                f"(from {os.path.basename(source)}): exit {run.returncode}; "
                                f"{line.strip()[:160]}")

    print(f"{len(first)} algorithms, {3 * len(first)} files, {len(failures)} not as expected")
    for failure in failures:
        print("  UNEXPECTED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
