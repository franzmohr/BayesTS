# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

"""A NaN in any input a model reads is refused by `bayests check`, by name.

    python check_nonfinite.py <bayests> <fixture directory> <scratch directory>

For every algorithm the generated fixtures cover, and every floating-point
dataset under /data, /priors or /initial that any of that algorithm's fixtures
carries, a copy of the first such fixture gets one NaN in that dataset, and
`bayests check` must exit 1 saying that something "must be finite".

Every (algorithm, dataset) pair once rather than every fixture: the fixtures of
one algorithm differ in which datasets they carry, not in how a dataset is
read, so the pairs are what coverage is measured in. A new dataset a model
learns to read is picked up by the first fixture that writes it, with nothing
here to update -- which is the point. A NaN that got through to the numerics
used to surface as "inv_sympd(): matrix is singular", or, in a forecast or a
score, not at all.

Integer datasets are covered too, rewritten as doubles: R writes a whole
number as a double unless told otherwise, and the reader accepts one, so a
count like /priors/u_sigma/df can arrive holding a NaN. That path used to cast
the NaN to an int -- undefined behaviour -- because every comparison with NaN
is false and the whole-number test let it through. Either way the message
has to name the problem: "finite", or the NaN itself as a range check on a
scalar shows it.
"""

import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

SECTIONS = ("data", "priors", "initial")

# Refused by name: the message says the value must be finite, or shows the NaN
# it was given, as a range check on a scalar like rho does ("got nan"). Anything
# else -- an exit 0, or a failure deep in the numerics -- is what this guards
# against. Whole words: "not positive definite", the message a NaN used to
# produce, contains "finite".
REFUSED = re.compile(r"\bfinite\b|\bnan\b", re.IGNORECASE)


def model_group(f):
    """The file's model: at the root, or in the one group --group would name."""
    if "model" in f and "algorithm" in f["model"].attrs:
        return ""
    raise LookupError("no /model at the root")


def algorithm_of(f):
    value = f["model"].attrs["algorithm"]
    return value.decode() if isinstance(value, bytes) else str(value)


def input_datasets(f):
    found = []
    for section in SECTIONS:
        if section not in f:
            continue

        def visit(name, obj, section=section):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiu" and obj.size > 0:
                found.append(f"/{section}/{name}")

        f[section].visititems(visit)
    return found


def main():
    bayests, fixtures, scratch = sys.argv[1:4]
    os.makedirs(scratch, exist_ok=True)

    # (algorithm, dataset) -> the first fixture that carries it
    pairs = {}
    for name in sorted(os.listdir(fixtures)):
        if not name.endswith(".h5"):
            continue
        path = os.path.join(fixtures, name)
        try:
            with h5py.File(path, "r") as f:
                model_group(f)
                algorithm = algorithm_of(f)
                for dataset in input_datasets(f):
                    pairs.setdefault((algorithm, dataset), path)
        except (LookupError, OSError):
            continue  # a multi-model file, or one another test is writing

    if not pairs:
        print("no fixtures found in", fixtures)
        return 1

    failures = []
    for (algorithm, dataset), source in sorted(pairs.items()):
        target = os.path.join(scratch, "nonfinite.h5")
        shutil.copyfile(source, target)
        with h5py.File(target, "a") as f:
            if "posterior" in f:
                del f["posterior"]
            # np.array rather than .astype(): a scalar dataset reads back as a
            # NumPy scalar, which is immutable, and .flat[0] = nan on one
            # silently changes nothing -- the NaN would never reach the file.
            values = np.array(f[dataset][()], dtype=float)
            values.flat[0] = np.nan
            attrs = dict(f[dataset].attrs)
            del f[dataset]
            f[dataset] = values
            for key, value in attrs.items():
                f[dataset].attrs[key] = value

        run = subprocess.run([bayests, "check", target], capture_output=True, text=True)
        said = run.stdout + run.stderr
        refused = bool(REFUSED.search(said))
        if run.returncode != 1 or not refused:
            line = next((l for l in said.splitlines() if "rror" in l or "arning" in l), "")
            failures.append(f"{algorithm} {dataset} (from {os.path.basename(source)}): "
                            f"exit {run.returncode}; {line.strip()[:160]}")

    print(f"{len(pairs)} (algorithm, dataset) pairs, {len(failures)} not refused")
    for failure in failures:
        print("  NOT REFUSED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
