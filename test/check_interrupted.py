# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

"""A run of `coefficients` stopped while writing is estimated again, not kept.

    python check_interrupted.py <bayests> <fixture directory> <scratch directory>

`coefficients` skips a model whose posterior is already there, and "there" used
to mean one dataset: in half the models not the last the stage writes, in the
two discounted ones the first. A run stopped between two writes left a file
whose rerun skipped and whose later stages failed, for good. The stage now
marks /posterior "writing" before its first write and "complete" after its
last. For one fixture of every algorithm this checks that:

  * a run leaves the mark at "complete", and a second run skips;
  * a file still marked "writing" is estimated again, saying so, and ends
    "complete" -- and `bayests check` says beforehand that it will be;
  * `loglik` refuses a file marked "writing" -- except on the two discounted
    models, whose log likelihood is recomputed from the data in closed form
    and never reads the stored posterior;
  * a file with a posterior and no mark at all -- one written before the mark
    existed -- still skips, as it always did.

The kill itself is not reproduced: which write it lands between is a race. What
it leaves is the mark at "writing", and that is set here directly.
"""

import os
import shutil
import subprocess
import sys

import h5py

CLOSED_FORM = {"VarTvpDiscount", "VecTvpDiscount"}


def algorithm_of(path):
    with h5py.File(path, "r") as f:
        if "model" not in f or "algorithm" not in f["model"].attrs:
            return None
        value = f["model"].attrs["algorithm"]
    return value.decode() if isinstance(value, bytes) else str(value)


def marker(path):
    with h5py.File(path, "r") as f:
        if "posterior" not in f:
            return None
        value = f["posterior"].attrs.get("coefficients")
    return value.decode() if isinstance(value, bytes) else value


def set_marker(path, value):
    with h5py.File(path, "a") as f:
        if value is None:
            del f["posterior"].attrs["coefficients"]
        else:
            f["posterior"].attrs["coefficients"] = value


def main():
    bayests, fixtures, scratch = sys.argv[1:4]
    os.makedirs(scratch, exist_ok=True)

    # The first fixture of every algorithm, by name.
    chosen = {}
    for name in sorted(os.listdir(fixtures)):
        if name.endswith(".h5"):
            path = os.path.join(fixtures, name)
            try:
                algorithm = algorithm_of(path)
            except OSError:
                continue
            if algorithm:
                chosen.setdefault(algorithm, path)

    failures = []

    def run(*args):
        p = subprocess.run([bayests, *args], capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def expect(ok, algorithm, what, said=""):
        if not ok:
            line = next((l for l in said.splitlines() if l.strip()), "")
            failures.append(f"{algorithm}: {what}  | {line[:150]}")

    for algorithm, source in sorted(chosen.items()):
        path = os.path.join(scratch, "interrupted.h5")
        shutil.copyfile(source, path)
        with h5py.File(path, "a") as f:
            if "posterior" in f:
                del f["posterior"]

        code, said = run("coefficients", path)
        expect(code == 0 and marker(path) == "complete", algorithm,
               f"a run should leave the mark at complete, got exit {code} and {marker(path)!r}", said)

        code, said = run("coefficients", path)
        expect(code == 0 and "Skipping" in said, algorithm, "a second run should skip", said)

        set_marker(path, "writing")
        code, said = run("check", path)
        expect(code == 0 and "estimate it again" in said, algorithm,
               "check should say a file marked writing will be estimated again", said)

        code, said = run("loglik", path)
        if algorithm in CLOSED_FORM:
            expect(code == 0, algorithm, "loglik recomputes in closed form and should run", said)
        else:
            expect(code == 1 and "did not finish" in said, algorithm,
                   f"loglik on a file marked writing should refuse, got exit {code}", said)

        set_marker(path, "writing")
        code, said = run("coefficients", path)
        expect(code == 0 and "Estimating again" in said and marker(path) == "complete",
               algorithm, f"a file marked writing should be estimated again, got exit {code} "
               f"and {marker(path)!r}", said)

        set_marker(path, None)
        code, said = run("coefficients", path)
        expect(code == 0 and "Skipping" in said and marker(path) is None, algorithm,
               "a file from before the mark should still skip, untouched", said)

    print(f"{len(chosen)} algorithms, {len(failures)} failures")
    for failure in failures:
        print("  FAIL:", failure)
    return 1 if failures or not chosen else 0


if __name__ == "__main__":
    sys.exit(main())
