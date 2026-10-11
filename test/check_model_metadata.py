# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr

"""A host's metadata below /model is not an unread dataset; anything else still is.

    python check_model_metadata.py <bayests> <fixture directory> <scratch directory>

`bayests check` warns about every dataset in a model's tree its reader never
opened, because one nobody reads is usually the trace of a different model. A
host keeps metadata of its own in groups below /model -- bvartools'
write_to_hdf5() stores a sign-restriction identification as
/model/sign_restrictions/restrictions beside the rotations in /posterior/q, and
BayesTS Studio writes the same layout -- and no reader opens a dataset anywhere
in /model, so check leaves those groups out (list_model_datasets()).

For every algorithm the generated fixtures cover, a copy of its first fixture
gets that group and must check without a warning, through each model's own
reader. A second copy carries the group too, plus a dataset directly in /model,
where the readers look for attributes, and one in /initial: check must warn
about exactly those two, so the rule stays as narrow as it says. The grouped
fixture runs the same pair under --group.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

# bvartools' layout: one column per restriction, rows the one-based impulse and
# response positions, the sign and the horizon -- (4, n) in dataspace terms.
COLUMNS = ["impulse", "response", "sign", "horizon"]
RESTRICTIONS = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, -1.0], [0.0, 0.0]])

# The datasets that must still be named, model-relative.
STRAY = ["/model/stray", "/initial/stray"]


def algorithm_of(group):
    value = group["model"].attrs["algorithm"]
    return value.decode() if isinstance(value, bytes) else str(value)


def add_metadata(model):
    restrictions = model.create_group("model/sign_restrictions")
    restrictions.attrs["max_tries"] = 1000
    restrictions.attrs["period"] = 1
    dataset = restrictions.create_dataset("restrictions", data=RESTRICTIONS)
    dataset.attrs["columns"] = COLUMNS


def warnings_of(text):
    return [line.strip() for line in text.splitlines() if "warning:" in line]


def run_check(bayests, target, group):
    command = [bayests, "check", target] + (["--group", group] if group else [])
    run = subprocess.run(command, capture_output=True, text=True)
    return run.returncode, run.stdout + run.stderr


def main():
    bayests, fixtures, scratch = sys.argv[1:4]
    os.makedirs(scratch, exist_ok=True)

    # algorithm -> its first fixture with the model at the root, and one model
    # under a group of a multi-model file, as (path, group).
    cases = {}
    grouped = None
    for name in sorted(os.listdir(fixtures)):
        if not name.endswith(".h5"):
            continue
        path = os.path.join(fixtures, name)
        try:
            with h5py.File(path, "r") as f:
                if "model" in f and "algorithm" in f["model"].attrs:
                    cases.setdefault(algorithm_of(f), (path, ""))
                elif grouped is None:
                    def find(name, obj):
                        if isinstance(obj, h5py.Group) and "model" in obj and \
                                "algorithm" in obj["model"].attrs:
                            return "/" + name
                        return None
                    group = f.visititems(find)
                    if group:
                        grouped = (path, group)
        except OSError:
            continue  # one another test is writing

    if not cases:
        print("no fixtures found in", fixtures)
        return 1
    if grouped:
        cases["grouped " + grouped[1]] = grouped

    failures = []
    for label, (source, group) in sorted(cases.items()):
        target = os.path.join(scratch, "metadata.h5")

        shutil.copyfile(source, target)
        with h5py.File(target, "a") as f:
            add_metadata(f[group] if group else f)
        code, said = run_check(bayests, target, group)
        if code != 0 or warnings_of(said):
            failures.append(f"{label} with /model/sign_restrictions "
                            f"(from {os.path.basename(source)}): exit {code}; "
                            f"{'; '.join(warnings_of(said))[:200]}")

        shutil.copyfile(source, target)
        with h5py.File(target, "a") as f:
            model = f[group] if group else f
            add_metadata(model)
            for stray in STRAY:
                model[stray.lstrip("/")] = np.zeros(2)
        code, said = run_check(bayests, target, group)
        warned = warnings_of(said)
        expected = [f"{stray} is in the file" for stray in STRAY]
        named = all(any(e in w for w in warned) for e in expected)
        if code != 0 or not named or len(warned) != len(STRAY):
            failures.append(f"{label} with {' and '.join(STRAY)} "
                            f"(from {os.path.basename(source)}): exit {code}; "
                            f"{'; '.join(warned)[:200]}")

    print(f"{len(cases)} models, {2 * len(cases)} files, {len(failures)} not as expected")
    for failure in failures:
        print("  UNEXPECTED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
