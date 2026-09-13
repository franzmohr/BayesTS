#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Franz X. Mohr
"""Run the Python examples in the agent documentation against a built bayests.

    check_agent_recipes.py <bayests> <bayests_make_model_fixture> <skill dir> <work dir>

agents/skills/bayests/ is what a coding agent copies into someone else's
project. An example there that no longer matches the file format does not fail
here -- it fails later, silently, as a model that runs and means something
else. So this executes each example, runs the sampler over the file it wrote,
and checks the shapes the text states against the ones the run produced.

Every Python fence in the skill has to be claimed by a scenario below, and one
that nothing claims fails the check: a new example is either verified or
reported, never quietly skipped. Examples are addressed by file, section
heading and position within the section, so renaming a heading fails here too
-- update the address when you do.

Every file an example writes also has to pass `bayests check` with no
warnings: an example that leaves a dataset the model never reads, or an
attribute no model looks at, is teaching the reader to write one.

What this does not cover: the R snippet in results.md, the bash fences other
than the generator's, and the tables. A table claim is checked only where an
example below reads the dataset it describes.

Exits 0 when every scenario passed and every example was run, 1 otherwise.
"""

import contextlib
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import traceback

import h5py
import numpy as np

VAR = ("references/recipes.md", "A complete VAR from h5py")
RESULTS = ("references/results.md", "From Python")


def read_fences(skill_dir):
    """{(file, heading, language, index): code} for every fence under skill_dir.

    `index` counts fences of the same language within one `## ` section, so the
    second Python example under a heading is index 1.
    """
    fences = {}
    for path in sorted(skill_dir.rglob("*.md")):
        name = path.relative_to(skill_dir).as_posix()
        heading, counts, language, lines = "", {}, None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if language is None:
                if line.startswith("```"):
                    language, lines = line[3:].strip(), []
                elif line.startswith("## "):
                    heading = line[3:].strip()
            elif line.startswith("```"):
                key = (name, heading, language)
                index = counts.get(key, 0)
                counts[key] = index + 1
                fences[(name, heading, language, index)] = "\n".join(lines) + "\n"
                language = None
            else:
                lines.append(line)
    return fences


class _Replacing(h5py.File):
    """An h5py.File on which assigning to an existing path replaces it.

    Only for an example the text says to write *in place of* lines of another;
    everywhere else a duplicate write is a documentation bug and must raise.
    """

    def __setitem__(self, name, obj):
        if name in self:
            del self[name]
        super().__setitem__(name, obj)


@contextlib.contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def inside_with(base, *additions):
    """`base` with each addition appended to the `with` block it ends in."""
    last = [line for line in base.splitlines() if line.strip()][-1]
    if not last.startswith("    "):
        raise AssertionError(
            f"{VAR[0]} '## {VAR[1]}' no longer ends inside its `with` block, which "
            "the examples that add to that block rely on")
    return base + "".join("\n" + textwrap.indent(code, "    ") for code in additions)


def expect_shapes(path, expected):
    problems = []
    with h5py.File(path, "r") as f:
        for dataset, shape in expected.items():
            if dataset not in f:
                problems.append(f"{dataset}: absent, documented as {shape}")
            elif f[dataset].shape != shape:
                problems.append(f"{dataset}: {f[dataset].shape}, documented as {shape}")
    if problems:
        raise AssertionError(f"{path.name}:\n  " + "\n  ".join(problems))


def read(path, dataset):
    with h5py.File(path, "r") as f:
        return f[dataset][:]


def posterior_arrays(path):
    """{name: array} for every dataset under /posterior."""
    arrays = {}
    with h5py.File(path, "r") as f:
        f["posterior"].visititems(
            lambda name, obj: arrays.__setitem__(name, obj[:]) if isinstance(obj, h5py.Dataset) else None)
    return arrays


class Checker:
    def __init__(self, bayests, generator, skill_dir, work_dir):
        self.bayests_exe = bayests
        self.generator = generator
        self.work_dir = work_dir
        self.fences = read_fences(skill_dir)
        self.claimed = set()

    # -- the examples --------------------------------------------------------

    def code(self, file, heading, index=0, language="python"):
        key = (file, heading, language, index)
        if key not in self.fences:
            raise LookupError(
                f"{file}: no {language} example {index + 1} under '## {heading}' -- "
                "was it renamed, moved or removed?")
        self.claimed.add(key)
        return self.fences[key]

    def run_python(self, directory, code, namespace=None, replacing=False):
        namespace = {} if namespace is None else namespace
        original = h5py.File
        with _working_directory(directory):
            if replacing:
                h5py.File = _Replacing
            try:
                exec(compile(code, "<documentation example>", "exec"), namespace)
            finally:
                h5py.File = original
        return namespace

    # -- the binary ----------------------------------------------------------

    def bayests(self, directory, *args):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        return subprocess.run([self.bayests_exe, *args], cwd=directory, env=env,
                              capture_output=True, text=True)

    def check_clean(self, directory, name):
        """`bayests check` accepts the file and has nothing to warn about."""
        result = self.bayests(directory, "check", name)
        warnings = [line for line in result.stdout.splitlines() if "warning:" in line]
        if result.returncode != 0 or warnings:
            raise AssertionError(
                f"bayests check {name} exited {result.returncode} with "
                f"{len(warnings)} warning(s)\n{result.stdout}{result.stderr}")

    def posterior(self, directory, name):
        result = self.bayests(directory, "posterior", name)
        if result.returncode != 0:
            raise AssertionError(
                f"bayests posterior {name} exited {result.returncode}\n"
                f"{result.stdout}{result.stderr}")

    # -- the scenarios -------------------------------------------------------

    def scenario_var(self, d):
        ns = self.run_python(d, self.code(*VAR, 0))
        k, nparams, tt, h, it = (ns[v] for v in ("k", "nparams", "tt", "h", "iterations"))
        self.check_clean(d, "var.h5")
        self.posterior(d, "var.h5")
        expect_shapes(d / "var.h5", {
            "/posterior/a/coeffs": (nparams, it),
            "/posterior/u_sigma_inv/coeffs": (k * k, it),
            "/posterior/u_omega_inv/coeffs": (k, it),
            "/posterior/forecast": (h * k, it),
            "/posterior/loglik": (tt, it),
        })

        self.run_python(d, self.code(*VAR, 1), ns)
        diagonal = np.diag(ns["A1"])
        if not np.all((diagonal > 0.2) & (diagonal < 0.8)):
            raise AssertionError(
                f"the diagonal of A1 came back {diagonal}; the example simulated 0.5 "
                "and the text says it comes back near that")

        # results.md reads a file called model.h5.
        shutil.copy(d / "var.h5", d / "model.h5")
        results = self.run_python(d, self.code(*RESULTS, 0))
        if results["post_mean"].shape != (nparams,):
            raise AssertionError(f"results.md: post_mean is {results['post_mean'].shape}")
        results = self.run_python(d, self.code(*RESULTS, 2))
        if results["fcst"].shape != (h, k, it):
            raise AssertionError(f"results.md: the forecast reshapes to {results['fcst'].shape}")

        # A second run finds the posterior and leaves it alone.
        before = read(d / "var.h5", "/posterior/a/coeffs")
        second = self.bayests(d, "posterior", "var.h5")
        if second.returncode != 0:
            raise AssertionError(f"a second run exited {second.returncode}")
        if not np.array_equal(before, read(d / "var.h5", "/posterior/a/coeffs")):
            raise AssertionError("a second run re-estimated, where the text says it skips")

        self.run_python(d, self.code("references/recipes.md", "Re-running", 0))
        self.run_python(d, self.code("references/pipeline.md", "Re-running does nothing", 0))
        for name in ("var.h5", "model.h5"):
            with h5py.File(d / name, "r") as f:
                if "posterior" in f:
                    raise AssertionError(f"{name} still has /posterior after deleting it")

    def scenario_covariance_block(self, d):
        covar = self.code("references/recipes.md", "Adding a covariance block", 0)
        ns = self.run_python(d, inside_with(self.code(*VAR, 0), covar))
        self.check_clean(d, "var.h5")
        self.posterior(d, "var.h5")
        k, it = ns["k"], ns["iterations"]
        expect_shapes(d / "var.h5", {"/posterior/psi/coeffs": (k * k, it)})

    def scenario_variable_selection(self, d):
        selection = self.code("references/recipes.md", "Adding variable selection", 0)
        ns = self.run_python(d, inside_with(self.code(*VAR, 0), selection))
        self.check_clean(d, "var.h5")
        self.posterior(d, "var.h5")
        expect_shapes(d / "var.h5", {"/posterior/a/lambda": (ns["nparams"], ns["iterations"])})

    def scenario_selection_with_covariance_block(self, d):
        base = self.code(*VAR, 0)
        covar = self.code("references/recipes.md", "Adding a covariance block", 0)
        selection = self.code("references/recipes.md", "Adding variable selection", 0)
        psi_selection = self.code("references/recipes.md", "Adding variable selection", 1)

        # The failure the text promises when the psi datasets are left out --
        # which check has to predict, since the run makes it.
        self.run_python(d, inside_with(base, covar, selection))
        checked = self.bayests(d, "check", "var.h5")
        if checked.returncode != 1 or "psi_lambda" not in checked.stderr:
            raise AssertionError(
                "bayests check accepted a file the run refuses for want of /initial/psi_lambda\n"
                f"{checked.stdout}{checked.stderr}")
        result = self.bayests(d, "posterior", "var.h5")
        output = result.stdout + result.stderr
        if result.returncode != 1 or "psi_lambda" not in output:
            raise AssertionError(
                "without the psi selection datasets the text promises exit 1 naming "
                f"/initial/psi_lambda; got exit {result.returncode}\n{output}")

        ns = self.run_python(d, inside_with(base, covar, selection, psi_selection))
        self.check_clean(d, "var.h5")
        self.posterior(d, "var.h5")
        k, nparams, it = ns["k"], ns["nparams"], ns["iterations"]
        expect_shapes(d / "var.h5", {
            "/posterior/a/lambda": (nparams, it),
            "/posterior/psi/lambda": (k * k, it),
        })

    def scenario_time_varying(self, d):
        tvp = self.code("references/recipes.md", "A time-varying model", 0)
        ns = self.run_python(d, inside_with(self.code(*VAR, 0), tvp), replacing=True)
        # The text says to write these in place of the VAR's lines, one of which
        # the example has no dataset to overwrite with.
        with h5py.File(d / "var.h5", "a") as f:
            del f["/initial/u_sigma_inv"]
        self.check_clean(d, "var.h5")
        self.posterior(d, "var.h5")
        k, nparams, tt, h, it = (ns[v] for v in ("k", "nparams", "tt", "h", "iterations"))
        expect_shapes(d / "var.h5", {
            "/posterior/a/coeffs": (nparams * tt, it),
            "/posterior/a/sigma": (nparams, it),
            "/posterior/forecast": (h * k, it),
            "/posterior/loglik": (tt, it),
        })

        ns["a"] = read(d / "var.h5", "/posterior/a/coeffs")
        self.run_python(d, self.code("references/recipes.md", "A time-varying model", 1), ns)
        if ns["a_path"].shape != (tt, nparams, it):
            raise AssertionError(f"recipes.md: a_path is {ns['a_path'].shape}")

        shutil.copy(d / "var.h5", d / "model.h5")
        results = self.run_python(d, self.code(*RESULTS, 1))
        if results["a_path"].shape != (tt, nparams, it):
            raise AssertionError(f"results.md: a_path is {results['a_path'].shape}")

    def scenario_vec(self, d):
        ns = self.run_python(d, self.code("references/recipes.md", "A VEC", 0))
        self.check_clean(d, "vec.h5")
        self.posterior(d, "vec.h5")
        k, nparams, tt, h, it = (ns[v] for v in ("k", "nparams", "tt", "h", "iterations"))
        expect_shapes(d / "vec.h5", {
            "/posterior/a/coeffs": (nparams, it),
            "/posterior/beta/coeffs": (ns["k_beta"] * ns["rank"], it),
            "/posterior/u_sigma_inv/coeffs": (k * k, it),
            "/posterior/forecast": (h * k, it),
            "/posterior/loglik": (tt, it),
        })

        # In levels: the first horizon sits by the last level, not by zero.
        first = read(d / "vec.h5", "/posterior/forecast")[:k].mean(axis=1)
        last = ns["L"][-1]
        if np.max(np.abs(first - last)) > 0.5 * np.min(np.abs(last)):
            raise AssertionError(
                f"the first forecast horizon averages {first} against a last level of "
                f"{last}; the text says the forecast is in levels")

    def scenario_factor_model(self, d):
        ns = self.run_python(d, self.code("references/recipes.md", "A factor model", 0))
        self.check_clean(d, "dfm.h5")
        self.posterior(d, "dfm.h5")
        k, tt, h, it = (ns[v] for v in ("k", "tt", "h", "iterations"))
        n_factors = ns["n_factors"]
        expect_shapes(d / "dfm.h5", {
            "/posterior/lambda/coeffs": (k * n_factors, it),
            "/posterior/a/coeffs": (ns["n_a"], it),
            "/posterior/factors/coeffs": (n_factors * tt, it),
            "/posterior/u_sigma_inv/coeffs": (k, it),
            "/posterior/v_sigma_inv/coeffs": (n_factors, it),
            "/posterior/forecast": (h * k, it),
            "/posterior/loglik": (tt, it),
        })

    def scenario_generator(self, d):
        fence = self.code("references/recipes.md", "Generating a fixture instead", 0, "bash")
        command = fence.splitlines()[0].split()
        if command[0] != "bayests_make_model_fixture":
            raise AssertionError(f"the generator example no longer starts with the generator: {command}")
        result = subprocess.run([self.generator, *command[1:]], cwd=d,
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(
                f"{' '.join(command)} exited {result.returncode}\n{result.stdout}{result.stderr}")
        self.check_clean(d, command[1])
        self.posterior(d, command[1])

    def scenario_wrong_error_spelling(self, d):
        # The text says the wrong spelling switches nothing on and raises no
        # error. check is where it shows: accepted, and warned about.
        covar = self.code("references/recipes.md", "Adding a covariance block", 0)
        if '"gamma+covar"' not in covar:
            raise AssertionError("the covariance block example no longer spells gamma+covar")
        self.run_python(d, inside_with(self.code(*VAR, 0), covar.replace('"gamma+covar"', '"sv+covar"')))
        result = self.bayests(d, "check", "var.h5")
        if result.returncode != 0:
            raise AssertionError(f"check refused a file a run accepts\n{result.stdout}{result.stderr}")
        for expected in ('switches no covariance block on', '/priors/psi/mu is in the file'):
            if expected not in result.stdout:
                raise AssertionError(f"check did not warn '{expected}'\n{result.stdout}")

    def scenario_broken_files(self, d):
        # Files a run refuses only after the chain has finished, which check has
        # to refuse up front. Each is the complete VAR with one thing wrong.
        base = self.code(*VAR, 0)
        cases = {
            "no forecast regressors": ('del f["/data/forecast/x"]', "/data/forecast/x is missing"),
            "z against p": ('f["/model"].attrs["p"] = 2', "/data/train/z has 12 columns"),
        }
        for label, (change, message) in cases.items():
            self.run_python(d, inside_with(base, change))
            checked = self.bayests(d, "check", "var.h5")
            run = self.bayests(d, "posterior", "var.h5")
            if run.returncode != 1:
                raise AssertionError(f"{label}: expected the run to fail, it exited {run.returncode}")
            if checked.returncode != 1 or message not in checked.stderr:
                raise AssertionError(
                    f"{label}: the run fails, and check exited {checked.returncode} without "
                    f"'{message}'\n{checked.stdout}{checked.stderr}")

        # And one that only looks broken: "no forecast" written as h = 0 rather
        # than left out, which is how R and Python callers spell it. Having
        # nothing to do is not failing, so the run exits 0 and check accepts it.
        self.run_python(d, inside_with(base, 'f["/model"].attrs["h"] = 0'))
        checked = self.bayests(d, "check", "var.h5")
        run = self.bayests(d, "posterior", "var.h5")
        if run.returncode != 0:
            raise AssertionError(
                f"a written h = 0: expected the run to skip the forecast, it exited "
                f"{run.returncode}\n{run.stdout}{run.stderr}")
        if checked.returncode != 0 or "forecast: none asked for" not in checked.stdout:
            raise AssertionError(
                f"a written h = 0: the run succeeds, and check exited {checked.returncode} "
                f"without 'forecast: none asked for'\n{checked.stdout}{checked.stderr}")

    def scenario_exit_codes(self, d):
        # references/pipeline.md: a path that does not exist never started, so
        # 2; a path that exists but is not an HDF5 file is 1.
        for command in ("posterior", "coefficients", "forecasts", "loglik", "check"):
            for extra in ((), ("--all-groups",)):
                result = self.bayests(d, command, "no-such-model.h5", *extra)
                if result.returncode != 2 or "Path does not exist" not in result.stderr:
                    raise AssertionError(
                        f"{command} on a missing path {' '.join(extra)}: expected exit 2 "
                        f"saying so, got {result.returncode}\n{result.stdout}{result.stderr}")

        (d / "notes.txt").write_text("not a model\n")
        result = self.bayests(d, "posterior", "notes.txt")
        if result.returncode != 1:
            raise AssertionError(
                f"posterior on a file that is not HDF5: expected exit 1, got "
                f"{result.returncode}\n{result.stdout}{result.stderr}")

    def scenario_seed(self, d):
        # references/model-file.md and pipeline.md: /model/seed makes a model's
        # draws a function of its file, however the run is split up and whatever
        # ran before it.
        self.run_python(d, inside_with(self.code(*VAR, 0), 'f["/model"].attrs["seed"] = 20260913'))
        self.check_clean(d, "var.h5")
        checked = self.bayests(d, "check", "var.h5")
        if "seed: 20260913" not in checked.stdout:
            raise AssertionError(f"check did not report the seed\n{checked.stdout}")

        # Copies taken before anything runs, so each holds the same data.
        (d / "walk").mkdir()
        for target in ("again.h5", "staged.h5", "other.h5", "negative.h5",
                       "walk/first.h5", "walk/second.h5"):
            shutil.copy(d / "var.h5", d / target)

        self.posterior(d, "var.h5")
        reference = posterior_arrays(d / "var.h5")

        self.posterior(d, "again.h5")
        for command in ("coefficients", "loglik", "forecasts"):
            result = self.bayests(d, command, "staged.h5")
            if result.returncode != 0:
                raise AssertionError(
                    f"{command} staged.h5 exited {result.returncode}\n{result.stdout}{result.stderr}")
        # Two identical models in one walk: whichever the filesystem lists second
        # starts after the first has used the generator.
        walk = self.bayests(d, "posterior", "walk")
        if walk.returncode != 0:
            raise AssertionError(f"posterior walk exited {walk.returncode}\n{walk.stdout}{walk.stderr}")

        for path in (d / "again.h5", d / "staged.h5", d / "walk" / "first.h5", d / "walk" / "second.h5"):
            got = posterior_arrays(path)
            if got.keys() != reference.keys() or any(
                    not np.array_equal(reference[name], got[name]) for name in reference):
                raise AssertionError(f"{path.name}: the same seed gave different draws")

        with h5py.File(d / "other.h5", "a") as f:
            f["/model"].attrs["seed"] = 20260914
        self.posterior(d, "other.h5")
        if np.array_equal(reference["a/coeffs"], read(d / "other.h5", "/posterior/a/coeffs")):
            raise AssertionError("a different seed gave the same draws")

        # Refused before a chain is spent on it, by the run and the check alike.
        with h5py.File(d / "negative.h5", "a") as f:
            f["/model"].attrs["seed"] = -1
        for command in ("check", "posterior"):
            result = self.bayests(d, command, "negative.h5")
            if result.returncode != 1 or "/model/seed" not in result.stderr:
                raise AssertionError(
                    f"{command} on a negative seed: expected exit 1 naming /model/seed, got "
                    f"{result.returncode}\n{result.stdout}{result.stderr}")

    def scenario_thin(self, d):
        # references/model-file.md and results.md: /model/thin keeps one draw in
        # thin, every result still holds `iterations`, and the mcmc attributes
        # count iterations after the burn-in.
        ns = self.run_python(d, inside_with(self.code(*VAR, 0), 'f["/model"].attrs["thin"] = 3'))
        self.check_clean(d, "var.h5")
        checked = self.bayests(d, "check", "var.h5")
        if "one in 3" not in checked.stdout:
            raise AssertionError(f"check did not report the thinning\n{checked.stdout}")
        self.posterior(d, "var.h5")
        k, nparams, tt, h, it = (ns[v] for v in ("k", "nparams", "tt", "h", "iterations"))
        expect_shapes(d / "var.h5", {
            "/posterior/a/coeffs": (nparams, it),
            "/posterior/forecast": (h * k, it),
            "/posterior/loglik": (tt, it),
        })
        with h5py.File(d / "var.h5", "r") as f:
            attrs = {name: int(value) for name, value in f["/posterior/a/coeffs"].attrs.items()}
        if (attrs.get("start"), attrs.get("end"), attrs.get("thin")) != (3, 3 * it, 3):
            raise AssertionError(
                f"/posterior/a/coeffs carries {attrs}; results.md documents start 3, "
                f"end {3 * it}, thin 3")

    # -- the run -------------------------------------------------------------

    def run(self):
        scenarios = [(name[len("scenario_"):], getattr(self, name))
                     for name in sorted(vars(Checker)) if name.startswith("scenario_")]
        failed = []
        for name, scenario in scenarios:
            directory = self.work_dir / name
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)
            try:
                scenario(directory)
                print(f"ok      {name}")
            except Exception:  # every scenario runs, whatever the one before did
                failed.append(name)
                print(f"FAILED  {name}")
                print(textwrap.indent(traceback.format_exc(), "    "))

        unclaimed = sorted(key for key in self.fences
                           if key[2] == "python" and key not in self.claimed)
        for file, heading, _, index in unclaimed:
            print(f"UNRUN   {file} '## {heading}' Python example {index + 1}: "
                  "no scenario in test/check_agent_recipes.py runs it")

        print(f"\n{len(scenarios) - len(failed)} of {len(scenarios)} scenarios passed, "
              f"{len(unclaimed)} example(s) not run")
        return 1 if failed or unclaimed else 0


def main(argv):
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    bayests, generator, skill_dir, work_dir = argv[1:]
    return Checker(bayests, generator, pathlib.Path(skill_dir).resolve(),
                   pathlib.Path(work_dir).resolve()).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
