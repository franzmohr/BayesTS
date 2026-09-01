# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and test

Presets carry no compiler or dependency paths — those live in the gitignored
`CMakeUserPresets.json`. On this machine the configured presets are
`my-windows-default` (Debug, `BAYESTS_NATIVE_ARCH=ON`) and `my-windows-release`.

```bash
cmake --preset my-windows-default                              # configure
cmake --build build/bin/my-windows-default                     # build
ctest --test-dir build/bin/my-windows-default --output-on-failure
```

Each preset gets its own tree under `build/bin/<preset>/`. Configuring in the
repository root is refused with a `FATAL_ERROR`.

Test names are `unit.<name>`, `fixture.<name>` and `golden.<name>`. Each
`fixture.*` writes a model file into `build/bin/<preset>/test/fixtures/` and the
`golden.*` beside it runs all three entry points over it; they are paired with
CTest `FIXTURES_SETUP`/`FIXTURES_REQUIRED`, so naming one golden test regenerates
just its input:

```bash
ctest --test-dir build/bin/my-windows-default -R golden.VarTvpGamma-covar
ctest --test-dir build/bin/my-windows-default -R unit.kalman
cmake --build build/bin/my-windows-default --target docs    # Doxygen, optional
```

`VarNormalWishart` and `VecNormalWishart` cannot be generated and their tests
only appear when pointed at a recorded model file. Those `*.h5` files are
gitignored but present in this working tree:

```bash
cmake --preset my-windows-default \
      -DBAYESTS_WISHART_FIXTURE=test/VarNormalWishart-1.h5
```

## Architecture

The unit of work is a **model file**: one HDF5 file holding data, priors,
starting values and a `/model/algorithm` attribute. `bayests <command> <path>`
reads it, runs the sampler that attribute names, and writes posterior draws,
forecasts and pointwise log likelihood back into the same file. Nothing else is
passed on the command line, so a run is fully described by the file it is given.

Four layers, and which one code belongs in is decided by what it may touch:

| Layer | Contents | Constraint |
| --- | --- | --- |
| `include/bayests/` | Sampler classes, `inputs.h` / `results.h` / `priors.h` / `spec.h` | The whole contract an embedded host sees; nothing under `src/` is public |
| `src/core/` | The numerics (`models/`, `algorithms/`) | Links neither HDF5 nor HighFive, prints nothing, no global state beyond the Armadillo RNG |
| `src/io/hdf5/` | Translation between the file on disk and those structs | The **only** layer that may include HighFive |
| `src/models/` | `BaseModel` front-ends the CLI drives, registered in `model_factory.cpp` | Reached through `models.h` |

**Treat a new dependency across those lines as a design change, not a build fix.**
The layering exists so the same sampler objects serve both this command line and
an embedded host — an R package cannot let a library own the files or the
console. Progress and cancellation reach samplers through the `Reporter`
interface instead of `stdout`.

Two rules that break the host silently rather than here:

- **Include `"bayests/arma.h"`, never `<armadillo>`.** It is the single point
  Armadillo enters the project, which lets a host redirect all of it with
  `BAYESTS_ARMA_HEADER=<RcppArmadillo.h>`. RcppArmadillo must be the first thing
  every translation unit sees, or the samplers draw from Armadillo's RNG rather
  than R's and `set.seed()` stops reaching the draws.
- **The core is vendored — copied, not linked — into downstream R packages.**
  `src/core/` and `include/bayests/` are copied into `bvartools` (see the
  [PR template](.github/pull_request_template.md)), and a DFM subset into a
  second package. A core change is only half done until it is propagated.

### Conventions at the file boundary

- Draws run along the **rows on disk** (what R and `coda` expect of an `mcmc`
  object) and are transposed to one draw per column inside the samplers. Flipping
  the layout is `src/io/hdf5/`'s job.
- Variable-selection positions are stored **one-based** and converted on read.
- I/O takes a `ModelFile` (file + group), not a `HighFive::File`. Keep naming
  absolute paths (`"/data/train/y"`, `"/posterior/a/coeffs"`) and the model works
  under `--group` for free. Reaching past it via `file.file()` puts the reader
  back at the root and breaks multi-model files.
- The `error` attribute turns the covariance block on and its spelling is
  model-specific: `gamma+covar`, `sv+covar`, `wishart`.
- Exit codes are load-bearing: 1 means the run started and something failed,
  2 means it never started (unusable command line).

### Model taxonomy

Fifteen registered algorithms. Six VARs × {`NormalWishart`, `NormalGamma`,
`NormalStochvol`, `TvpWishart`, `TvpGamma`, `TvpStochvol`}, the same six as VECs,
plus `VecKlgs2010`, `DfmNormalGamma` and `DfmNormalStochvol`. `Normal` vs `Tvp`
names the *coefficients*; the third part names the error precision. Two
algorithms carry the weight underneath: `kalman_durbin_koopman_2002` (whole-path
simulation smoother for time-varying coefficients) and `stochvol_ocsn_2007` (the
ten-component normal mixture). `chan_jeliazkov_2009` draws banded state paths and
serves the DFM factor path.

Each VEC differs from the VAR beside it in one place: the first `k * rank`
regressors are `beta' w_{t-1}`, a function of the current draw rather than data,
so a Gibbs block for beta is added. All six forecast in levels by rewriting the
draws as the level VAR they imply — their `/data/forecast/z` is in the level
layout, not the differenced one `/data/train/z` uses.

Combinations the samplers **reject** rather than silently ignore: SSVS with
stochastic volatility or time-varying parameters; selection on a VEC's loadings;
`structural` with a Wishart precision or a covariance block; any selection scheme
on `VecKlgs2010`.

## Adding a model

Work bottom-up, in the order in `CONTRIBUTING.md` §"The order to add the files
in": structs in `include/bayests/` → sampler declaration → `src/core/models/` →
`src/io/hdf5/` → `src/models/` + `model_factory.cpp` → fixture in
`test/make_model_fixture.cpp`. Each step builds on its own, and step 3 is the
checkpoint that proves the model is host-embeddable.

Steps 3–5 each add a line to a **different** `CMakeLists.txt`. A forgotten one
surfaces as an undefined reference at link time, not a compile error — add the
file to its target in the same edit that creates it.

Naming follows the layer: snake_case under `src/core/` and `src/io/`, PascalCase
under `src/models/` and for type names. The model's full name appears in every
one of them (`VarNormalWishartDraws`, not `CoefficientDraws`).

## Verifying a change to a sampler

`ctest` is a **smoke test**: every sampler and code path runs, and a fixture that
throws fails. It does not catch a change in the numbers, which is what most
regressions here look like. For that, pin the RNG and diff fingerprints on the
same machine and the same build:

```bash
test/record_fingerprints.sh build/bin/my-windows-default test/baselines/before.txt
# make the change, rebuild
test/record_fingerprints.sh build/bin/my-windows-default test/baselines/after.txt
diff test/baselines/before.txt test/baselines/after.txt
```

`test/baselines/` is the place for these — gitignored, and unlike `build/` it
survives a `rm -rf build`. Date the files or name them for the change they
precede; a stale baseline still diffs cleanly enough to look meaningful, which
makes it worse than none. No fingerprint is checked in as an expected value:
they shift in the last digits with the compiler, the BLAS and the CPU. The
`fingerprints.yml` workflow runs the same base-vs-head comparison on every PR.

Two traps:

- Samplers are only reproducible single-threaded. `test/CMakeLists.txt` pins
  `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` on the fixture and golden
  tests (the unit tests check exact identities and do not need it); set them by
  hand when running a binary directly.
- **A green golden test is not proof the sampler worked.** `bayests_golden` exits
  non-zero only if a fixture throws all the way out, and the `BaseModel`
  front-ends catch every exception and print to stderr. A failure on one of the
  three subcommands leaves that dataset absent and the test green. Read the
  fingerprints and check every dataset the model should carry is there —
  `absent` next to `/posterior/forecast` in a fixture with positive `h` is a
  failure, not a configuration.

## Recording the change

`CHANGELOG.md` under *Unreleased*. Every entry touching a sampler must state one
of: *draws are unchanged* (and how that was verified), *draws change by a
rounding error* (with the worst relative difference), or *draws change* (naming
the models and configurations, and why the new numbers are right). A commit
message is not a substitute — a refactor and a posterior-moving fix travel in the
same commit more often than anyone intends. Downstream packages vendoring the
core have to tell their users the same thing.

New source files need the `SPDX-License-Identifier: BSD-3-Clause` header every
existing one carries. Assisted commits get a `Co-Authored-By` trailer.
