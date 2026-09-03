# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and test

The committed presets (`default`, `release`) carry no compiler or dependency
paths — those live in the gitignored, machine-specific `CMakeUserPresets.json`,
which inherits from them. Run `cmake --list-presets` to see the ones configured
here; substitute that name for `<your-preset>` below.

```bash
cmake --preset <your-preset>                              # configure
cmake --build build/bin/<your-preset>                     # build
ctest --test-dir build/bin/<your-preset> --output-on-failure
```

Each preset gets its own tree under `build/bin/<preset>/`. Configuring in the
repository root is refused with a `FATAL_ERROR`. See README §"Building from
source" for the toolchain and dependency setup.

Test names are `unit.<name>`, `fixture.<name>` and `golden.<name>`. Each
`fixture.*` writes a model file into `build/bin/<preset>/test/fixtures/` and the
`golden.*` beside it runs all three entry points over it; they are paired with
CTest `FIXTURES_SETUP`/`FIXTURES_REQUIRED`, so naming one golden test regenerates
just its input:

```bash
ctest --test-dir build/bin/<your-preset> -R golden.VarTvpGamma-covar
ctest --test-dir build/bin/<your-preset> -R unit.kalman
cmake --build build/bin/<your-preset> --target docs    # Doxygen, optional
```

All eighteen samplers are covered from a clean clone: `test/make_model_fixture.cpp`
writes a model file for every one of them, and no `*.h5` is checked in. A file
recorded from a real run — real data, a real prior — is the one thing the
generator cannot supply, so any number of them can be added as extra golden
tests, and nothing in the suite depends on one being there:

```bash
cmake --preset <your-preset> \
      -DBAYESTS_RECORDED_FIXTURES="path/to/a.h5;path/to/b.h5"
```

## Architecture

The unit of work is a **model file**: one HDF5 file holding data, priors,
starting values and a `/model/algorithm` attribute. `bayests <command> <path>`
reads it, runs the sampler that attribute names, and writes posterior draws,
forecasts and pointwise log likelihood back into the same file. Nothing else is
passed on the command line, so a run is fully described by the file it is given.

Five layers, and which one code belongs in is decided by what it may touch:

| Layer | Contents | Constraint |
| --- | --- | --- |
| `include/bayests/` | Sampler classes, `inputs.h` / `results.h` / `priors.h` / `spec.h` / `reporter.h` | The whole contract an embedded host sees; nothing under `src/` is public |
| `src/core/` | The numerics (`models/`, `algorithms/`) | Links neither HDF5 nor HighFive, prints nothing, no global state beyond the Armadillo RNG |
| `src/io/hdf5/` | Translation between the file on disk and those structs | The **only** layer that includes HighFive headers. `HighFive::File` reaches the layers above through them; it must never reach `src/core/` |
| `src/models/` | `BaseModel` front-ends the CLI drives, registered in `model_factory.cpp` | Reached through `models.h` |
| `src/reporters/` | `console_reporter.cpp`, the CLI's `Reporter` | The one place the library's progress becomes `stdout` |

`src/*.cpp` above those is the command line itself — `bayests.cpp`, the option
parsing, and one file per subcommand.

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

Eighteen registered algorithms. Six VARs × {`NormalWishart`, `NormalGamma`,
`NormalStochvol`, `TvpWishart`, `TvpGamma`, `TvpStochvol`}, the same six as VECs,
plus `VecKlgs2010`, four DFMs — `DfmNormalGamma`, `DfmNormalStochvol`,
`DfmTvpGamma` and `DfmTvpStochvol`, which is the same 2×2 of coefficients against
errors the VAR and VEC rows have, minus the Wishart column a *dynamic factor
model* cannot use — and `FavarNormalWishart`. `Normal` vs `Tvp` names the
*coefficients* — for a factor model that is the loadings and the transition, and
`Tvp` moves both. The third part names the error precision.

**The FAVAR row has the Wishart column the DFM row lacks, and that is not an
inconsistency.** A factor model's *idiosyncratic* precision is diagonal by
assumption — errors that may correlate leave the factors nothing to explain — so
no member of either row offers a choice about it, and the third part of a
`Favar*` name refers to the state innovation precision Q instead. Q is a VAR's
innovation covariance: its observed block is an ordinary one and its cross block
is the correlation between the factor innovations and the shock to the observed
variables, which is what a FAVAR is estimated to measure.

**That choice fixes the identification, and the two cannot be picked
separately.** A rotation `F -> C F` is invisible in the measurement if the
loadings absorb it. A DFM rules it out with a unit lower triangular loading block
*and* a diagonal V, which together admit only `C = I`. A FAVAR has no diagonal V
to offer, so its leading loading block is the **identity** instead — see
`VarSpec::n_favar_lambda()`, which is paired with `n_lambda()` and agrees with it
at no dimension. Taking the DFM's rule over leaves a model that runs, produces
plausible numbers, and has loadings free to wander along a ridge. Two algorithms carry the weight underneath:
`kalman_durbin_koopman_2002` (whole-path simulation smoother for time-varying
coefficients) and `stochvol_ocsn_2007` (the ten-component normal mixture).
`chan_jeliazkov_2009` draws banded state paths and serves the DFM factor path.
It has a second entry point, `chan_jeliazkov_2009_conditional`, which holds the
trailing elements of every state column at observed values rather than drawing
them — what a FAVAR's part-data state needs. The two share the assembly and the
draw; conditioning sits between them and neither half knows about it. It returns
one column per period and no trailing column to drop, the state past the end of
the sample being half data and half not.

**One function draws that path for all four DFMs**, `draw_factor_path()` in
`src/core/models/dfm_support.h` — with `draw_conditional_factor_path()` beside it
for the FAVAR, on the same contract — and every one of its four per-period arguments
may arrive as one block or as a stack of one per period. The two that describe
the transition — the coefficients and their innovation covariance — are shifted
by a period on the way into the band sampler, which indexes the transition
producing state column `t` by `t - 1`; the two that describe the measurement are
not. The prior over the first `p` states takes both transition arguments
unshifted. Getting any of that wrong estimates a different model rather than
failing, so read the comment there before touching it. `stochvol_ksc_1998` is the seven-component mixture of
Kim, Shephard and Chib beside it: the two differ only in the mixture constants
and share the draw in `stochvol_mixture.h`. No sampler selects it — it is kept as
the coarser reference the unit test checks the shared draw against, so leave it
in place and do not wire a second copy of the mixture next to it.

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
test/record_fingerprints.sh build/bin/<your-preset> test/baselines/before.txt
# make the change, rebuild
test/record_fingerprints.sh build/bin/<your-preset> test/baselines/after.txt
test/diff_fingerprints.sh test/baselines/before.txt test/baselines/after.txt
```

`diff_fingerprints.sh` names the fixtures that moved and nothing else; pass one
of those names as a third argument to see its lines. Reach for it rather than
`diff` — a plain diff of two recordings is empty when nothing moved and close to
a megabyte when a shared algorithm did, and the fixture names are what decides
where to look either way. Either script also accepts a raw `ctest -V` redirect,
reduced the same way before anything is compared.

`test/baselines/` is the place for these — gitignored, and unlike `build/` it
survives a `rm -rf build`. Date the files or name them for the change they
precede; a stale baseline still diffs cleanly enough to look meaningful, which
makes it worse than none. No fingerprint is checked in as an expected value:
they shift in the last digits with the compiler, the BLAS and the CPU. The
`fingerprints.yml` workflow runs the same base-vs-head comparison on every PR.

A fingerprint recording is not something to read in full: the suite at `-V` is
close to 900 KB (169 tests from a clean clone), and a recording of it around
90 KB. Redirect, then read the reduction — both scripts do this by design, and
neither `ctest -V` nor a `test/baselines/` file belongs on a terminal it is not
being paged through. The same goes for a green `ctest` run: `> /tmp/ctest.log
2>&1` and read `tail -3`, then grep the log if anything failed. What one failing
golden test prints is the ~7 KB of fingerprints its fixture carries.

Two traps:

- Samplers are only reproducible single-threaded. `test/CMakeLists.txt` pins
  `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` on the fixture and golden
  tests (the unit tests check exact identities and do not need it); set them by
  hand when running a binary directly.
- **A green golden test is not full proof the sampler worked.** The `BaseModel`
  front-ends catch every exception and print to stderr, so a failed subcommand
  leaves its datasets absent rather than failing the run. `bayests_golden` now
  fails a fixture whose *stage* produced nothing — no draws, no
  `/posterior/loglik`, or no `/posterior/forecast` where `h` is positive — which
  catches an input the sampler rejected outright. It cannot catch a model that
  wrote some of its datasets and not others: `absent` is the right fingerprint
  for a dataset another model owns, and only a per-model table could tell those
  apart. Read the fingerprints when adding a fixture.

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
