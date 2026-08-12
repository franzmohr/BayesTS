# Changelog

Notable changes to BayesTS, newest first. Entries are written for the people who
consume the samplers rather than for whoever wrote the commit.

**Every entry that touches a sampler says what it does to the numbers.** That is
the whole reason this file exists and the one thing a git log cannot be trusted
to tell you: a message describing a refactor is not evidence that the refactor
was behaviour-preserving, and the two arrive in the same commit often enough.
Each such entry states one of

* *Draws are unchanged* — verified, not assumed. Say how: the fingerprint
  comparison in CONTRIBUTING.md, and how much of it ran.
* *Draws change by a rounding error* — the arithmetic was reassociated but the
  posterior is the same. Give the worst relative difference seen.
* *Draws change* — the posterior moved. Name the models and the configurations,
  and say why the new numbers are the right ones.

The core is vendored by downstream packages that keep their own release notes and
have to describe the same change to their users, so an entry that omits this
forces someone to rederive it from the diff.

Dates are ISO. Versions follow the `project(VERSION)` in `CMakeLists.txt`.

## Unreleased

### Fixed

* **BVS was not selecting over the contemporaneous coefficients in the
  time-varying models.** `VarTvpGamma` and `VarTvpStochvol` hold `psi` as a path,
  `n_psi x tt`, but the candidate the likelihood ratio scored switched a position
  off with a single linear index into that matrix. Column-major, so with the
  position below `n_psi` it reached row `pos` of period 0 and left every later
  period untouched — where the `a` block in the same files correctly zeroed the
  whole row.

  The mask applied to the drawn path on the way out was never affected, so the
  reported coefficients stayed consistent with the reported indicators. The
  damage was confined to the comparison: the excluded candidate differed from the
  included one in one period out of `tt`, the ratio between them was
  correspondingly close to zero, and inclusion won essentially every time. At
  `tt = 24` all three free coefficients stayed in for all 80 draws of the
  regression fixture, against 26 of 240 once the candidate spans the sample.

  *Draws change* for `VarTvpGamma` and `VarTvpStochvol` configured with both a
  covariance block and variable selection. Nothing else moves: of the 32 golden
  fixtures, exactly the two that reach this path changed and the other 30 are
  byte-identical.

### Changed

* `ctest` now carries the shared library search path on Linux as well as Windows.
  The Windows half already prepended each `CMAKE_PREFIX_PATH` entry's `bin/` to
  `PATH`, because Windows has no RPATH; the Unix half was missing on the
  assumption that RPATH always covers it. It does not. A library reached only as
  the transitive dependency of an imported target contributes no RPATH entry, so
  the link succeeds and the test binary then fails to start with `libopenblas.so.0:
  cannot open shared object file` — which is what building against a side-by-side
  prefix such as vcpkg produces. `<prefix>/lib` and `<prefix>/debug/lib` are now
  prepended to `LD_LIBRARY_PATH` for every test. macOS is deliberately not
  covered, since `DYLD_LIBRARY_PATH` is stripped from protected processes.

  *Draws are unchanged*: this only affects how a test process finds its
  libraries. All 320 fingerprints identical.

* The HDF5 target is resolved at configure time instead of being hardcoded as
  `hdf5::hdf5-shared` in three places, so the project builds against an HDF5
  packaged by someone other than its author. Which target an HDF5 config exports
  is not part of any contract, and two things vary independently: the link type,
  since a config may install a `-shared` variant, a `-static` one or both; and the
  namespace, since some packagings export `hdf5::hdf5-static` and others a bare
  `hdf5-static`. A hand-built 2.0 installs both variants, namespaced; MSYS2's 2.2
  package is static-only and unnamespaced, and nothing but a second machine was
  ever going to reveal that.

  Six spellings are now tried, shared before static and namespaced before bare, so
  an HDF5 offering several keeps linking the one it always did. An HDF5 offering
  only a static library is fine — it just means nothing needs bundling beside the
  executable on Windows. A build against one that exports none of the six fails
  with the HDF5 version, the candidate list and every HDF5 target that *does*
  exist, instead of three identical "target was not found" errors naming no
  package.

  *Draws are unchanged*: all 320 fingerprints identical, and on an HDF5 packaged
  the way it was before, the resolution picks exactly the target that was
  hardcoded.

* `cmake_minimum_required` is 3.25, down from 4.2, and the presets declare schema
  version 6 rather than 8. Nothing in the project used a CMake 4 feature, so the
  old floor excluded every distribution that does not ship a bleeding-edge CMake
  — Debian 12, Ubuntu 24.04, RHEL 9 — for no benefit. 3.25 is what is actually
  required, and `CMakeLists.txt` now records which feature sets it so the floor
  does not drift up again by accident: `packagePresets` at 3.25, the
  `ENVIRONMENT_MODIFICATION` test property at 3.22, and
  `install(... RUNTIME_DEPENDENCY_SET)` at 3.21.

  *Draws are unchanged.* Worth checking rather than assuming: lowering the
  declared version reverts every policy introduced between 3.25 and 4.2 to OLD,
  which can change how things are compiled. A clean reconfigure reports no policy
  or deprecation warnings, and all 320 fingerprints are identical.

* The two variable selection schemes are now `core/algorithms/ssvs.{h,cpp}` and
  `core/algorithms/bvs.{h,cpp}`, one file per scheme, replacing thirteen
  near-identical copies of the sweeps spread across the six samplers — ten of BVS
  and three of SSVS, two per file distinguished only by an `a_` or `psi_` prefix.
  The per-block state that used to be a dozen loose locals is `SsvsBlock` and
  `BvsBlock`, held as `std::optional`, so "these coefficients are not selected
  over" is the empty optional rather than a bool guarding a pile of empty
  matrices. `bvs_sweep()` takes the log likelihood as a callable, which is what
  lets one sweep serve a constant-coefficient model and a time-varying one, and a
  dense precision and a sparse one.

  *Draws are unchanged.* Verified bit-identical across all 320 fingerprints of
  the 32 golden fixtures, by rebuilding the pre-refactor sources on the same
  machine and diffing.

* Repeated index arithmetic in the samplers moved into `core/models/model_support.h`:
  `fill_strict_lower_triangle()` for the packed-triangle to unit-lower-triangular
  fill (nine copies, including the `A0_inv` assembly in two `forecast()` bodies
  and a sparse `Psi`), `fill_psi_path()` for the same fill per period into a
  time-varying `Psi` (four copies), and `build_psi_regressors()` for the
  triangular design matrix of the psi block (four copies).

  *Draws are unchanged*, verified the same way.

### Added

* Golden fixtures combining a covariance block with variable selection, for the
  four models that have a psi block. `make_model_fixture` only writes
  `/model/priors/psi` — and with it psi's own `varsel` attribute — when `covar` is
  set, and every selection fixture had `covar 0`, so **the psi selection step was
  unexercised in every model.** That is the gap the fix above sat in. 48 tests
  became 64.

* GitHub Actions workflows under `.github/`. `ci.yml` builds and runs the suite on
  Ubuntu (Debug and Release) and on Windows under MSYS2 UCRT64 — the Fortran
  requirement rules MSVC out, so CI uses the same toolchain a local Windows build
  does. `fingerprints.yml` automates the before/after comparison from
  CONTRIBUTING.md: it builds the base commit and the merge result back to back on
  one runner and reports which fixtures moved, which is the only way the
  comparison is valid, since fingerprints are not portable across toolchains. It
  reports rather than fails, because a moved number is often the intent.
  `docs.yml` builds the Doxygen target, which nothing else exercises since it is
  not part of ALL. `release.yml` packages a tag, refusing one that disagrees with
  `project(VERSION)`, and drafts the release rather than publishing it.

  CI cannot check the numbers on its own. It has no recorded expectation to
  compare against — there can't be one — and `BAYESTS_WISHART_FIXTURE` is not in
  the repository, so the `w-*` tests do not run there. 48 of the 64 tests do.

* `test/record_fingerprints.sh`, which reduces `ctest -V` to just the fixture
  headers and fingerprint lines. `ctest -V` interleaves those with progress bars,
  timings and absolute paths that differ between two runs of an unchanged build,
  so a raw before/after diff is noisy enough to hide a real hunk.
