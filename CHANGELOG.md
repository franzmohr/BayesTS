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

* **The VEC-to-VAR transformation put the identity where A_0 belongs.**
  `vec_to_var_coefficients()` built the first level lag as `I + Pi_y + Gamma_1`.
  For a structural model it is `A_0 + Pi_y + Gamma_1`: the contemporaneous matrix
  stands to the left of `dy_t`, hence to the left of `y_t`, and the `y_{t-1}` that
  substituting `dy_t = y_t - y_{t-1}` gives back carries `A_0` into `A_1`. Every
  other block was already right — `A_0` does not enter them — and so was the
  pass-through of the contemporaneous coefficients themselves.

  *Draws change* for structural VEC models, and for nothing else: `A_0` is the
  identity whenever `spec.structural` is false, which restores the old
  expression exactly. What moves is `A_1`, by `A_0 - I`, which is strictly lower
  triangular — so the first equation was right and every other one was wrong by
  the loadings of the equations above it. The visible effect is on
  `VecNormalWishartSampler::forecast()`, which converts before it simulates, and
  on any host that asks for the level parameterisation. Nothing recorded covers
  it: no fixture is a structural VEC, and the full suite passes unchanged.

  `test/unit_vec_to_var.cpp` gained a case that pins the identity for `k = 3`
  and asserts that the identity matrix would *not* satisfy it.

* **A_0 was unpacked in Psi's order, which is not A_0's order.** Both are unit
  lower triangular and stored as their `k(k-1)/2` free elements, but Psi is packed
  row by row and `A_0` column by column — the order the surviving columns of
  `kron(-y, I_k)` are in once the diagonal and everything above it is dropped.
  One function, `fill_strict_lower_triangle()`, was reading both, and its comment
  claimed the two were the same arithmetic. They agree up to `k = 3` and diverge
  from `k = 4` on. The forecast of a structural `VarNormalGamma`,
  `VarNormalStochvol` or `VarTvpGamma` therefore inverted a transposed-in-part
  `A_0` for four variables or more. The two orders now have a function each, in
  `core/algorithms/triangular_packing.h`, which says why they differ.

  *Draws change* for structural models with `k >= 4`, in the forecast only —
  `A_0` is not read anywhere else — and by an amount that has no bound worth
  quoting, since it is a different matrix being inverted rather than the same one
  computed differently. For `k <= 3` the two orders coincide element for element
  and *draws are unchanged*: the fixtures are all `k = 3` and include a structural
  forecast for each of the three models, and all 67 tests pass with the recorded
  fingerprints.

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

* **The VEC-to-VAR transformation is now part of the public contract.**
  `vec_to_var_spec()` and `vec_to_var_coefficients()` moved from
  `src/core/algorithms/vec_to_var.h` to `include/bayests/vec_to_var.h` and, with
  the header, from namespace `bayests::core` to `bayests`. The implementation
  did not move and is unchanged, so *draws are unchanged* — the arithmetic is
  the same instructions on the same inputs; the full `ctest` suite, 67 tests
  including the golden fingerprints, passes untouched.

  The reason to promote it: a VEC and its level VAR are the same model in two
  parameterisations, and an embedded host that wants the level one for impulse
  responses, variance decompositions or a forecast in levels had no way to ask
  for it. `VecNormalWishartSampler::forecast()` already used the transformation
  internally, so the numerics were there and only the declaration was out of
  reach. bvartools' `bvecmodel_to_bvarmodel()` is the first outside caller.

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
