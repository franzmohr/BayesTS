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

### Added

* **`DfmTvpStochvol`**, a dynamic factor model whose loadings, factor transition
  and both error covariances all move with time -- the seventeenth registered
  algorithm, and the widest model here.

  ```
  x_t = Lambda_t f_t + u_t,                  u_t ~ N(0, U_t),
  f_t = sum_{j=1..p} A_{j,t} f_{t-j} + v_t,  v_t ~ N(0, V_t),
  ```

  with U_t = diag(exp(h^u_t)) and V_t = diag(exp(h^v_t)), and every free element
  of Lambda, every element of [A_1 .. A_p] and every element of both
  log-volatilities a random walk of its own. `DfmTvpGamma`'s coefficients over
  `DfmNormalStochvol`'s errors, and nothing new of its own: the file format is the
  union of theirs and adds no dataset either does not have. Nine Gibbs blocks
  against seven. That completes the dynamic factor row as the same 2x2 of
  coefficients against errors the VAR and VEC rows carry, minus the Wishart
  column, which a factor model has no use for -- an unrestricted idiosyncratic
  covariance competes with the factor structure for the same common variation.

  Why both halves. A model that carries one of them has to explain the other with
  what it has, and the two are easy to mistake for one another: a series whose
  loading fell looks like a series whose idiosyncratic variance rose, and a period
  of common turbulence looks like a transition that changed. Carrying both is what
  lets the data say which, and `test/unit_dfm_tvp_stochvol.cpp` puts that claim to
  a sample in which the loadings fall, the idiosyncratic volatility falls and the
  factor volatility rises at once, and checks that all three come back.

  One interaction is worth knowing about because it belongs to neither parent. The
  loading paths are drawn row by row, each weighted by its own series' volatility
  period by period, so the periods in which a series was quiet identify its
  loading path and the periods in which it was wild largely do not -- while the
  path is free to move between them. That is the drift and the reweighting acting
  on the same block, and it is the reason the two halves separate at all rather
  than trading off along a ridge.

  **The factor path draw is now one function for all four dynamic factor models**,
  `draw_factor_path()` in `src/core/models/dfm_support.h`, in place of the three
  near-copies that had grown up beside each other -- `draw_factor_path`,
  `draw_factor_path_sv` and `draw_factor_path_tvp`. Each of its four per-period
  arguments may arrive as one block or as a stack of one per period, and this
  model is the first to stack all four: a loading matrix, a measurement
  covariance, a transition and a transition covariance. The two that describe the
  *transition* go into the band sampler shifted by a period, because it indexes
  the transition producing state column t by t - 1 while these models index block
  t at period t; the two that describe the *measurement* do not, and the prior
  over the first p states takes both transition arguments unshifted. Three copies
  of that convention was the arrangement `stochvol_mixture.h` warns about at
  length, and with a fourth model it would have been four.

  **Draws are unchanged** for every model that existed before, and the merge is
  the reason that needed checking rather than asserting. The full before/after
  comparison from CONTRIBUTING.md was run over it on one build and reports
  **74 unchanged, 0 moved**; the same comparison over the new sampler on top
  reports 74 unchanged again, the only difference being its two fixtures.
  `test/unit_dfm_normal_stochvol.cpp` also prints every message its `validate()`
  produces, and that output is byte-identical across the extraction of
  `validate_dfm_stochvol_block()` from it.

  `src/core/` and `include/bayests/` both change, so this is one for the
  vendoring packages to propagate.

  Two generated fixtures join the matrix, `plain` and `nofcst`, and
  `test/unit_dfm_tvp_stochvol.cpp` covers what neither parent's test reaches: the
  factor path with all four arguments stacked, against a dense posterior built
  with every one of them indexed at period t, which is where an off-by-one in
  either shift would show and where neither could be masked by the other argument
  being constant. 159 tests from a clean clone become 164.

* **`DfmTvpGamma`**, a dynamic factor model whose loadings and factor transition
  follow random walks — the sixteenth registered algorithm.

  ```
  x_t = Lambda_t f_t + u_t,                  u_t ~ N(0, U),  U diagonal,
  f_t = sum_{j=1..p} A_{j,t} f_{t-j} + v_t,  v_t ~ N(0, V),  V diagonal,
  ```

  with every free element of Lambda and every element of [A_1 .. A_p] a random
  walk of its own. `DfmNormalGamma` with its two normal priors on the
  coefficients replaced by two state equations and nothing else changed: `Tvp`
  names the coefficients, as it does in `VarTvpGamma` against `VarNormalGamma`,
  and it names *both* coefficient blocks — a model in which only the loadings
  drifted would be a different one and is not what this is. Seven Gibbs blocks
  against five.

  What the drift is for. A loading is a series' exposure to the common factor,
  and that it held over the whole sample is the assumption a factor model makes
  most often and defends least: a series can enter or leave the common component
  without anything about the factor itself changing. A constant-loading model has
  nowhere to put that except the idiosyncratic variance, which then carries it as
  noise the series is credited with throughout, including in the periods where
  the exposure did hold. Drift in the transition is the other half — the
  persistence of the common component is what a forecast from it runs on.

  The identifying block still does not drift. Lambda's leading `n_factors` square
  block stays unit lower triangular in every period, because only the product
  `Lambda_t f_t` is identified: letting the block move would let the rotation and
  the scale of the factors wander over the sample, and a loading path would then
  describe the normalisation as much as the exposure it is read as.

  Three things in the numerics are worth knowing about:

  * The factor path needed no new algorithm. `chan_jeliazkov_2009` already took a
    measurement matrix and a transition per period, so a drifting Lambda and a
    drifting A reach it as stacks. Both need the shift `draw_factor_path_sv`
    already applies to its covariances — the band sampler indexes the transition
    that *produces* state column t by `t - 1`, while this model's block t is
    period t's own — and both are taken unshifted for the prior over the first
    p states, which are the truncated transitions rather than transitions
    producing a later column. Handing either over unshifted throughout would
    estimate a model whose coefficients lag by a period: a different model, and
    not a broken one, since nothing would fail.
  * What a drifting Lambda costs is the shortcut the band sampler takes when the
    measurement is the same in every period. `Z'U^-1 Z` is formed `tt` times
    rather than once, and with many observed series that is the dominant cost of
    the assembly. There is no version of the model that avoids it.
  * The Kronecker identity `DfmNormalGamma` leans on for its transition — the
    `n_factors` equations sharing their regressors, so the posterior precision
    collapses to `kron(X X', V^-1)` and no `(tt n_factors) x (n_factors^2 p)`
    matrix is built — is a statement about a single coefficient vector and does
    not survive the coefficients becoming a path. The transition path is drawn as
    one state of `n_factors^2 p` elements against the SUR design
    `kron(x_t', I)`, scattered per period. The loading paths are drawn row by
    row, which is not an economy but the shape of the problem: row i has
    `min(i, n_factors)` free elements against different regressors, and given the
    factors and a diagonal U the rows are conditionally independent.

  **Draws are unchanged** for every model that existed before. The new sampler
  and its file format are additive, and the one thing it touches that was already
  there is `initial_state_covariance()` in `src/core/models/dfm_support.h`, which
  gains a shape dispatch so that a stack of one transition per period can be
  passed for the first p states — the same arrangement it already had for
  `v_sigma`, and the same one `chan_jeliazkov_2009` uses. For every existing
  caller the stride is zero and `a_mat.submat(0, c0, n - 1, c1)` is the
  `a_mat.cols(c0, c1)` it replaces, element for element. Verified rather than
  argued: the full before/after comparison from CONTRIBUTING.md was run on one
  build with only that hunk reverted, and reports **72 unchanged, 0 moved**, the
  only difference being the two new fixtures.

  `src/core/` and `include/bayests/` both change, so this is one for the
  vendoring packages to propagate.

  Two generated fixtures join the matrix, `plain` and `nofcst` — a DFM takes no
  variable selection, no covariance block and no contemporaneous coefficients, so
  there are no other rows for one — and `test/unit_dfm_tvp_gamma.cpp` covers what
  the state equations changed: the stacking conventions as exact identities, the
  factor path against a dense posterior built with a loading matrix and a
  transition per period (which is what pins the shift), and recovery of a loading
  that really moves and a transition that really decays. 154 tests from a clean
  clone become 159. `/posterior/lambda/sigma` joins the fingerprint list in
  `test/golden_models.cpp`, so every recording gains one line per fixture.

* **`bayests_golden` fails a run that produced nothing, instead of printing
  `absent` fourteen times and passing.** The `BaseModel` front-ends catch every
  exception and print it to stderr, so until now a model file the sampler
  rejected outright ran all three entry points, wrote no dataset at all and left
  the test green. The harness now checks each stage against its own output and
  exits 1 if any of them came back empty: no posterior draws, no
  `/posterior/loglik`, or no `/posterior/forecast` in a file whose `h` is
  positive — the last being the case `CONTRIBUTING.md` has called a failure
  rather than a configuration for as long as it has said anything about it.

  **Draws are unchanged**: `test/golden_models.cpp` is the harness, not a
  sampler, and the change is confined to what it does after the run. The whole
  suite passes as before, which is itself the check that the three conditions
  do not fire on a legitimate fixture — including the fifteen `nofcst` rows,
  which have no `h` and so are not asked for a forecast.

  This is a floor, not a guarantee. A model that writes some of its datasets and
  not others still passes: `absent` is the correct fingerprint for a dataset
  belonging to another model, and only a per-model table of expected outputs
  could distinguish the two. The fingerprints are still worth reading when a
  fixture is added.

* **All fifteen samplers now have a generated fixture, so the suite depends on
  no data outside the repository.** `VarNormalWishart` and `VecNormalWishart`
  were the two exceptions: their tests existed only when
  `BAYESTS_WISHART_FIXTURE` or `BAYESTS_VEC_FIXTURE` pointed at a recorded model
  file, and since `*.h5` is gitignored and a recording runs to hundreds of
  megabytes, that meant they never ran in CI — including in `fingerprints.yml`,
  which is what would have caught a shared refactor moving their posteriors.

  **Draws are unchanged.** The change is confined to `test/`; no sampler, no
  header and no I/O code was touched. The full before/after comparison from
  CONTRIBUTING.md was run on one build: **64 unchanged, 0 moved**, with the only
  other differences the four fixtures derived from recordings that no longer
  exist and the eight generated ones that replace them.

  The exclusion turned out to be historical rather than technical.
  `make_model_fixture` already wrote every dataset either reader asks for:
  `VarNormalWishart` is the coefficient block of `write_var_normal_gamma` beside
  the error block of `write_var_tvp_wishart`, and `VecNormalWishart` reads the
  same file `VecKlgs2010` does — every VEC fixture carries both the compact and
  the SUR regressors, so only `/model/algorithm` separated them. Eight rows join
  the generation matrix: `plain`, `ssvs`, `bvs` and `nofcst` for each. No
  `covar` row, because neither model has a psi block, and no `structural` row,
  because `require_identified_structural()` refuses A_0 against an unrestricted
  Sigma. 138 tests from a clean clone become 154.

  What a generated file cannot be is a real sample under a real prior. The two
  model-specific build options are replaced by one that is not tied to a model:

  ```bash
  cmake --preset <preset> -DBAYESTS_RECORDED_FIXTURES="a.h5;b.h5"
  ```

  Each file gets a `golden.recorded-<name>` test dispatched on its own
  `/model/algorithm`, and nothing in the suite depends on one being supplied.
  `test/make_varsel_fixture.cpp` is removed with them: it existed to bolt a
  selection block and forecast regressors onto a recording so
  `VarNormalWishart`'s varsel and forecast branches could be reached at all, and
  the generated rows now reach them directly.

  Nothing under `src/core/` or `include/bayests/` changed, so there is nothing
  here for the vendoring packages to propagate.

* **`DfmNormalStochvol`**, a dynamic factor model with stochastic volatility in
  both error terms — the fifteenth registered algorithm.

  ```
  x_t = Lambda f_t + u_t,                u_t ~ N(0, U_t),  U_t = diag(exp(h^u_t)),
  f_t = sum_{j=1..p} A_j f_{t-j} + v_t,  v_t ~ N(0, V_t),  V_t = diag(exp(h^v_t)),
  ```

  with every element of both log-volatilities a random walk of its own.
  `DfmNormalGamma` with its two gamma priors replaced by two stochastic
  volatility blocks and nothing else changed: `Normal` still says the loadings and
  the factor transition are constant, as it does in `VarNormalStochvol` against
  `VarTvpStochvol`. Seven Gibbs blocks against five. The volatility draw is the
  ten-component mixture of Omori et al. (2007), through the same
  `stochvol_ocsn_2007` the VEC samplers call.

  Both placements are there because neither substitutes for the other.
  Idiosyncratic volatility reweights the series that identify the factors, which
  is what a sample spanning a change in volatility needs and what keeps a single
  wild observation from being dragged into the factor. Factor-innovation
  volatility is the common component's own, and it is what stops the `k`
  idiosyncratic variances from jointly absorbing a shock every series felt at
  once — the factor is otherwise flattest exactly when it should move most.

  Three things in the numerics are worth knowing about:

  * The factor path needed no new algorithm. `chan_jeliazkov_2009` already took a
    covariance per period in both equations. What it does need is the period
    indexing to be right, and it is off by one from this model's: it indexes the
    transition that produces state column `t` by `t - 1`, so the stack handed to
    it is this model's shifted up by a period. `draw_factor_path_sv` in
    `dfm_support.h` is the only place that shift lives, and it is the one thing
    here a plausible mistake would leave *running* — a model whose volatility lags
    its own innovations by a period fails nothing.
  * The prior over the first `p` factors now uses those `p` periods' own
    covariances rather than one repeated. `initial_state_covariance` takes either
    shape, dispatching on the height, so `DfmNormalGamma` is unaffected.
  * The transition loses `DfmNormalGamma`'s Kronecker collapse:
    `sum_t kron(x_t x_t', V_t^-1)` does not factor into `kron(X X', V^-1)`. What
    survives is that `V_t` is diagonal, so the `n_factors` equations are
    conditionally independent and each contributes its own weighted
    cross-product — `n_factors` products of size (`n_factors p`)²`tt` instead of
    `tt` Kronecker products, and nothing allocated per period. Still no
    `(tt n_factors) x (n_factors² p)` matrix.

  *There is nothing to compare the draws against* — no second implementation and
  no closed form — so `test/unit_dfm_normal_stochvol.cpp` covers it the way
  `unit_dfm_normal_gamma.cpp` covers its neighbour, and only for what the
  volatility changed. Forty-seven checks in four groups: the per-period
  conventions exactly, including the transition moments against the Kronecker sum
  they replace (agreement to 1e-11) and a variance made large in one period only,
  which has to move that period and no other; the factor path against a dense
  posterior built with a covariance per period, at `p` = 0, 1 and 2 (the mean and
  covariance of 30 000 draws within 0.005 of it, where the alignment is what the
  comparison pins, since the dense construction has no room for an off-by-one);
  recovery from a sample simulated with one variance falling and the other rising
  over 800 periods, which the chain has to find both the level and the direction
  of; and what `validate()` refuses, the swapped `k`/`n_factors` widths among it.
  Two fixtures run the whole thing through the file layer and the command line.

  *Draws are unchanged* for every existing model. Three shared files were touched
  and none of them alters an existing path: `precision_of` in
  `chan_jeliazkov_2009.cpp` gained a diagonal branch that returns the same
  numbers a Cholesky inverse of a diagonal matrix does; `initial_state_covariance`
  gained a second accepted shape, with a test that the one-covariance spelling is
  bit-identical to the stack of copies; and `DfmNormalGammaInput::validate()`
  moved its shared checks into `validate_dfm_shape()` unchanged. The full suite
  (145 tests, up from 140, including the recorded `VarNormalWishart` and
  `VecNormalWishart` fixtures) passes.

  One performance note that is not specific to this model. `precision_of` is
  called once per *period* when a covariance moves with time, and it was a dense
  `inv_sympd` — at 100 series over 300 periods, 300 O(k³) factorisations of a
  matrix that is zero off the diagonal, some 3e8 flops a draw. It now scans for
  diagonality in O(k²) and divides. Every time-varying parameter model here
  reaches it with a diagonal state covariance, so they get the same saving.

* **`--group <path>`** on all four commands: the group a model's tree hangs
  under inside its HDF5 file. Without it every path is read from the root of the
  file exactly as before, so nothing that already works has to change; with it
  one file can hold several models side by side —

  ```bash
  bayests posterior models.h5 --group /models/3
  ```

  — and a directory walk looks for the same group in every file it visits. A
  `--group` that cannot name a group exits 2 before a file is opened; one that
  names nothing in the file exits 1, saying which path was missing.

  What made this more than a new flag is that every path the io layer names is
  absolute, and in HDF5 a leading slash resolves from the root of the file even
  through a group handle — so handing the readers a `HighFive::Group` would have
  silently gone on reading the root. The group is therefore put on the front of
  the path by a new `ModelFile` handle (file plus group) that the io layer takes
  in place of a `HighFive::File`. It converts implicitly from one, which is why
  no reader, writer or fixture that has no group to name changed at all.

  *Draws are unchanged*, and not only by inspection: `test/CMakeLists.txt` now
  writes the `VarNormalGamma-plain` fixture a second time under `/models/3` and
  runs it through the golden harness with `--group`. The two print identical
  fingerprints, digit for digit, over all thirteen posterior datasets — same
  sampler, same seed, the same numbers in a different place in the file. The full
  suite (140 tests, including the recorded `VarNormalWishart` and
  `VecNormalWishart` fixtures) passes unchanged. `test/unit_model_group.cpp`
  covers the other end: that a model written under a group is at that group and
  *not* at the root, which a round-trip test alone could not tell apart from a
  prefix that was dropped.

* **`DfmNormalGamma`**, a dynamic factor model — the first model here that is not
  a regression, and the fourteenth registered algorithm.

  ```
  x_t = Lambda f_t + u_t,                u_t ~ N(0, U),  U diagonal,
  f_t = sum_{j=1..p} A_j f_{t-j} + v_t,  v_t ~ N(0, V),  V diagonal,
  ```

  for `k` observed series and `n_factors` unobserved ones, with a normal prior on
  the free loadings and on the transition and independent gamma priors on both
  precisions. Five Gibbs blocks: the factor path, the loadings, the two
  precisions, the transition. After Chan, Koop, Poirier and Tobias (2019); the
  reference implementation is bvartools' `dfmpost()`.

  *There is nothing to compare the draws against* — no second implementation
  here, and no closed form. What `test/unit_dfm_normal_gamma.cpp` does instead is
  check the model from three sides. The conventions are pinned exactly: which
  elements of `Lambda` are free, where the transition's lag blocks go, what the
  residual is, and what the prior over the first `p` factors comes to, each
  against a hand-derived expectation. The factor block is checked against its own
  definition: the test builds the `tt·n_factors` square precision the reference
  implementation builds, inverts it, and compares the mean and covariance with
  30 000 draws from the banded sampler, at transition orders zero, one and two —
  agreement to 0.003 on both, against posterior standard deviations of order 0.5.
  And the whole chain is run on a simulated sample of 800 periods, where it
  recovers the loadings to 0.08, the transition to 0.03 and both precisions to
  16%.

  Three things follow from the factors being unobserved. There is no
  `/data/train/z`: `/data/train/y` is all the data, and the forecast — which runs
  the transition on from the last drawn factors — needs no out-of-sample matrix
  at all, only `/model/h`. A whole factor path is part of every draw and is
  written to `/posterior/factors/coeffs`. And the reported pointwise log
  likelihood is the *conditional* one, `p(x_t | f_t, Lambda, U)`, evaluated at the
  stored path; the marginal would need a Kalman filter per draw and is a
  different quantity, which matters for what an information criterion computed
  from it means.

  The path is drawn whole by `chan_jeliazkov_2009`, whose band this posterior
  fits: O(`tt` `n_factors`³) against the O(`tt`³ `n_factors`³) of forming that
  precision and factorising it, which is what `dfmpost()` does. Factors before
  the sample are zero rather than drawn — bvartools' convention — and the
  covariance that implies over the first `p` of them is what is handed over as the
  band sampler's prior, so its prior-plus-transitions decomposition reproduces
  the model exactly rather than approximately.

  **Three defects in the reference implementation are not reproduced.** Between
  them they mean `dfmpost()` is correct only at one factor and a transition of
  order at most two, so there is no configuration where this sampler could have
  been made to agree with it and be right:

  - `.post_lambda` draws with `solve(chol(K, "lower"), z)`, whose covariance is
    `(L'L)^-1` rather than the `K^-1` intended. The two coincide only when the
    block is 1x1, so the loadings are drawn from the wrong distribution as soon
    as there is more than one factor. This uses `draw_normal_precision()`, which
    factorises once and solves against the upper factor.
  - `dfmpost()` builds the transition's regressor matrix with
    `x_a[(i - 1) + 1:n, ]` where the block for lag `i` occupies rows
    `(i - 1) * n + 1:n`. For `n > 1` the lag blocks are laid on top of one
    another.
  - `generate_lower_block_diagonal()` writes past the end of the matrix for
    `p >= 3`, and drops a coefficient block from the last columns for `p >= 3`.
    This builds the equivalent structure itself and is exercised at `p = 2` by
    the fixtures and at `p = 0, 1, 2` by the unit test.

  Two smaller divergences are deliberate rather than corrective. The free
  loadings are ordered row by row wherever they appear as a vector — the starting
  value and both halves of the prior — which is the order the equation-by-equation
  draw consumes them in; `dfmpost()` stores them column-major (`lower.tri`) but
  slices the prior precision row-major, a mismatch invisible only because that
  prior is a scalar diagonal. And the posterior stores `Lambda` whole, as vec of
  the `k` x `n_factors` matrix with the identifying ones and zeros in place,
  rather than the free elements alone.

* **`VecKlgs2010`**, the cointegration sampler of Koop, León-González and
  Strachan (2010) written against the compact regressors instead of the SUR
  system. The thirteenth registered algorithm, and the first that is not a model
  of its own: it draws exactly the posterior `VecNormalWishart` draws.

  What changes is the coefficient block. A VEC's k equations share their
  regressors, so its SUR design matrix is `z = kron(W_x, I_k)` and the posterior
  precision factors — `z' kron(I_tt, Sigma^-1) z = kron(W_x' W_x, Sigma^-1)`,
  with the right-hand side collapsing to `vec(Sigma^-1 Y' W_x)` the same way.
  Forming those directly leaves a Gram product that is `n_x` square over tt
  periods instead of `k n_x` square over `tt k` rows — O(tt n_x²) against
  O(tt k³ n_x²) — and builds no `(tt k) x (k n_x)` matrix at all. The beta block
  and Sigma's are unchanged; beta's regressors are `kron(alpha, w_t')`, which
  varies with t and has no such structure to exploit. Measured through the
  bvartools binding on a three-variable VEC of level order four, rank one, 160
  periods and 1000 draws: 0.05 s against 0.41 s. The gap widens with k.

  *Draws are unchanged.* Not by fingerprint comparison but by construction, and
  checked as such: `test/unit_vec_klgs_2010.cpp` seeds Armadillo's generator
  once, draws one iteration from each sampler on the same sample and compares
  them element by element, then repeats it for a chain of 80 after 40 burn-in
  and compares the posterior means and the pointwise log likelihoods. Both
  chains consume the RNG in the same order and the same amounts, so the only
  difference available to them is the last bits of a differently associated
  matrix product; the observed one is below 1e-9 on every element.

  Reading the input is where a caller sees the difference. The regressors arrive
  at `/data/train/x`, `tt` rows by one column per regressor, rather than at
  `/data/train/z`; `/data/train/w` and `/data/forecast/z` are unchanged, the
  latter still in the level VAR layout every VEC forecast expects. Variable
  selection is not implemented and is refused rather than ignored — SSVS and BVS
  both act on the columns of the matrix this sampler declines to build, and
  `validate()` says so and points at `VecNormalWishart`.

  Two details are taken from `VecNormalWishart` rather than from bvartools'
  `.simulation_klgs2010`, which is the R implementation this follows otherwise.
  The Wishart prior scale is added to Sigma's posterior scale instead of being
  overwritten by the cointegration term, and the rank is counted in the degrees
  of freedom that go with it; the alternative leaves `/priors/u_sigma/scale`
  dead data that a caller has every reason to think is being used. The prior mean
  of `a` enters as `V^-1 mu` rather than as `mu`. Both matter only for a prior
  that is not the default flat one, and both are what makes the two samplers here
  the same sampler.

* **`chan_jeliazkov_2009`**, the precision based alternative to
  `kalman_durbin_koopman_2002` for the same conditional posterior, after Chan and
  Jeliazkov (2009). Rather than filtering forward and sampling backward, the
  whole path is one Gaussian vector whose precision is block tridiagonal — the
  state equation is first order Markov and each period's measurement touches one
  state — and it is drawn in a single pass over a block banded Cholesky. Same
  argument shapes as the smoother, including the constant and per-period forms of
  `sigma_u`, `sigma_v` and `B`, and the same `M x (T+1)` return — column `i` is
  the state period `i`'s observation loads on, and the last column is the
  transition applied once past the end of the sample — so the two are
  interchangeable and can be put through the same inputs.

  *No draws change:* nothing calls it. The samplers still use the smoother, and
  the measurements below are why.

  **The transition may be of any order p.** `B` carries the p coefficient
  matrices side by side — M x pM for a transition that holds throughout, MT x pM
  for one per period — and p is read off the width, so an M x M argument is the
  first order case and nothing about the existing interface moves. `a_init` and
  `P_init` then cover the first p states jointly, pM and pM x pM, which for p = 1
  is what they already were. The precision is block banded of bandwidth p, and
  the same sweep factorises it whatever p is.
  
  This is where the precision formulation is *better* than the smoother rather
  than merely different. An order p state equation reaches the simulation
  smoother only in companion form, which inflates the state to pM and makes
  `Sigma_v` singular — workable there, since it only ever takes a square root of
  it, but not here, and not something a caller should have to construct. The
  precision route takes `H = I - A_1 L - ... - A_p L^p` directly, at bandwidth p,
  with `Sigma_v` the nonsingular M x M innovation covariance it actually is. A
  dynamic factor model with a VAR on the factors is the case that wants this.

  Cost at p > 1 grows with the band, not with the state: an order 4 transition on
  three factors over 200 periods with 100 observed series costs 0.56 ms against
  0.43 ms at order 1. The p = 1 timings below are unchanged by the
  generalisation — measured before and after with the same harness, 5.54 ms
  against 6.10 ms at T = 200, M = 45, which is within the run to run spread.

  **It is about twice as slow as the smoother on the shapes this library runs.**
  Measured against it in the same binary, constant covariances, `K = 3`, with the
  BLAS pinned to one thread — the same pinning `test/CMakeLists.txt` applies to
  the golden harness, and the regime an embedded host with a reference BLAS is in:

  | T | M | `chan_jeliazkov_2009` | `kalman_durbin_koopman_2002` | ratio |
  | --- | --- | --- | --- | --- |
  | 89 | 21 | 0.79 ms | 0.43 ms | 0.54x |
  | 200 | 21 | 1.86 ms | 0.88 ms | 0.47x |
  | 500 | 21 | 4.13 ms | 2.19 ms | 0.53x |
  | 89 | 45 | 2.59 ms | 1.35 ms | 0.52x |
  | 200 | 45 | 6.10 ms | 2.95 ms | 0.48x |
  | 500 | 45 | 17.05 ms | 7.52 ms | 0.44x |

  The ratio is flat in both T and M, which is the useful part: the two are the
  same order and differ by a constant. Both are O(T M^3) and neither ever forms a
  `TM x TM` matrix, so there was no asymptotic advantage to win here — the
  expectation going in was that there was, and that was simply wrong. Per period
  this one does a Cholesky, a triangular solve with M right hand sides and one
  symmetric product; the smoother's inner loop is two `gemm` calls. Roughly twice
  the arithmetic, and `gemm` is the better optimised kernel of the two.

  **A threaded BLAS makes it worse, not better, and badly so.** The same
  measurement with OpenBLAS left to use every core puts M = 45, T = 200 at 18.7 ms
  instead of 6.1 ms — a factor of three lost — while the smoother is unchanged at
  2.9 ms. Per period this issues several small BLAS calls where the smoother
  issues two larger ones, and thread synchronisation on a 45 x 45 operation costs
  more than it saves. Anyone benchmarking these two against each other should pin
  the threads first, or the answer is about the BLAS rather than about the
  algorithms.

  Two things were tried and are in the code: `solve_opts::fast` on the triangular
  solves, worth about 20%, and skipping `B'Sigma_v^-1 B` when `B` is the identity,
  which every time varying parameter model here uses. Three alternatives to the
  M x M triangular solve were measured and all were slower, `inv(trimatl) * U`
  by 2.5 times.

  **Exploiting the structure inside the blocks was tried, and made it slower.**
  With `B` the identity the off-diagonal block is `-Sigma_v^-1`, which is diagonal
  in `VarTvpWishart` and `VarTvpGamma`, and the factor `S` is then exactly lower
  triangular — verified, zero above the diagonal to the last bit. That licenses a
  spelling in which `S` is never formed at all, since
  `S_i'S_i = Lam_i D_i^-1 Lam_i` and the two other places `S` appears are products
  with a vector: a symmetric inverse and some `M^2` scaling in place of a
  triangular solve with M right hand sides and a full `M x M x M` product. Fewer
  flops on paper, 15 to 30% slower at every size measured, because `inv_sympd`
  factorises the block a second time and the inverse is a poorer LAPACK kernel
  than the `gemm` it displaced. Measured, reverted, and the reasoning left in a
  comment at the loop so the next reader does not spend the afternoon on it. The
  lesson generalises: what matters at these block sizes is which kernel the
  largest term lands in, not how many flops it is.

  **And there is no third thing to try.** `Z_t'Sigma_u^-1 Z_t` has rank K, which
  is 3 against an M of 21 or 45, so the first diagonal block is a rank 3 update to
  a diagonal — but that is where it ends. `R_0` is nonsingular and `Lam_0` is a
  nonsingular diagonal, so `S_0 = R_0^-T Lam_0` is nonsingular and `S_0'S_0` is
  positive definite of *full* rank M. Subtracting it leaves `D_1` dense and of no
  special form, and so is every block after it. The low rank measurement structure
  is a one-period saving, not a per-period one, and the Schur complement destroys
  it whatever the model does.

  So for a random walk over diagonal state variances with no missing observations
  — which is every model in this library — the cost is irreducibly O(T M^3) on
  dense M x M blocks, and what is left of the gap to the smoother is which BLAS
  kernel the largest term lands in rather than any structure still on the table.
  A general sparse Cholesky is the wrong direction for the same reason: measured
  through CHOLMOD on this exact matrix at T = 200, M = 45, the factorisation alone
  takes 13.9 ms against this routine's 6.1 ms for the whole draw, and a
  fill-reducing ordering makes it 17.7 ms, since a band has no fill to reduce.
  Sparse libraries earn their keep when the pattern is irregular enough that it
  cannot be hard-coded; a fixed band is the case where they have least to add.

  **The picture inverts when the measurement is the large dimension.** A dynamic
  factor model has a small state and many observed series, which is the mirror
  image of a time varying parameter VAR, and the smoother's per-period
  `inv(Z_t P Z_t' + Sigma_u)` is then an N x N inverse. Same harness, same
  pinning, diagonal idiosyncratic errors, T = 200:

  | N | r | `chan_jeliazkov_2009` | `kalman_durbin_koopman_2002` | ratio |
  | --- | --- | --- | --- | --- |
  | 20 | 3 | 0.24 ms | 1.66 ms | 6.9x |
  | 50 | 3 | 0.28 ms | 6.83 ms | 24x |
  | 100 | 3 | 0.53 ms | 231.5 ms | 439x |
  | 200 | 3 | 1.24 ms | 681.0 ms | 549x |

  Two honest qualifications. The smoother is being asked to do something a
  factor model implementation would not ask of it — collapsing the N-dimensional
  observation to an r-dimensional sufficient statistic first, which a diagonal
  `Sigma_e` permits, recovers most of that. And this routine still forms
  `Z_t'Sigma_u^-1` against the full N x N matrix rather than exploiting the
  diagonal, and recomputes `Lambda'Sigma_e^-1 Lambda` every period although a
  factor model's loadings do not vary with t; hoisting that is worth roughly
  another order of magnitude at N = 200 and is the obvious thing to do if this is
  ever put on a factor model's hot path.

  So this is here as a second, independent implementation to validate the first
  against, and as the algorithm of choice when the state is small, the
  measurement is wide, or the transition is of order greater than one, rather
  than as a replacement. Where the approach does win in this
  library is already in use and at the other extreme of the same trade:
  `stochvol_mixture.h` draws a scalar state, where the band is tridiagonal, the
  blocks are numbers, and there is no dense block arithmetic to lose on.

  One capability difference: `P_init` has to be invertible here, where the
  smoother also takes a singular one. A precision based sampler needs
  `P_init^-1`, and a state fixed at `a_init` rather than tightly distributed
  around it is a model one state block shorter. It throws rather than pretending,
  and the note on the function says so.

* **`test/unit_chan_jeliazkov.cpp`**, registered as `unit.chan_jeliazkov`. The
  load bearing check is agreement with `kalman_durbin_koopman_2002`: the same
  inputs through both, 20,000 draws each, and the sample means have to match to
  sampling error — they agree to 1.4% of a posterior standard deviation, against
  a bound of four standard errors. That is what says this is the same
  distribution and not merely a plausible one. Around it, three exact statements
  about which period each block belongs to: a constant argument agreeing with its
  replication, a state variance negligible in all but one period producing one
  jump in that period, and a precise measurement with identity regressors pinning
  each state to its own observation. The last two pin the transition and the
  measurement blocks separately, which the first cannot — a uniform argument has
  no period to be wrong about. `sigma_v` cannot be set to exactly zero as
  `unit_kalman.cpp` does, since this sampler inverts it, so that identity is
  stated with a tolerance instead.

  Two more cover the order p band. The sharper one is that an order 2 transition
  with a zero second lag, given the joint prior on `(s_0, s_1)` that the first
  order model implies for its own first two states, *is* the first order model —
  same precision, same factor, same right hand side, same random numbers in the
  same order — so it has to give the first order draw. It does, to 7e-16
  relative, which puts the whole of the banded machinery against a path already
  validated against the smoother. The other pins what agreement cannot: with the
  measurement uninformative and both variances negligible the posterior is the
  prior, and the prior is a deterministic recursion the test computes for itself,
  so `A_1` being the *first* block of columns of `B`, `a_init` running forward in
  time, and the transition indexed `t - 1` producing column `t` are all
  falsifiable.

  Verified against mutations: dropping `B'Sigma_v^-1 B`, shifting the measurement
  block by one period, reading the lag blocks in reverse order, indexing the
  transition by `t` instead of `t - 1`, and dropping one term from the band
  accumulation are all caught.

* **`test/unit_kalman.cpp`**, covering the simulation smoother without standing
  up a sampler. Three identities do the work. A constant argument and a stack of
  `T` copies of it describe the same model, so they must give the same draw from
  the same seed — which is what says the constant and time varying paths through
  the function are the same path. With `sigma_v` zero in every period but one,
  the state cannot move except at that period, so the drawn path is piecewise
  constant with its single jump in exactly the right place — an exact statement
  about which period a block governs, which the agreement test cannot make, since
  a uniform argument has no period to be wrong about. And with identity
  regressors and a measurement variance next to nothing the state has no freedom
  left, so column `i` of the result has to be `y_i` — which is what fixes where
  the `T + 1` returned columns sit against the `T` observations, the thing every
  caller had wrong (see Fixed). Registered as `unit.kalman`. All three were
  checked against deliberate mutations: a stride forced to zero, so time
  variation is silently ignored, fails the second while passing the first; a
  block off by one period fails the second; and the third fails on the
  `.cols(1, T)` reading, which is asserted explicitly rather than left implied,
  so a path flat enough for either alignment to fit cannot satisfy it.

* **`VecNormalGamma`, `VecNormalStochvol`, `VecTvpWishart` and `VecTvpGamma`.**
  With `VecTvpStochvol` below and `VecNormalWishart` already there, the VEC side
  now mirrors the VAR side exactly: twelve registered algorithms, the same six
  error and coefficient specifications with and without a cointegration
  relation. Nothing new is invented — each is the VAR sampler of the same name
  with the VEC's two coefficient blocks in front of it, so what to read is the
  VAR for the error block and `VecNormalWishart` or `VecTvpStochvol` for the
  rest.

  *Draws are unchanged* for everything that already existed. Verified: the
  golden output was recorded from a build of the sources as they stood before
  any of this VEC work, on the same machine, and once the CTest indices are
  stripped the diff against the run after is a pure insertion. All 396
  pre-existing fingerprints are byte-identical, 288 lines were added, and
  `ctest` is green at 116 tests.

  Three things are worth knowing about the numbers the new models produce.

  - **What the cointegration space prior conditions on.** It puts
    `alpha | G ~ N(0, v^-1 (beta' P_tau^-1 beta)^-1 kron G)`, so it needs an
    error precision. With a constant one that is simply the precision;
    `VecNormalStochvol` has a different one in every period and takes their
    average over the sample, which is the `g_i` of bvartools' `.bvecalg`. The
    average appears only where the prior does — beta's own posterior uses the
    per-period precisions in full.
  - **What it contributes back.** Because the prior conditions alpha on Sigma,
    Sigma's posterior owes it a term: `VecNormalWishart` adds
    `v^-1 alpha (beta' P_tau^-1 beta) alpha'` to its scale and `rank` to its
    degrees of freedom. Independent gammas and a stochastic volatility path have
    no conjugate update for that, and `.bvecalg` attempts none, so
    `VecNormalGamma` and `VecNormalStochvol` add nothing. The two time-varying
    VECs add nothing either, for a different and stronger reason: their loadings
    are a random walk whose innovation variance is drawn from a gamma of its own
    and never sees Sigma at all.
  - **Where beta's posterior cannot take the shortcut.** `VecNormalWishart`
    contracts the data term to `kron(Alpha' S Alpha, sum_t w_t w_t')`, which
    needs one `S` for the whole sample. `VecNormalStochvol` has tt of them, so it
    builds the regressors out in full and contracts against the block diagonal,
    as `.bvecalg` does.

  Shared rather than copied five times: `src/core/models/vec_support.h` now holds
  the four Kronecker forms of `alpha_t beta_t' w_t` and the alpha
  reparameterisation, and `src/core/inputs.cpp` grew a set of validation helpers
  — the time-varying state block, the VEC column count, the loadings-are-not-
  selectable rule, the stochastic volatility block, the Wishart block. The three
  existing time-varying VAR validators were collapsed onto the first of those.
  Every message is byte-identical to what it was, which the fingerprint
  comparison above covers along with everything else.

  `test/make_model_fixture.cpp` writes all five generatable VECs from one set of
  data builders, and the suite gained 19 fixtures: the plain, BVS, covariance,
  both and no-forecast combinations of each, plus SSVS for `VecNormalGamma` and
  no covariance row for `VecTvpWishart`, which has no psi block. Structural rows
  came with the forecast fix below.

* **`VecTvpStochvol`, a VEC with time-varying parameters and stochastic
  volatility.** The eighth registered algorithm, and the port of bvartools'
  `.bvectvpalg` for its `sv` and `sv+covar` error specifications: every
  coefficient follows a random walk, the cointegration vectors among them, and
  the errors carry stochastic volatility with an optional time-varying
  covariance block. BVS is available for the coefficients and for that block;
  SSVS is not, as in every other time-varying model.

  *Draws are unchanged* elsewhere. Verified rather than assumed: the golden
  output was recorded from a build of the pre-change sources on the same machine
  and diffed against the same run after, and once the CTest indices are stripped
  — every test after the new ones renumbered — the diff is a pure insertion of
  the five `VecTvpStochvol-*` blocks. All 396 pre-existing fingerprints are
  byte-identical. `ctest` is green, 78 tests.

  What is new numerically is one Gibbs block. A VEC's first `k * rank`
  regressors are `beta' w_{t-1}`, so the sampler alternates between two state
  paths conditioned on each other: `a` given the regressors beta implies, then
  beta given the loadings `a` carries, each drawn as a block with the Durbin and
  Koopman (2002) smoother. Three things differ from the R implementation, and
  each is documented where it happens:

  - The coefficient blocks are drawn with the smoother, where `.bvectvpalg`
    draws one `(n_a * tt)`-dimensional normal. Same target, and the same choice
    every other time-varying model here already made.
  - Variable selection may not reach the loadings. `.bvectvpalg` applies BVS to
    the whole of `a`; excluding a loading is a change in the rank of Pi, which
    beta's state equation does not model, so `validate()` rejects it — the rule
    `VecNormalWishart` already enforced.
  - `beta`'s state equation takes an autoregression `rho`, new in
    `TvpCointSpacePrior` and read from `/priors/beta/rho`. `.bvectvpalg`
    hardcodes the random walk; this defaults to `0.999`, the Koop,
    Leon-Gonzalez and Strachan (2011) form, so that beta_t has a stationary
    distribution and the prior on it is proper. A random walk in a parameter
    identified only up to scale has nothing pulling it back, and its variance
    grows over the sample. `rho = 1` is still accepted. The innovation variance
    stays the identity either way — that is what pins beta's scale against
    alpha's, and it is not a knob.

  The forecast is the constant VEC's: the last in-sample period of `a`, `beta`
  and the precision is rewritten as the level VAR it implies and simulated from
  there, so `/data/forecast/z` is expected in the level layout. The log
  likelihood scores every period under its own coefficients and its own
  cointegration vectors.

  `test/make_model_fixture.cpp` can write the model, unlike the other VEC, so
  six fixtures cover it from a clean clone: plain, BVS, covariance block, both,
  structural and no forecast.

* **`test/diff_fingerprints.sh`**, which compares two fingerprint recordings by
  fixture rather than by line. Test tooling and documentation only — nothing
  under `src/` or `include/` was touched, so draws are unchanged by
  construction.

  `diff before.txt after.txt` prints nothing when no number moved and every
  fingerprint line of every affected fixture when one did, so a change to a
  shared algorithm produces tens of thousands of lines that say no more than the
  list of fixture names would. The new script prints that list, prints one
  fixture's lines when given its name as a third argument, and exits non-zero
  when anything moved or appears on only one side. Either argument may be a
  `record_fingerprints.sh` recording or a raw `ctest -V` redirect: both are
  reduced to headers and fingerprint lines before anything is compared, so the
  two forms are comparable against each other and recordings made before this
  existed are still usable.

  Its verdict was checked against the `fingerprints.yml` step summary, which
  reports the same comparison for a pull request and keeps its own
  implementation of it — on the psi-scope-fix recordings both name the same two
  fixtures, `VarTvpGamma-bvs-covar` and `VarTvpStochvol-bvs-covar`.

  README, `CONTRIBUTING.md` and `CLAUDE.md` gave `ctest -V > file` followed by a
  plain `diff` as the recipe, which is where the 400–800 KB recordings in
  `test/baselines/` came from; all three now give the two scripts instead.

### Changed

* **README: the places it had drifted from the code are corrected.**
  Documentation only — no sampler, no header and no I/O code was touched, so
  draws are unchanged by construction. Each was checked against the source named
  beside it:

  * The coverage sentence still read "twelve of the fourteen samplers", from
    before `DfmNormalStochvol`. Thirteen of fifteen have a generated fixture —
    the distinct model column of `test/CMakeLists.txt`'s
    `bayests_add_model_fixture` calls. The same two numbers were stale in
    `CLAUDE.md`, along with a test count of 133 that is now 138 from a clean
    clone.
  * "The twelve VARs and VECs support … a structural form" contradicted the
    README's own structural paragraph. `require_identified_structural()` refuses
    it for the four Wishart models and `VecKlgs2010`; eight of the twelve take
    it.
  * The `wishart` spelling of `error` was attributed to `VarTvpWishart` alone.
    Both `var_tvp_wishart_io.cpp` and `vec_tvp_wishart_io.cpp` compare against
    it, and neither model has a psi block, so it turns no covariance block on —
    which is the opposite of what the sentence around it says.
  * The model-file table was missing every VEC cointegration path
    (`/priors/beta`, `/initial/beta`, `/posterior/beta/coeffs`), the DFM's
    `/initial/lambda`, `/initial/v_sigma_inv`, `/initial/u_h`, `/initial/v_h`,
    `/posterior/lambda/coeffs`, `/posterior/factors/coeffs` and
    `/posterior/v_sigma_inv/coeffs`, and the psi block's own `varsel` attribute
    at `/model/priors/psi` — which is what decides selection on the covariance
    block in the four time-varying models that have one, and which the README
    had never mentioned in either place.
  * `--no-coefficients` and the `--group=<path>` spelling are documented;
    `BAYESTS_VEC_FIXTURE` and `BAYESTS_RUNTIME_DEP_DIRS` join the build-options
    table, both of which the prose already used.
  * "Two algorithms carry the implementation weight" is three: the README's own
    DFM paragraph describes `chan_jeliazkov_2009` as the third. The references
    section listed two of the six works the body cites or the source implements,
    and now lists all six.
  * Two sentences pointed at `test/golden_models.cpp` for "the recorded
    fingerprints", which it does not hold and which the Tests section says is
    never checked in.

* **README: the model-file orientation is now stated for both kinds of reader,
  and the algorithm count is fifteen.** Documentation only — no sampler, no
  header and no I/O code was touched, so draws are unchanged by construction.

  The old wording, "draws run along the **rows** on disk", is true of what an R
  session sees and false of what the file holds. In HDF5's own dataspace terms
  every stored matrix is one row per quantity and one column per draw — `h5py`
  reports `/posterior/a/coeffs` of a 12-parameter, 80-draw model as `(12, 80)`
  — and R's readers show the transpose because R is column-major where HDF5 is
  row-major. Both orientations are now given, and named as such, since a
  downstream reading these files from Python or C was being told the wrong
  thing. Verified against `write_armadillo_matrix_to_hdf5`, which transposes
  the already-transposed `write_draws` argument back to quantity-by-draws
  before writing, and against fixtures read from both `h5py` and `hdf5r`.

  The count sentence still said fourteen, and enumerated one dynamic factor
  model, from before `DfmNormalStochvol` was added above.

* **`chan_jeliazkov_2009` accepts a constant `z`.** It was the one argument of
  the four that had to arrive as a stack of one block per period; `sigma_u`,
  `sigma_v` and `B` all already took either form. A `z` of `K` rows now means one
  measurement matrix that holds for every period, and in that case — with a
  constant `sigma_u` alongside it — `Z' Sigma_u^-1 Z` is the same block
  throughout and is formed once instead of `T` times, which turns the assembly
  from O(T K M²) into O(K M²).

  Added for `DfmNormalGamma`, whose measurement matrix is its loading matrix and
  whose `K` is large by construction: replicating it `T` times and re-deriving the
  same `M x M` block from it every period was the dominant cost of a draw.

  *Draws are unchanged, bit for bit.* The constant path computes the same product
  the loop computed, once, and `test/unit_chan_jeliazkov.cpp` asserts exact
  equality between a constant `z` and its replication at three sample lengths and
  with `sigma_u` in both forms. One test changed with it: the rejected-input case
  passed a `z` of `K` rows to show that a wrong height is refused, and `K` rows
  are now valid, so it passes `K + 1`.

* **`kalman_durbin_koopman_2002` stops decomposing the same matrix once per
  period.** Each of `sigma_u`, `sigma_v` and `B` may be given as one matrix that
  holds for every period or as a stack of one per period, and both forms remain
  supported in any combination — a time varying error covariance, state
  innovation covariance and transition are all still available. What changed is
  how the constant form is reached. It used to be replicated into `T` copies of
  itself up front, which allocated a `KT x K` or `MT x M` matrix the caller had
  not asked for and, worse, left the loops unable to see that the blocks were
  identical: a constant covariance was eigendecomposed `T` times. The samplers
  hand it constant covariances, and `VarTvpWishart` and `VarTvpGamma` hand it a
  *diagonal* one, so most of that work produced a matrix whose square root is an
  elementwise `sqrt`. The blocks are now indexed with a stride that is zero for
  the constant form, so the body is written once, nothing is replicated, and a
  covariance that holds throughout is decomposed once.

  *Draws are unchanged.* Bit-identical, not to a rounding error: computing one
  eigendecomposition instead of `T` copies of it yields the same numbers, and the
  order in which the random number generator is consumed did not move. Verified
  three ways. Against the previous implementation compiled alongside the new one,
  over `T` in {2, 5, 40, 120}, `K` in {1, 3}, and all six combinations of
  constant and time varying arguments: worst difference exactly zero. The 60
  `Tvp` golden fixtures are green. And from R, through the vendored copy in
  bvartools, the constant, stacked and genuinely time varying cases are all
  `identical()` to draws recorded before the change.

  1.9x faster on the shape the samplers use — `T = 89`, `K = 3`, `M = 21`,
  constant covariances — measured against the previous implementation in the same
  binary. The gain is entirely the removed decompositions, so it grows with `T`
  and vanishes when every argument really is time varying.

  Two further changes are deliberately *not* here, because both would move the
  numbers while leaving the distribution intact, which is the most expensive kind
  of change to verify: taking the square root by Cholesky rather than by
  eigendecomposition (roughly three times cheaper per call, but a different `A`
  and so a different draw from the same random numbers), and a fast path for the
  diagonal case. `symmetric_sqrt` in that file records why it is an
  eigendecomposition.

  The signature also takes all seven arguments by `const` reference now. It used
  to take three by value and mutate them, which is what the replication needed;
  they are no longer touched. Existing call sites compile unchanged, and the
  `const_cast` at `var_tvp_stochvol.cpp:195` is now redundant.

* **A structural model now requires a diagonal error covariance, and is rejected
  without one.** `A_0` is unit lower triangular with `k(k-1)/2` free elements;
  the data determine only the reduced form, whose error covariance
  `Omega = A_0^-1 Sigma A_0^-T` has `k(k+1)/2`. Against a diagonal `Sigma` the
  count is exact and `(A_0, diag Sigma)` is the unique LDL factor of `Omega` —
  the recursive SVAR. Against an unrestricted `Sigma` the structural side
  carries `k^2` parameters, and a `k(k-1)/2` dimensional set of them fits
  identically: the likelihood is flat along it and a draw of `A_0` is the prior
  plus wherever the chain last wandered.

  Two things leave `Sigma` unrestricted, and the second is the one easily
  missed: a Wishart prior on the error precision, and a covariance block, since
  `Psi` is then a second unit lower triangular matrix doing `A_0`'s job. Both
  are now refused by `validate()` in all twelve models when
  `spec.n_structural() > 0`, with a message that counts the parameters out and
  names the way forward. `structural` therefore pairs with `gamma` or `sv`
  without a covariance block, and nothing else. With `sv` it is better than
  exactly identified — the volatility moving over the sample identifies `A_0`
  through heteroskedasticity.

  *Draws are unchanged.* Nothing that was estimable before is estimable
  differently now; two configurations that used to run stop running. Both were
  fixtures added earlier in this same release — `VarTvpWishart-structural` and
  `VecTvpWishart-structural` — and neither is in any tagged version, so no
  recorded result changes. Every remaining structural fixture pairs `A_0` with a
  diagonal covariance and its fingerprints are byte-identical.

  Rejected rather than warned about, deliberately. Inference on the unidentified
  configurations would still be coherent under a proper prior, and everything
  these models *report* is a function of the reduced form alone — forecasts and
  the pointwise log likelihood are invariant to position on the ridge and would
  be correct. But the reason to set the flag is to read `A_0`, and there it
  would be noise wearing the shape of an estimate.

  `test/unit_identification.cpp` pins both directions: the three sound
  combinations are accepted and the two unidentified ones rejected. It is a unit
  test rather than a fixture because the golden harness cannot express a
  rejection — the front-ends swallow the exception, so a refused input reads
  there as a passing test that wrote nothing. `make_model_fixture` refuses the
  same combinations for the same reason.

### Fixed

* **The seven-component stochastic volatility mixture was attributed to the
  wrong author.** `stochvol_ksc_1998.cpp`, `stochvol_mixture.h` and
  `stochvol_ocsn_2007.cpp` named it "Kohn, Shephard and Chib (1998)" in five doc
  comments. It is Kim, Shephard and Chib (1998) — the KSC the file name has
  carried all along, and the citation README already gives correctly. Comments
  only; no constant, no code path and no header changed, so draws are unchanged
  by construction.

* **Every time varying parameter sampler read the simulation smoother's output
  one period late.** `kalman_durbin_koopman_2002` returns `M x (T+1)` columns, of
  which column `i` is the state period `i`'s observation loads on for
  `i = 0 ... T-1`; column `T` is the transition applied once past the end of the
  sample and is informed by no observation. Every caller kept `.cols(1, tt)` —
  states 2 to T+1 — and then paired them with the regressors of periods 1 to T.

  Within the same iteration that put the wrong period's coefficients into four
  places at once. The residuals `u` were formed against next period's
  coefficients, so `Sigma`'s posterior scale was inflated and fed straight back
  into the next smoother pass. The state innovation variance saw `a_2 - a_0` as
  one increment when its variance is `2Q`, never saw `a_1 - a_0` at all, and
  counted `a_{T+1} - a_T` — a step drawn from the prior — in its place, so
  `a_sigma` came out biased up. The initial state was drawn from `a_2` using
  `Q^-1` where it needed `(2Q)^-1`. And BVS scored its inclusion candidates
  against the shifted path. On the way out, every reported coefficient path was
  shifted a period ahead of the data, with the last period carrying a draw the
  data had never touched — which is also the period `read_draws_at_period()`
  hands to the forecast.

  Now `.cols(0, tt - 1)`, at all thirteen call sites: the `a` block of all six
  TVP models, the `psi` block of `VarTvpGamma`, `VarTvpStochvol`, `VecTvpGamma`
  and `VecTvpStochvol`, and the `beta` block of the three time varying VECs.
  Nothing downstream of the slice needed changing — `a_lag`, the `a0` draw and
  the residual loop were all already written for the correct alignment, which is
  what made the slice the only thing wrong.

  *Draws change*, for `VarTvpWishart`, `VarTvpGamma`, `VarTvpStochvol`,
  `VecTvpWishart`, `VecTvpGamma` and `VecTvpStochvol`, in every configuration —
  there is no path through any of them that did not go through the shift. The
  size of the change scales with the state innovation variance, so a tightly
  shrunk path moves little and a freely moving one moves a lot; either way the
  old numbers answered a question about the wrong period. The constant
  coefficient and constant covariance samplers are untouched: they never call the
  smoother.

  The new numbers are right because the alignment is not a convention to choose
  but a property of the recursions, and it is now pinned two ways. An independent
  port of the function was compared against the analytic posterior of a small
  linear Gaussian model computed in closed form — 200,000 draws, mean and full
  covariance matching to Monte Carlo error on `.cols(0, T - 1)`, and the mean off
  by up to 2.3 posterior standard deviations on `.cols(1, T)`. In the repository, the
  identity added to `unit_kalman.cpp` below asserts it directly.

  How it went unnoticed: the smoother is correct, and every identity
  `unit_kalman.cpp` had compared one call of it against another — a shift is
  present in both sides of such a comparison and cancels. The docstring said
  "the initial state in column 0", which reads as though column 0 precedes the
  sample and the first observed state is therefore column 1. It does not: column
  0 is `a_1`, already smoothed against every observation. That note has been
  rewritten to say which column belongs to which period and what a caller that
  keeps the last one has done. The golden harness could not have caught this
  either — it fails only on a fixture that throws, and a shifted path is a
  perfectly well formed one.

* **Half the forecasts ignored the contemporaneous coefficients of a structural
  model.** `VarNormalWishart`, `VarTvpWishart` and `VarTvpStochvol` never split
  the trailing `k(k-1)/2` rows off `a`, and never applied `A_0^{-1}` to the
  simulated path. The other three did. Because every VEC forecast converts to
  its level parameterisation and hands the path to `VarNormalWishartSampler`,
  all six VECs inherited it too — a structural VEC could not be forecast at all.

  *Draws change*, in exactly one place across the whole recorded suite:
  `VarTvpStochvol-structural` now has a `/posterior/forecast` where it had none.
  That is the whole diff. Every other fingerprint is byte-identical, including
  the three samplers whose implementation was refactored onto the new shared
  helpers and whose `A_0^{-1}` was hoisted out of the horizon loop.

  How it went unnoticed: the failure was loud but swallowed. With `z` supplied
  correctly the coefficient count no longer matched and the sampler threw, the
  `BaseModel` front-end caught it and printed to stderr, and the golden harness
  only fails on a fixture that throws all the way out — so a test asking for
  `h = 4` passed while writing no forecast. `VarTvpStochvol-structural` had been
  doing exactly that. There is a silent version too, for a caller who supplies
  `z` *with* those columns: the counts then match, the contemporaneous
  coefficients multiply whatever is in them, and `A_0^{-1}` is never applied.

  The split now lives once, in `core/models/model_support.h`, as
  `split_structural_coefficients()` and `structural_inverse()`, and all six VAR
  forecasts use it. `nparams` is still each caller's own — off the posterior
  where the coefficients are constant, off the spec where they are a path and
  the posterior holds one period — because that is the one part that legitimately
  differs; splitting on `z.n_cols` is the mistake the helper's comment warns
  against.

* **`VarTvpWishart` sliced its coefficient path at the wrong stride to forecast
  from.** Its HDF5 reader took the width of a period from `z.n_cols`, which for
  a structural model is short by the contemporaneous block, so
  `read_draws_at_period()` cut the stored path across period boundaries and
  returned a matrix that was neither the last period nor any other. Its two
  siblings already counted off the spec. Found by the fixture added above: with
  the sampler fixed, this was what still stopped that model forecasting.

  *Draws are unchanged* for every non-structural model — the two counts agree
  when there is no contemporaneous block — which the fingerprint comparison
  confirms. The identification rule above has since made a structural
  `VarTvpWishart` unreachable, so this is now defensive rather than load-bearing;
  it is kept because the spelling it replaces was wrong on its own terms and
  neither sibling shares it.

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

* **Armadillo and HDF5 are found without a CMake config package.** Both were
  required in CONFIG mode, which is what vcpkg, MSYS2 and a source build install
  — and what a distribution does not. Debian's `libarmadillo-dev` ships no
  `ArmadilloConfig.cmake`, so `find_package(Armadillo CONFIG REQUIRED)` failed
  outright; `libhdf5-dev` ships pkg-config files and `h5cc` but no
  `hdf5-config.cmake`, so the target-name search added above had nothing to
  search. CMake's own `FindArmadillo` and `FindHDF5` read both, and are now tried
  after the config packages: the module defines variables but no target for
  Armadillo, so the `armadillo` target the four internal libraries link is built
  from them, and `FindHDF5`'s `HDF5::HDF5` joins the candidate list as its last
  entry. A machine with both a config package and a system one keeps linking
  what it always did. This is what makes a build against distribution packages
  possible at all, and the snap below is its first consumer.

  *Draws are unchanged* on any machine that was already building — the fallbacks
  are reached only where the configure previously failed. On a machine reaching
  them for the first time the numbers are a separate question, and not one this
  change decides: a distribution's Armadillo links whichever BLAS the
  distribution chose, and fingerprints have never been comparable across
  toolchains. Full suite, 67 tests, passes unchanged against vcpkg.

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

* A snap package, in `snap/snapcraft.yaml`. `bayests` builds as a strictly
  confined snap for amd64 and arm64 on the `core24` base, taking Armadillo,
  HDF5, OpenBLAS and the Fortran runtime from Ubuntu 24.04 rather than vcpkg —
  which the dependency-discovery change above is what allows. HighFive is the
  exception and is cloned from the tag in `.github/highfive-version`, because
  the archive is still on 2.x; that pin is repeated as `source-tag` in the yaml
  and the two have to be bumped together, since 3.0.0's betas all report 3.0.0
  in their headers and no check can tell them apart. The snap's version is read
  out of `project(VERSION)` at pull time by the expression `release.yml` already
  uses, so it cannot drift from the source. Confinement is strict: `$HOME`
  through the `home` interface, anything else through `removable-media`, which
  the user connects.

  HighFive's major version is now checked wherever it is found. The search no
  longer suppresses the compiler's default include directories — it had to stop,
  or a HighFive installed at `/usr/include/highfive` would have been invisible
  to exactly the builds this is for — and 2.x is what a system directory is
  likely to hold. It fails with the version it found instead of a page of
  template errors naming neither.

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
