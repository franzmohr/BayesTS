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

New entries go here, under an `### Added`, `### Changed` or `### Fixed`
heading, and move down into a version section when one is cut.

## 0.3.0 — 2026-09-22

### Added

- **`VarTvpDiscount` and `VecTvpDiscount`, two models with an answer rather than
  a chain.** The matrix normal dynamic linear model of West and Harrison (1997,
  ch. 16) with Uhlig's (1997) discounted Wishart on the error precision: the
  coefficients follow a random walk and the error covariance drifts, and both
  posteriors are closed form. `bayests` runs them through the same three stages
  as everything else, and `coefficients` is still the stage that estimates them;
  it just consumes no random numbers.

  Two new `/model` attributes drive them, both in `(0, 1]` and both defaulting
  to 1. `delta_beta` discounts the coefficient covariance each period, which is
  exactly a random walk whose innovation covariance is
  `((1 - delta) / delta) C_{t-1}`; `delta_sigma` discounts the Wishart, a
  stochastic volatility law of its own — multiplicative beta shocks to the
  precision, where `stochvol_ocsn_2007` approximates a Gaussian autoregression
  in the log variance. **Both at one is not a no-op** but the conjugate normal
  inverse Wishart posterior of a constant coefficient model, computed one period
  at a time, and that identity is what `unit.var_tvp_discount` pins them
  against.

  What no chain changes in the file:

  - The regressors are `/data/train/x`, the compact layout `VecKlgs2010` reads,
    and a file carrying only `/data/train/z` is refused. For `VecTvpDiscount`
    that dataset holds the short-run blocks alone; the `rank` error correction
    columns are built from `w` and `beta`.
  - `/priors/a` holds `mean` and `cov`, not `mu` and `v_inv`. A matrix normal
    prior: `mean` is the `n_x` by `k` coefficient matrix and `cov` the regressor
    side of its covariance, the equation side being the error covariance the
    Wishart prior already carries. Different names for a different object, so a
    file bringing the other pair is read as having no coefficient prior at all
    and `bayests check` says which datasets nothing opened.
  - `burnin` must be 0 and `thin` 1. Refused rather than ignored, because both
    reach the file: the `start`/`end`/`thin` attributes beside a forecast come
    from `thin`, and draws labelled as though a sweep had happened are output
    that looks like output. `iterations` keeps its meaning — how many i.i.d.
    draws a forecast takes.
  - `/model/seed` reaches the forecast and nothing else. Estimation draws no
    random numbers, so two runs agree to the bit.
  - **The posterior is written as a posterior.** `/posterior/a/mean`,
    `a/scale`, `a/cov`, `u_sigma/scale` and `df`, one column per *period* rather
    than per draw and with no `start`/`end`/`thin` beside them. `u_sigma/scale`
    is a covariance, which is why it is not at `u_sigma_inv`. There is
    deliberately **no `/posterior/a/coeffs`**: the posterior of one period is
    exact and easy to draw from, but joining one draw per period would look
    exactly like a sampled coefficient path and is not one, the smoothed
    posterior being dependent across periods. `/posterior/loglik` has one row
    instead of one per draw, the parameters being integrated out exactly, and
    summed it is the log marginal likelihood of the sample.

  **`VecTvpDiscount` holds the cointegration space fixed**, and that is the one
  assumption separating it from the three sampling VECs. `/initial/beta` is read
  from where every VEC keeps its space and in the same layout, but not as a
  starting value: nothing iterates, so it is what the run conditions on, and it
  is written back to `/posterior/beta/coeffs` so the loadings have a relation
  attached. Conditional on it a VEC *is* a VAR in the regressors
  `[beta' w_t, the short-run blocks]`, which every equation shares, and the
  whole of `VarTvpDiscount` applies unchanged.

  The space cannot be allowed to move and stay closed form, and the reason worth
  writing down is not the conjugacy. A VEC's measurement is `alpha_t beta_t' w_t`,
  two latent blocks multiplying each other, which is why the three sampling VECs
  alternate two simulation smoother passes rather than filtering once; a discount
  replaces one pass, not the alternation. Beyond that, `TvpCointSpacePrior` fixes
  the innovation variance of `beta_t` at the identity *because that is what pins
  beta's scale against alpha's*, and a discount replaces a fixed innovation
  variance with `((1 - delta) / delta) C_{t-1}`, which moves with the data. So
  discounting the space is not a cheaper Koop, León-González and Strachan (2011)
  — it is a model in which alpha and beta slide against each other along a ridge.
  Use `VecTvpDiscount` to ask how the *adjustment* to a given long-run relation
  has moved, and `VecTvpWishart` and its two siblings to ask whether the relation
  itself did.

  What the fixed space buys back is a number the samplers cannot give cheaply:
  the sum of `/posterior/loglik` is the exact marginal likelihood of the sample
  given that space, the rank and the two discounts, at one pass over the sample.
  A grid over candidate spaces, ranks or discounts is then a list of files with
  no chain run for any of them, which is also why a grid over the two discounts
  is a list of models rather than a vector in one file.

  What they give up against the samplers is stated rather than hidden. The
  volatility does not mean revert; one discount factor stands in for both the
  persistence and the variance of the volatility process, so `delta_sigma` is
  not the autoregressive parameter of a log volatility; and the degrees of
  freedom converge to `1 / (1 - delta_sigma)` whatever the prior says, which
  must exceed `k` and is refused where it does not. Variable selection and
  `structural` are refused too — there are no draws for an inclusion indicator
  to accompany, and an inverse Wishart posterior leaves `Sigma` unrestricted.

  **Draws are unchanged** for every other model. No sampler was touched; the two
  new `/model` attributes are read through defaults of 1 that only these two
  models consult, and `read_spec()` is otherwise as it was. Verified rather than
  assumed: all 102 pre-existing fixtures print their previous fingerprints digit
  for digit, recorded from this commit and from its parent on one machine and
  one build, and the four new rows are the only ones added.

  One thing to know before running that comparison yourself. `bayests_golden`
  fingerprints the union of every model's outputs, so the five datasets above
  are now printed for *every* fixture — as `absent` on the other hundred and two
  — and `diff_fingerprints.sh` therefore reports all of them as moved across
  this change. Strip the five `absent` lines before comparing, or compare only
  recordings taken on the same side of it.

- **A `pre-push` hook that runs the Linux CI jobs before a push leaves the
  machine**, in `.githooks/`. `git config core.hooksPath .githooks`, once per
  clone, and `git push` builds and tests the commits it is about to send in the
  image `docker/` already carried, refusing the push if `ctest` fails. Nothing
  about the jobs is duplicated -- it is the same image and the same
  `docker/ci.sh` -- only when they run.

  What it mounts is the *commit being pushed*, checked out into a throwaway
  worktree under `build/pre-push-src`, rather than the working tree. That is
  what the runner will see; running the image by hand remains the way to ask
  whether what is on disk right now would survive, and the hook says which it
  is testing when the tree is dirty. The whole matrix runs, `Debug` then
  `Release`, incremental after the first push through the `bayests-ci-work`
  volume; `BAYESTS_PREPUSH_JOB="ci Release"` cuts it to one leg,
  `BAYESTS_PREPUSH=off` skips it and `git push --no-verify` bypasses it.

  It fails closed: no docker, no running daemon or no image is a refusal naming
  the command that fixes it, not a pass. Output goes to `build/docker-out`, and
  the hook clears the packages there before a run that will produce new ones --
  `docker/ci.sh` copies an archive out by the name CPack just used and leaves
  the previous version's beside it, so the directory otherwise accumulates a set
  per version and describes two builds as though they were one. Draws are
  unaffected -- no file under `src/`, `include/` or `test/` is touched.

- **`/posterior/forecast/loglik`, the score of a forecast.** `bayests forecasts`
  writes it wherever the file carries `/data/test/y`, one column per realised
  period and one row per draw. The reserved name of the forecast group is a
  name no longer.

  **Each column conditions on the realised observations before it**, not on the
  path the forecast simulated. Column `i` is the one step ahead predictive
  density of period `T+i` given everything known up to `T+i-1`, so the log of
  the mean of `exp()` over draws, summed over the columns, is
  `log p(y*_{T+1..T+n} | data)` -- the log predictive likelihood of the whole
  realised stretch, the joint, factorised. Scoring against a simulated history
  would instead give the marginal density of each horizon on its own: a
  defensible quantity, a different one, and one that cannot be summed, since
  marginals do not make a joint.

  That choice is what makes it cheap. With the history realised rather than
  simulated, the regressors of the scored periods do not depend on the draw, so
  the score is the model's own pointwise log likelihood over those periods on a
  sample whose lag blocks are the realised values. Nothing new is written down
  per algorithm, which is what keeps one density per model rather than two that
  can drift apart; `unit.predictive_score` pins the two against each other by
  handing a model a stretch of its own sample as what its horizon realised.

  **A drifting model is scored under states carried forward.** The six VARs are
  scored today -- `VarNormalWishart`, `VarNormalGamma`, `VarNormalStochvol`,
  `VarTvpWishart`, `VarTvpGamma` and `VarTvpStochvol` -- and for the four that
  let something move, each draw's coefficients, log-volatilities and `Psi` take
  one step of their random walk per scored period, before the period they belong
  to, by the same rule a forecast carries them. `/model/forecast_states` decides
  whether they step at all: under `hold` the sample's last state is repeated,
  which scores the model whose drift stops where the sample does, a different
  model from the one estimated and worth asking for only deliberately.

  One state path per draw is one sample of it, which is all the density needs --
  averaging `exp()` over draws integrates the state out with everything else the
  posterior carries. It does mean the score is **drawn rather than computed**,
  so two runs under `simulate` give two answers, as two forecasts do, and
  `/model/seed` is what repeats either.

  **A VEC is scored in levels**, the parameterisation it forecasts in: its draws
  are rewritten as the level VAR they imply and the score is that VAR's, which
  is what the forecast already does. So `/data/test/y` of a VEC holds the
  realised **levels**, the series `/data/forecast/x` carries the lags of, and
  not the differences `/data/train/y` holds.

  A drifting VEC moves more than a drifting VAR does: its cointegration vectors
  step by their own state equation rather than by a random walk, and the level
  coefficients are not linear in the states, so the change of basis is made
  again at every scored period rather than once per draw. It is the same walk
  the forecast takes -- each model's is written once and both drivers call it --
  so a file cannot be forecast under one set of states and scored under another.

  **A factor model is scored by filtering**, which is a different problem.
  Every other model reaches its realised history through its regressors, so the
  density is its own pointwise log likelihood on another sample. A factor
  model's history reaches the density through the latent factors, and its
  in-sample likelihood conditions on the factors the sampler drew -- of which
  there are none outside the sample. So at every scored period the realised `y`
  updates the distribution of the factors before the next is predicted, and the
  column is that period's prediction error decomposition: a Kalman filter over
  the scored periods, per draw, started from the drawn factors and certain of
  them. Summing the columns is then the joint log density of the realised
  stretch, as it is for a VAR. Simulating the factors forward instead, as the
  forecast does, would give each horizon's marginal density: defensible,
  different, and not what this dataset holds elsewhere. The filter draws
  nothing, so a factor model's score repeats without a seed; what moves between
  runs is a `Stochvol` model's volatility under `simulate`.

  A time-varying factor model steps its states per scored period in the order
  the forecast steps them, and **only the free elements of `Lambda` walk**: the
  identifying block is fixed, was never drawn in any period of the sample, and a
  score that let it move would change the rotation and the scale the factors
  were estimated under.

  **Seventeen of the twenty algorithms are scored**: every VAR, every VEC and
  every dynamic factor model. `FavarNormalWishart` has no entry point yet. A
  structural model refuses for good, its regressors holding the contemporaneous
  observations and its density the Jacobian of `A_0`, and the two quantile
  models never reach it, having no forecast at all. `bayests check` says which
  side of that line a file is on.

  The two members of the forecast group are now asked for separately, so adding
  `/data/test/y` to a file that was already forecast is enough to score it; the
  paths do not have to be thrown away first. **Draws are unchanged**: the score
  draws nothing -- it evaluates a density and touches the generator not at all --
  and the 341 tests pass.

- **`/data/test/y`, the observations a forecast is scored against.** One row per
  period and one column per variable, in the layout and the variable order of
  `/data/train/y`, and optional. It is the one thing in `/data` that no sampler
  reads: not a sample anything is estimated on, but what the horizon turned out
  to be.

  It is in the file so that one window of an expanding window exercise carries
  what it is to be judged by. Scoring a folder of them is then one file at a
  time, where before the caller had to hold window *i+1* in memory to supply the
  observation window *i* predicted. That is also where the reserved
  `/posterior/forecast/loglik` will take its observations from.

  Fewer than `h` periods is accepted and means what it says -- the last windows
  of an expanding window realise fewer periods than they forecast. More than `h`
  is refused, since the file then describes a horizon other than the one
  `/model` asks for, and so is the stacked spelling `/data/train/y` also
  accepts: with `h` unknown, `h*k` numbers in one column could be in either
  order, and scoring against the wrong one would be a reshuffle of the right
  numbers. `bayests check` reports how many periods are there and warns where
  there are some but no horizon to use them for.

  Nothing computes from it yet. Every model's reader takes it into
  `Input::test.y`, which is what keeps `bayests check` from reporting it as a
  dataset the model never reads; the forecast score that consumes it is still to
  come. **Draws are unchanged**: no sampler sees this, and the 340 tests pass.

- **Six fixtures that are scored, and a golden test that insists on it.** No
  generated fixture carried `/data/test/y`, so nothing in the suite ever reached
  the predictive density: nineteen algorithms can be scored and the code that
  scores them was covered by unit tests alone. That is the gap the bug behind
  the positive semi-definite filter fix below came through, found by a third
  toolchain rather than by CI.

  `make_model_fixture` takes a `--score` flag, which writes the observations the
  horizon realised, and `bayests_golden` now fails a fixture that carries them
  and comes back without `/posterior/forecast/loglik`. One fixture per distinct
  implementation rather than one per algorithm: `VarNormalWishart-score` and
  `VarTvpStochvol-score` for `predictive_score.h` with states held and drifting,
  `VecNormalGamma-score` for a VEC scored in levels, `DfmTvpGamma-score` for
  `factor_score.h`, and `VarTvpDiscount-score` and `VecTvpDiscount-score` for
  the closed form the discounted pair carry.

  *Draws are unchanged.* The realised values come from a generator of their own,
  so the file's stream is consumed identically whether the flag is passed or
  not, and the new golden line is printed only for a fixture that asked to be
  scored. Verified rather than assumed: `record_fingerprints.sh` before and
  after, over the full suite on one machine and one build, reports **106
  fixtures unchanged, 0 moved**, the difference being the six new ones.

- **`BAYESTS_WERROR`, off by default and on in CI.** The project already
  compiled with `-Wall -Wextra` and produced no warnings of its own; nothing
  held it there. It is off for anyone building from source, because a newer
  compiler than a release was tested with finds new warnings and a user wants a
  binary rather than a diagnostic.

  `-Wfree-nonheap-object` is exempted under GCC, as a warning rather than an
  error. It fires inside Armadillo's `memory::release()` once the optimiser has
  inlined an expression template into the destructor of the temporary that built
  it, names an ordinary local `arma::mat` as the object being freed, and is not
  silenced by the SYSTEM include directories -- it is raised after inlining,
  where the header a line came from no longer decides.

- **Two exit codes asserted that were not.** `cli.refusals` covered 0 and 2 and
  never 1, though ten sites in `src/` return it and the difference is what tells
  a caller scripting `bayests` over a tree that a file is broken rather than
  that the command line was. Three cases now pin the line: a path that does not
  exist is 2 (nothing was opened, so nothing started), while a file named `.h5`
  that is not HDF5, and a file whose extension `is_hdf5_file()` turns away, are
  both 1.

- **`agents.recipes` runs everywhere the suite does.** The Windows job installed
  no h5py, so the test was dropped silently by the default
  `BAYESTS_TEST_AGENT_DOCS=AUTO` and the Python examples in `agents/` were
  checked on Linux alone. So did the image in `docker/`, which is the gate this
  project runs before a push -- it was passing a suite one test shorter than the
  runner's and saying so in a line of status output. All three now install h5py,
  name the interpreter and ask for the test by name, so losing it fails the job
  rather than shrinking it.

- **`VecTvpGamma` takes the non-centred prior too**, for its coefficients
  (loadings included) and its covariance block, as `VarTvpGamma` does; the
  cointegration space keeps its fixed unit state variance and the error
  precision its gamma prior. The covariance block reads one error covariance
  for every period, which `draw_noncentred_path()` already takes.
  `unit.noncentred` runs `VecTvpGamma` with a covariance block on a
  cointegrated pair whose first constant shifts, and two fixtures,
  `VecTvpGamma-noncentred` and `VecTvpGamma-noncentred-bvs-covar`, put the path
  through `golden.*` and `check.*`. *Draws are unchanged* for every file
  without `omega_v` and for the six existing non-centred fixtures: all
  118 fixtures that existed before fingerprint identically.

- **`VecTvpStochvol` takes the non-centred prior as well.** `omega_v` in place
  of `shape` and `rate` under `/priors/a`, `/priors/psi` or `/priors/u_sigma`
  draws that random walk non-centred and writes `omega`, `omega_log_zero` and
  `omega_log_zero_joint` beside its `sigma`, as `VarTvpStochvol` does. The
  coefficients include the loadings, whose regressors are the draw's
  `beta' w`. The cointegration space is untouched: its state variance is fixed
  at the identity to pin beta's scale, so there is no prior on it to replace.
  `validate_stochvol_block()`, which `VecNormalStochvol` shares, now checks the
  log-volatility's prior through `validate_state_variance_prior()`; that
  model's reader never fills `omega_v`, so what it accepts is unchanged.
  `unit.noncentred` runs `VecTvpStochvol` on a cointegrated pair whose first
  error variance jumps, and two fixtures, `VecTvpStochvol-noncentred` and
  `VecTvpStochvol-noncentred-bvs-covar`, put the path through `golden.*` and
  `check.*`. *Draws are unchanged* for every file without `omega_v`, and for
  the four non-centred VAR fixtures: all 116 fixtures that existed before
  fingerprint identically.

- **`VarTvpGamma` takes the non-centred prior too.** `omega_v` in place of
  `shape` and `rate` under `/priors/a` or `/priors/psi` draws that random walk
  non-centred and writes `omega`, `omega_log_zero` and `omega_log_zero_joint`
  beside its `sigma`, exactly as `VarTvpStochvol` does; the error precision
  does not move in this model, so there is no third block. The covariance
  block's error covariance is one matrix for every period here, so
  `draw_noncentred_path()` now takes one block as well as one per period, and
  `write_noncentred()` moves to `model_io_common` for the two readers to share.
  `unit.noncentred` runs `VarTvpGamma` with a covariance block on a shifting
  intercept, and two fixtures, `VarTvpGamma-noncentred` and
  `VarTvpGamma-noncentred-bvs-covar`, put the path through `golden.*` and
  `check.*`. *Draws are unchanged* for every file without `omega_v`, the
  `VarTvpStochvol` ones with it included: the full fingerprint recording, all
  114 fixtures that existed before, is identical.

- **A non-centred parameterisation of `VarTvpStochvol`'s random walks, and with
  it a test for whether each one moves at all.** Setting `omega_v` in place of
  `shape` and `rate` under `/priors/a`, `/priors/psi` or `/priors/u_sigma`
  writes that block's random walk as `x_t = x_0 + omega * x~_t` with `x~_t` a
  standard random walk, and puts a normal prior `N(0, omega_v)` on the signed
  standard deviation `omega` (Frühwirth-Schnatter and Wagner 2010; for the
  log-volatility, Kastner and Frühwirth-Schnatter 2014). Each draw takes the
  standardised path, then `(x_0, omega)` jointly as a regression on it, then a
  random sign switch. Because "the state does not move" is then `omega = 0`, a
  point inside the prior, the Bayes factor for time variation is a
  Savage-Dickey density ratio that one run estimates (Chan 2018): the sampler
  writes `omega` and the log density at zero of its conditional posterior,
  per state and per block, as `<block>/omega`, `<block>/omega_log_zero` and
  `<block>/omega_log_zero_joint`. `sigma` is still written, as `omega^2`, so
  forecasts, scores and everything else downstream read the block as before.
  A block given both priors is refused. The blocks are switched one by one, so
  a file can, say, test its volatilities while keeping its coefficients
  centred. `unit.noncentred` checks the ordinates against the normal they are
  the marginals of, and runs the sampler on simulated data where nothing moves,
  where one volatility jumps and where one intercept shifts: the Bayes factors
  land on the side the data were generated on. Two fixtures,
  `VarTvpStochvol-noncentred` and `VarTvpStochvol-noncentred-bvs-covar`, put
  the path through `golden.*` and `check.*`, and the golden harness now
  fingerprints the nine new datasets. *Draws are unchanged* for every file
  without `omega_v`: the full fingerprint recording, all 112 fixtures, is
  identical before and after, the only difference being the absent lines for
  the nine new datasets.

- **A warning when `bvs` is selecting against a prior too flat to select
  against, from `bayests check` and from the run itself.** BVS draws an excluded coefficient from its prior and then
  scores that draw against the data, so the flatter the prior the harder it is
  for anything to get back in once it is out, and the inclusion probabilities
  end up describing the prior rather than the data. Korobilis (2013, section
  3.1) puts the point where this takes over at a prior variance of around 100
  and quotes Kuo and Mallick's (1997) usable range of 0.25 to 25. On a
  three-variable fixture with the diagonal of `v_inv` moved from 1 to 0.001,
  mean inclusion across the twelve coefficients went from a spread of 0.10 to
  1.00 down to eleven of the twelve at 0.10 or below.

  The reading is `bayests::flat_selection_prior()` and the sentence is
  `bayests::flat_selection_message()`, both declared in
  `include/bayests/priors.h` so that a host vendoring the core can surface them
  its own way. It reports the diagonal of the prior precision at the selected
  positions -- the conditional prior variance of the draw BVS actually scores,
  which is also what keeps it defined for a singular `v_inv`.

  Two places say it, from that one wording. `bayests check` prints it before
  anything runs, and the seven constant-coefficient samplers that offer `bvs`
  emit it through `Reporter::message()` before their first draw, once per
  selection block -- so an embedded host, an R package or anyone who never runs
  `check` hears it too. Neither refuses the file and the exit code stays 0.
  Constant-coefficient blocks only: a random walk has no one prior variance to
  compare against a threshold, so the time-varying models are left to their
  documentation, which now says so.

  **Draws are unchanged**, verified rather than assumed: the samplers gained a
  call that consumes no random numbers, and a fingerprint recording over all
  112 fixtures before and after the change is identical -- 112 unchanged, 0
  moved, from `test/diff_fingerprints.sh`. The suite passes unchanged, 374
  tests from a clean clone.

### Changed

- **The forecast paths move from `/posterior/forecast` to
  `/posterior/forecast/forecasts`**, turning that path from a dataset into a
  group. **This breaks readers that name the old path**, and every one of them
  has to be changed; there is no compatibility branch.

  The group is the place for everything the forecast periods produce. `bayests`
  writes one member, `forecasts`, and reserves two for a host that holds the
  observations those periods realised: `errors`, realised minus forecast, and
  `loglik`, the log predictive density of the realised observation — the joint
  density of the `k` variables of one horizon, so that it sums over horizons to
  a log predictive likelihood. One dataset could not carry those: the eighteen
  model front-ends ask *have I forecast?* by looking at it, and there was
  nowhere to ask *have I scored?*. The leaves are named after what they hold
  rather than one of them being `draws`, since all three are draws.

  `/posterior/loglik` does not move. It is a different statistic and not the
  same one over other periods: it evaluates each in-sample observation under
  states that have already seen it, which is what WAIC and PSIS-LOO want and
  what a forecast score must not do.

  **Migrating a file is a rerun of `bayests forecasts`.** It replaces the old
  dataset with the group rather than failing on the name collision. Unlinking
  frees the name and not the space, so a file whose forecast was large is worth
  passing through `h5repack` afterwards.

  **Draws are unchanged.** Nothing in `src/core/` computes differently; the
  change is in `src/io/hdf5/` and the guards above it. All 1957 fingerprints
  over the 103 fixtures were recorded before and after on the same release
  build, and the two recordings are identical once the renamed label and its
  column padding are normalised away. The 103 lines that differ are the
  forecast rows of each fixture -- 79 written, 24 `absent` where the fixture
  asks for no horizon -- and every number on them is unchanged.

- **`/posterior/forecast/forecasts` and `/posterior/loglik` now carry the
  `start`, `end` and `thin` attributes** that every dataset under
  `/posterior/<block>/` already had. They are draws of the same chain at the
  same thinning, and the two writers of this format disagreed about it —
  bvartools wrote the attributes on all of its posterior datasets, `bayests` on
  the block ones alone. An R session reading a file written by either now gets
  the same `mcmc` object. **Draws are unchanged**; this adds three attributes
  and touches no value.

- **`main` is the development version, and a release is a tag.** Work happens on
  short-lived branches taken off `main` and merged back into it; a version is
  cut by tagging a commit on `main` and publishing a GitHub release for that
  tag. No branch tracks the last release, so **a downstream package vendoring
  `src/core/` and `include/bayests/` should copy from a tag and record which
  one** — between releases `main` carries whatever this *Unreleased* section
  describes, which is the opposite of what copying from a branch named for a
  release would suggest. CONTRIBUTING.md §*Branches* states the policy. This is
  process only: no source file changed and no draw moves.

- **CONTRIBUTING.md §*Cutting a release* lists the steps, in order.** The policy
  said what a release *is* -- a tag on `main` plus a GitHub release from it --
  and left the sequence to be remembered. Two of the steps cannot be taken at
  release time at all: the version DOI does not exist until Zenodo has archived
  the release, so adding it to `CITATION.cff` and putting the new version in the
  README's worked citation come after publishing. That is already how it has
  been done -- 0.2.0's DOI landed in its own commit the day the tag was made --
  and writing it down is what keeps it from depending on whoever remembers.
  Process only: no source file changed and no draw moves.

- **The simulation smoother leaves the identity transition out of its
  products.** Every time varying coefficient block is a random walk, so its
  transition `B` is the identity, and `kalman_durbin_koopman_2002` still copied
  it every period and multiplied by it as a full M x M matrix twice per period
  in the filter, plus a matrix-vector product in each of the other two passes. A constant `B`
  that is exactly the identity is now recognised once and those products are
  skipped; any other `B`, and a per-period stack, take the general path as
  before. With k = 5 and T = 176 on one thread, one smoother call takes 25% less
  time at M = 60 coefficients (5.1 to 3.8 ms), 35% less at M = 78 (10.2 to
  6.5 ms) and 38% less at M = 120 (34.5 to 21.1 ms), and this call is most of
  an iteration of every `*Tvp*` sampler. *Draws are unchanged*: a product with
  an exact identity is exact, and the one product regrouped by dropping `B`,
  `B P Z' F^-1`, is one Armadillo already evaluates as `B ((P Z') F^-1)`. The
  fingerprint comparison of all 120 fixtures before and after shows none moved,
  and `unit.kalman` now checks the identity's path against the general one
  exactly, at M = 60 and with more equations than states.

- **The four constant VECs refuse a prior the cointegration sampler cannot
  honour.** The collapsed sampler of Koop, León-González and Strachan (2010)
  relies on the loadings' prior being centred at zero and independent of every
  other coefficient (Proposition 1, eq. 8). It rebuilds that prior's precision
  block every draw, and the file's mean and cross-precision were read as given,
  so a non-zero value in either made the three Gibbs blocks condition on three
  different priors. Refused now, in `VecNormalWishart`, `VecKlgs2010`,
  `VecNormalGamma` and `VecNormalStochvol`:
  * a non-zero `/priors/a/mu` on the first `k*rank` positions;
  * a non-zero `/priors/a/v_inv` between those positions and any other;
  * a negative `/priors/beta/v_inv`;
  * a `p_tau_inv` that is not positive definite while `v_inv > 0`;
  * `/priors/beta/g_inv` anywhere but `VecNormalStochvol`.

  *Draws are unchanged* for every file still accepted: `VecNormalWishart` and
  `VecKlgs2010` are bit-identical in the fingerprint recording above.

- **The agent documentation says how `VarTvpStochvol` relates to Primiceri
  (2005), and builds his priors.** `algorithms.md` gains *VarTvpStochvol and
  Primiceri (2005)*: the sampler draws the mixture indicators in the order Del
  Negro and Primiceri (2015) corrected the appendix to, and differs from the
  paper in three ways a reader carrying his priors over has to know -- diagonal
  `Q`, `S` and `W` with an inverse-gamma prior per element, the log-volatility
  on the scale of the variance so that his `W` is a quarter of the one here,
  and no training-sample calibration. A table translates each of his priors
  into the datasets that hold it, reading every inverse Wishart as the
  inverse-gamma marginals of its diagonal. `recipes.md` gains *Primiceri's
  priors from a training sample*, which builds his benchmark -- OLS on 40
  periods, `k_Q = 0.01`, `k_S = 0.1`, `k_W = 0.01` -- for his three variables
  and two lags, and `agents.recipes` runs it: the file passes `bayests check`
  clean, every dataset has the shape the text states, and the structural
  standard deviations come back near the unit shocks it simulated. No code
  changed.

- **`ssvs` refuses three priors it could not honour.** The sweep draws each
  inclusion indicator by weighing `N(0, tau0²)` against `N(0, tau1²)` at its
  coefficient alone, which is George, Sun and Ni (2008, eq. 12 with R = I). The
  coefficient draw read the prior as the file gave it, though, so a file that
  departed from the paper had the two halves of one Gibbs step drawing from
  conditionals of different models, and the chain targeted neither. Refused
  now, at each selected position of `a` and of `psi`, in the four models that
  offer SSVS:

  * a non-zero prior mean -- the spike and slab are centred at zero, and a
    Minnesota mean of 1 on an own first lag was drawn around 1 and scored
    around 0;
  * a prior precision with anything off the diagonal in that row or column --
    the indicators are independent given the coefficients only when the
    selected coefficients are a priori independent of everything else;
  * `tau0` not smaller than `tau1` -- validation checked both were positive and
    nothing more, so a swapped pair ran and every indicator read backwards.

  The documentation now also says that the diagonal of `v_inv` at a selected
  position is read for the first draw only, and gives the paper's guidance on
  choosing `tau0` and `tau1`.

  **Draws are unchanged** for every file still accepted: only `validate()`
  moved, and a fingerprint recording over all 112 fixtures before and after
  says 112 unchanged, 0 moved. A file now refused was sampling a model other
  than the one it described, and has no numbers worth keeping.

- **The agent documentation now says what `bvs` needs from the coefficient
  prior, and what `include` does not cover.** Two ways of getting a run that
  finishes, writes a full posterior and answers a different question than the
  one asked, neither of which anything refuses:

  BVS draws an excluded coefficient from its prior and then scores that draw
  against the data to decide whether to let it back in, so a flat
  `/priors/a/v_inv` makes it very hard for anything to get back in once it is
  out, and pins the inclusion probabilities near zero.
  Korobilis (2013, §3.1) puts the point where this takes over at a prior
  variance around 100 and quotes Kuo and Mallick's (1997) usable range of 0.25
  to 25. `validate_normal_block()` checks that precision for being square and
  symmetric and says nothing about its size, so the symptom is a posterior
  inclusion probability pinned near zero everywhere, which reads as a finding.

  `include` names the positions selection *visits*. Every other position keeps
  the indicator `/initial/a_lambda` starts it at for the whole run, which is
  how an intercept is held unrestricted and equally how a coefficient is
  masked out of every draw without a word.

  **Draws are unchanged**: `agents/`, and nothing under `src/` or `include/`,
  was touched.

### Fixed

- **A factor model's score filter keeps its covariance positive semi-definite.**
  `score_factor_forecast()` updated it in the short form `P - K F K'`, a
  difference of two positive semi-definite matrices, and argued that was safe
  because the variance is rebuilt from the transition every period. It is not:
  `state_noise` carries `diag(v_var)` in the leading `n x n` block and zeros
  elsewhere, so `Q` is singular for any transition of order above one, and the
  lagged blocks of `P` are never refreshed by it — they only shift down
  through `T`. Rounding error accumulates in exactly those blocks until `P`
  drifts indefinite, and the definiteness check then rejects a forecast
  variance that is mathematically fine, throwing *"the one step ahead forecast
  variance of a scored period is not positive definite"* and failing the run.

  The update is now the Joseph form,
  `P = (I - K Z) P (I - K Z)' + K R K'`, a sum of two positive semi-definite
  terms and so one whatever rounding does. Beside it, `F` is factorised with a
  Cholesky rather than put through `arma::log_det()`: `F` is symmetric
  positive definite by construction, so a Cholesky is the test that matches it
  — it fails exactly when the matrix is not one, where `log_det` decides from
  an LU pivot count — and the same factor carries the determinant and both
  solves.

  **Scores change by a rounding error.** Both the determinant and the two solves
  take a different arithmetic path, so the last digits of a factor model's
  `/posterior/forecast/loglik` move. No fixture reaches this code — none
  carries `/data/test/y` — so the fingerprint comparison is silent across the
  change,
  all 106 unchanged, and what pins it is `unit.factor_score`, which compares the
  summed score with the closed-form joint log density of the realised stretch
  and holds to `1e-10` either side. That absence of a golden fixture is how this
  reached a tag: the only thing exercising the filter is that one unit test, and
  a `*-score` fixture would put it in front of every CI job instead.

  Whether the drift crossed zero depended on which BLAS rounded which way. It
  passed on two toolchains and failed on a third, which is why it survived to a
  release build rather than being caught where it was written.

- **The `.sha256` beside each package can be read by `sha256sum -c`.** CPack
  writes its `<hash>  <name>` line through a text-mode stream, so on Windows it
  ended CRLF and `sha256sum -c` took the carriage return as part of the file
  name: *"No such file or directory"*, then `FAILED`, which reads like a corrupt
  download rather than a formatting detail. The archives were never affected.
  It reached the releases rather than only a local build, the Windows packages
  being built on a Windows runner while the Linux ones beside them are LF.

  `CPACK_PACKAGE_CHECKSUM` is off and `cmake/normalise_checksums.cmake` writes
  the file instead. Correcting CPack's own afterwards is not available: the
  post-build hook runs after the package is written and *before* its checksum
  is, and `CPACK_PACKAGE_FILES` names the package inside its staging directory
  rather than where it ends up. The format is unchanged — the hash, two spaces
  and the base name — so anything that read the old files reads these.

- **`results.h` described the file layout backwards.** The two `Draws` structs
  told an embedding host that the convention is "draws in rows, for both the
  HDF5 files and R" and to transpose at the boundary. The R half is right and
  the HDF5 half is not: `write_draws()` transposes twice, so in dataspace terms
  every posterior dataset is one row per quantity and one column per draw, which
  is what h5py reports and what the README and `agents/` have always said. A
  host that believed the header would have written files transposed against the
  ones `bayests` reads. Documentation only -- no code path changes, and no
  dataset this project writes was ever in the other orientation.

- **Stale counts in the documentation.** `CITATION.cff` said twenty algorithms
  where `.zenodo.json` said twenty-two; the README and `agents/` said sixteen
  VARs and VECs where the catalogue in the same file lists nine and eight; and
  `results.md` summarised the nineteen scorable algorithms as "every VAR, every
  VEC and every dynamic factor model", which is twenty-one and contradicts the
  paragraph ninety lines below it that correctly excludes the quantile pair.

- **A forecast from a precision that does not move goes through
  `core::covariance_root()` as well.** The six VARs' forecasts --
  `VarNormalWishart`, `VarNormalGamma` and `VarNormalStochvol`, and the three
  `VarTvp*` under `forecast_states = "hold"` or without a covariance block --
  each carried an inline copy of the unguarded route the entry below replaces:
  `eig_sym(solve(u_sigma_inv, I))`, then `Q Λ^½ Q'` at every horizon. A drawn
  precision badly conditioned enough gave it an eigenvalue a rounding error
  below zero, and so a NaN in every variable of that draw. They now take
  `covariance_root(u_sigma_inv)` once per draw, which symmetrises and floors
  that eigenvalue at zero, and the three that had two branches for adding the
  error have one. The held path reads only `u_sigma_inv`, as before, so a
  posterior without `u_omega_inv` or `psi` still forecasts.

  **Draws change by a rounding error**, because `Q Λ^½ Q' e` is now evaluated
  as one root times `e` rather than in whatever order Armadillo chose for the
  four-term product. Over the full fingerprint recording (399 tests, 120
  fixtures) 20 fixtures moved, only in `/posterior/forecast/forecasts` and by
  at most 5.9e-16 relatively: the constant-coefficient VARs, the `*-hold`
  fixtures of the time varying ones, and the VECs whose forecast is the level
  VAR's through `VarNormalWishartSampler::forecast()` -- `VecKlgs2010`, the
  constant VECs and the time varying ones under `hold`. No estimated draw, log
  likelihood or simulated forecast moved. `unit.forecast_states` gains a held
  forecast from the badly conditioned precision of the entry below for all
  six VARs; every one of them fails on the old route.

- **A simulated forecast could turn NaN once a log-volatility drifted far.**
  Under `forecast_states = "simulate"` the error at each horizon was drawn
  through `core::covariance_root()`, which inverted the rebuilt precision
  `Psi' diag(exp(-h)) Psi` with `solve()` and took the square roots of the
  inverse's eigenvalues. Once the simulated log-volatilities had drifted far
  apart that inverse was ill-conditioned: it came back slightly asymmetric
  (Armadillo's `eig_sym(): given matrix is not symmetric`) and with an
  eigenvalue a rounding error below zero, whose square root put a NaN into
  every variable from that horizon on. Seen as one draw in 2000 in 4 of 84
  expanding windows of a four-variable `VarTvpStochvol` (`sv+covar`) after
  2020; `hold` never triggered it.

  The covariance block models now build the root from the factorisation they
  have instead of from the precision it multiplies out to: the covariance is
  `B B'` with `B = Psi⁻¹ diag(variances)^½`, a unit triangular solve no
  volatility can make ill-conditioned, and its symmetric root is `U S U'` for
  `B = U S V'`, whose singular values cannot be negative. That is the new
  overload `core::covariance_root(psi, variances)` in
  `src/core/models/forecast_states.h`, called with `exp(h)` by
  `VarNormalStochvol`, `VarTvpStochvol`, `VecNormalStochvol` and
  `VecTvpStochvol`, and with `1 / u_omega_inv` by `VarTvpGamma` and
  `VecTvpGamma`. The one-argument `covariance_root(precision)`, which
  `VecTvpWishart`, `VecTvpGamma` without a covariance block and now
  `FavarNormalWishart` call, symmetrises before `eig_sym()` and sets a
  negative eigenvalue to zero; `FavarNormalWishart`'s private copy of it,
  which did the same with `abs()`, is gone.

  **Draws change by a rounding error.** It is the same root of the same
  matrix, so the draws are those of the old route wherever it produced a
  number. Over the full fingerprint recording (399 tests, 120 fixtures) 30
  fixtures moved, all of them of the six models above, and in each only
  `/posterior/forecast/forecasts`, by at most 1.4e-15 relatively. Nothing else
  moved: no estimated draw, no log likelihood, no `hold` forecast, and no
  model outside the six. Most fixtures of the six without a covariance block
  did not move either, the diagonal root coming out bit-identical. On the Austrian file that showed the fault the 1999 draws
  the old route got right move by a median 7e-15 and at most 1.6e-6
  relatively -- the accuracy the inverse was losing there -- and the twelve
  NaN values of the remaining draw are finite. `unit.forecast_states` gains
  two cases: the root of a covariance whose log-volatilities start 36 apart
  (the route through the precision finds an eigenvalue of -3.5e-19 there),
  and all four stochastic volatility forecasts from there with steps of
  variance 4 over twelve horizons, which must be finite everywhere; on the old
  route between 46 000 and 85 000 of their 96 000 values were not.

- **`VecNormalGamma` drew its error precisions without the cointegration space
  prior's term.** The prior of Koop, León-González and Strachan (2010) scales
  the loadings by the error covariance, `alpha | beta, Sigma ~ N(0, v⁻¹
  (beta' P⁻¹ beta)⁻¹ ⊗ Sigma)`, so its density is also a factor in Sigma's
  posterior. `VecNormalWishart` has always included it, as the paper's eq. (8).
  `VecNormalGamma` left it out on the grounds that independent gammas had no
  conjugate form for it. Given Psi they do. The term is `rank`
  pseudo-observations `L` with `L L' = v alpha (beta' P⁻¹ beta) alpha'`
  (`core::coint_prior_pseudo_errors()`). They are appended to the errors, so
  each `omega_i` gains `r/2` on its shape and `(Psi L L' Psi')_ii / 2` on its
  rate, and the covariance block `psi`, BVS scoring included, gains the
  matching quadratic.

  **Draws change** for `VecNormalGamma` with any cointegration term — every
  configuration with `rank > 0`, with or without a covariance block or
  selection. The new numbers are the right ones. With one variable a Wishart
  on the precision is a gamma, so `VecNormalWishart` and `VecNormalGamma` with
  matching priors are the same model. `unit.coint_klgs_prior` now finds their
  posterior means of the precision within 0.07% of each other, against 2.6%
  apart with the term left out: the test fails when the term is removed. On
  the fixtures, the posterior mean of the precision of `VecNormalGamma-plain`
  rises by 3.7%. At its weak shrinkage (`v = 0.1`) the added shape dominates.

- **`VecNormalStochvol`'s loadings prior was scaled by a G that moved with the
  volatilities.** `G⁻¹` was the average per-period precision of the current
  draw, recomputed after every volatility draw. That made the prior on
  `alpha` a function of `h` which `h`'s own draw never saw, so the chain was
  not a Gibbs sampler for any prior. The paper allows any fixed, known G
  (p. 228, eq. 7), and G is now fixed for the whole run. It is the new
  optional `/priors/beta/g_inv` (`ConstantCointSpacePrior::g_inv`) when given.
  Otherwise it is the same average, taken once from the starting values.

  **Draws change** for `VecNormalStochvol` with `rank > 0`, in every
  configuration. The fixtures move by the difference between a G re-averaged
  every draw and one fixed at the start.

  Both verified with the full fingerprint recording: 15 of 118 fixtures moved,
  the eight `VecNormalGamma` and seven `VecNormalStochvol` ones, and nothing
  else. `VecNormalWishart` and `VecKlgs2010` are bit-identical.

- **`VarTvpDiscount` scored only the first horizon; every one after it came back
  as not a number.** `predictive_log_density()` read the rows of
  `/data/forecast/x` as they arrived, and the lagged endogenous blocks of a
  horizon past the first hold a placeholder there -- a forecast overwrites them
  as it simulates, and this recursion does not simulate. The first period was
  scored correctly, the second fed the filter that placeholder, and the state
  never recovered. It now builds its regressors with
  `core::realised_regressors()`, which is what the eight sampling VARs beside it
  have always done and which fills those blocks from `/data/test/y`.

  **Draws change** for this one entry point of this one model, from
  unusable to correct: the first scored horizon is unchanged and no horizon
  after it had a usable value to change. Nothing else moves -- `estimate()`,
  `forecast()` and `log_likelihood()` do not read the affected code, and
  `VecTvpDiscount` scores through `score_vec_forecast()`, which fills the blocks
  already.

- **The non-centred prior is documented outside `model-file.md` too.** The
  README said nothing about `omega_v`: its `/priors/a`, `/priors/psi` and
  `/priors/u_sigma` rows still gave `shape`/`rate` as the only prior on a random
  walk, and its `/posterior/` row did not name the three datasets a non-centred
  block writes. It now has a paragraph on which four models take the prior and
  what it buys, the prior and output rows, and the Frühwirth-Schnatter and
  Wagner (2010) reference. `algorithms.md` calls its table the full list of
  refusals and was missing the two this prior adds: `omega_v` beside
  `shape`/`rate` in one block, and an `omega_v` that is not finite and greater
  than zero. And in `src/core/inputs.cpp` the doc comment of
  `validate_tvp_block()` had ended up above `validate_state_variance_prior()`,
  inserted between the two, so the new function's documentation began with
  parameters it does not take; it is back where it belongs. Documentation
  only: no code path changes and no draw moves.

- **A NaN or an infinity in the input is refused, by name, before anything
  reads it.** `bayests check` accepted every such file. A run then failed deep
  in the numerics -- `inv_sympd(): matrix is singular or not positive definite`,
  `randg(): incorrect distribution parameters` -- naming neither the dataset nor
  the value, and in two places it did not fail at all: a NaN in
  `/data/forecast/x` came back as NaN forecasts, and one in `/data/test/y` as a
  NaN `/posterior/forecast/loglik`, both from runs that exited 0.

  Every input's `validate()` now checks the observations -- the training
  sample, the forecast regressors and the realised values -- and the three shape
  checks every prior and starting value passes through check the values too, so
  `bayests check` refuses the file a run would. Forecasting and scoring run from
  draws a previous stage wrote, without `validate()`, so they check
  `/data/forecast/x` and `/data/test/y` again themselves. The message names the
  dataset: *the realised observations /data/test/y must be finite, but 1 of its
  12 values is NaN or infinite; missing values are not supported*. The test
  reads the exponent bits rather than calling `std::isfinite()`, which a host
  compiling with `-ffast-math` may fold to true.

  Two readers in the I/O layer had the same hole. A count stored as a double --
  `/priors/u_sigma/df` as R writes it -- passed the whole-number test when it
  was NaN, every comparison with NaN being false, and was then cast to `int`,
  which is undefined behaviour; a finite value past the range of an `int` did
  the same. Both are refused now. And selection positions were handed to HDF5
  to convert to integers, which turned a NaN into 0 -- reported as "a position
  below 1" -- and truncated 2.5 to 2 without a word; they are read as doubles
  now, and a position that is not a whole number is refused.

  **Two things are refused that used to run**, both deliberately: a fractional
  selection position, which was silently truncated, and a non-finite value in
  a cell of `/data/forecast/x` that the forecast overwrites, which was harmless.
  The rule is that a file has no missing values, not that it has none where a
  model happens to look. A host feeding the vendored core values with NA in
  them gets the same refusals.

  `check.nonfinite` puts one NaN into every input dataset each algorithm's
  fixtures carry -- 476 pairs, integer datasets rewritten as doubles -- and
  fails on any that `bayests check` does not refuse by name. *Draws are
  unchanged*: `record_fingerprints.sh` before and after, full suite, reports
  120 fixtures unchanged and 0 moved.

- **A directory walk on Windows no longer goes round a junction cycle.** Given a
  directory, `bayests` walks it for model files, and the walk was documented not
  to follow directory links, "so a link cycle cannot make the walk endless". On
  Windows that did not hold: libstdc++ reports a junction as a plain directory,
  so the walk followed it, and a junction back to a directory it was already
  inside took it round and round until the path grew past what Windows opens.
  In the case this was found on, one model below such a junction was checked
  fourteen times -- how many depends on the length of the path -- after which
  the walk warned, wrongly, that the link's target did not exist, and exited 0.
  Linux, where the link is recognised and not followed, was never affected.

  Junctions are still followed, since one is how a folder of models gets pulled
  in from elsewhere. But each directory is now compared with its ancestors by
  file identity, not by name, and one that leads back to an ancestor is skipped
  with a warning saying so; and a file reached by two routes -- a junction into
  a different branch -- is run once. `cli.refusals` gains the cycle: the model
  below it must be checked exactly once on every platform, with the warning on
  Windows. The command line only; no core file changed and no draw moves.

- **The README's link to the algorithm references pointed at the wrong
  section**, on GitHub as well as on the documentation site. "Citing BayesTS"
  sends a reader to *References* below, and GitHub resolves `#references` to
  the first heading of that name -- the list of OpenMP and OpenBLAS links under
  *Multi-threading*, which is above it. That list is *Further reading* now. The
  documentation site made its heading anchors differently from GitHub, so it
  resolved neither this link nor the one to *With an AI coding assistant*; it
  now makes them GitHub's way (`MARKDOWN_ID_STYLE = GITHUB`, which the Doxygen
  1.9.8 CI installs supports), and it carries `docker/README.md`, `AGENTS.md`,
  `model-file.md` and `results.md` as pages, so the README's four links to them
  resolve there too. With five functions' parameters documented -- one of them,
  `chan_jeliazkov_2009_conditional()`, had only `known` -- and a `\omega` in
  `noncentred_support.h` moved inside its formula, where Doxygen had been
  dropping it as an unknown command, the site builds with no warnings where it
  had fourteen. Comments and configuration only; no draw moves.

- **A run stopped while writing its draws is no longer taken for a finished
  one.** `coefficients` skips a model whose posterior is already there, and
  "there" meant one dataset -- in eleven of the twenty-two models not the last
  one the stage writes, and in the two discounted models the first. A job
  killed, or a machine restarted, between two of those writes left a file
  whose rerun printed *Posterior data already exists in file. Skipping
  simulation.*, exited 0, and never wrote what was missing; `forecasts` then
  failed on it, every time, until `/posterior` was deleted by hand.

  The stage now marks `/posterior` with the attribute `coefficients` --
  `"writing"` before its first write and `"complete"` after its last, flushing
  the file each time so the marks reach the disk in that order. A file still
  marked `"writing"` is estimated again, with a line saying why; every reader
  of draws refuses one, so `forecasts` and `loglik` do not run on half a
  posterior; and `bayests check` says it will be estimated again rather than
  skipped. A file with no mark was written before it existed and is treated as
  it always was. The two discounted models' `loglik` is unaffected: it
  recomputes the closed form from the data and reads nothing stored.

  `cli.interrupted` checks all four behaviours for every algorithm. What a kill
  leaves is set directly -- where it lands between writes is a race. The
  change is to the command line and the I/O layer; no core file is touched.
  *Draws are unchanged*: fingerprints against the previous recording, full
  suite, 120 fixtures unchanged and 0 moved.

- **The source package carries `.github/highfive-version`.** The CPack source
  ignore list dropped all of `.github/`, and with it the one file there a
  build needs: the HighFive tag the README's Linux instructions clone
  (`git clone --branch "$(cat .github/highfive-version)"`), that
  `docker/Dockerfile` copies into the CI image, and that the configure-time
  check on HighFive's major version names. Someone building from
  `BayesTS-<version>-src` could follow none of them. The rest of `.github/` --
  workflows, the composite action, the issue and pull request templates and
  `dependabot.yml` -- is still left out. Packaging only; nothing is compiled
  differently and no draw can move.

## 0.2.0 — 2026-09-15

### Added

- **`VecNormalStochvol` and `VecTvpStochvol` write `/posterior/u_sigma_inv/sigma`
  too**, `k` per draw and `h_sigma` on their draws structs, so every stochastic
  volatility model's posterior now carries the variance of its log-volatility
  innovations. A VEC's forecast still holds its volatility at the last in-sample
  period and does not read it. **Draws are unchanged**: recorded before and
  after on the same build, the 14 stochastic volatility VEC fixtures moved only
  in that dataset, which went from `absent` to written, and the other 84 did not
  move at all.

- **`/posterior/u_sigma_inv/sigma`, the variance of the log-volatility
  innovations**, `k` per draw, written by `VarTvpStochvol` and
  `VarNormalStochvol` and kept in `h_sigma` on their draws structs, and by
  `DfmNormalStochvol` and `DfmTvpStochvol` as `u_h_sigma`, beside
  **`/posterior/v_sigma_inv/sigma`**, `n_factors` per draw, as `v_h_sigma`. The
  chain always drew them and threw them away; simulating the volatility forward
  steps by them. A file whose coefficients were drawn before it was stored makes
  `forecasts` exit 1 under `simulate`, naming the dataset and saying to delete
  `/posterior` and run again or to set `forecast_states` to `hold`. Storing it
  draws nothing, so the chain's draws are unchanged: see the verification above.

- **`/model/thin`, which keeps one draw in `thin` after the burn-in.** Every
  result is sized by the draws kept, and a time-varying model's coefficient path
  alone is `nparams * tt` numbers per draw. So the only way to run a slowly
  mixing chain longer used to be a file that grows with it. With `thin`, the
  chain runs `burnin + iterations * thin` draws and keeps `iterations`: the last
  of each block of `thin`, so the chain ends on a kept draw. `bayests check`
  prints the thinning, and a `thin` below 1, or a chain too long to count in an
  `int`, is refused before it starts.

  The `start`, `end` and `thin` attributes on `/posterior` datasets now say what
  was kept: `thin`, `iterations * thin` and `thin`, in iterations after the
  burn-in. Without thinning they are the 1, `iterations` and 1 they always were.

  **This is a core change.** `VarSpec` gains `thin`, `keeps()` and
  `kept_index()`, and all twenty samplers keep their draws through the two calls
  instead of testing `draw >= burnin` themselves. `thin` defaults to 1, so a
  vendoring package compiles unchanged and draws unchanged. It has to propagate
  the change to offer thinning at all.

  **Draws are unchanged** for every file without `thin`. At `thin = 1`,
  `keeps(draw)` is `draw >= burnin` and `kept_index(draw)` is `draw - burnin`, so
  the generator is consumed in the same order. Verified with
  `record_fingerprints.sh` before and after, on the same machine and build
  configuration: all 91 fixtures unchanged, none moved. The new `unit.thin`
  asserts the identity that makes thinning safe. A chain with `thin = t` equals,
  column for column, every `t`-th draw of the unthinned chain run from the same
  seed with `t` times the kept draws, for `VarNormalAld` and `VarTvpAld`. It also
  checks the arithmetic at `thin = 1` and both refusals. `agents.recipes` gains a
  `thin` scenario for the shapes, the `check` line and the `mcmc` attributes. All
  292 tests pass.

- **`/model/seed`, which makes a model's draws a property of its file.** A run
  was only as reproducible as the generator's state when the model's turn came.
  A command naming one model started from Armadillo's default state and so
  repeated. In a walk over a directory or `--all-groups`, though, each model
  started wherever the one before it had left the generator, so its draws
  depended on what else was in the directory. And a forecast run on its own
  drew different shocks from one run inside `posterior`.

  With a non-negative whole number in `/model/seed`, the command line seeds
  each stage just before running it. The chain starts from the seed itself, and
  the log likelihood and the forecast from streams derived from it by
  splitmix64. `posterior` therefore draws exactly what `coefficients`, `loglik`
  and `forecasts` draw run one after another, and a model draws the same in a
  walk as alone. A seed stored as a float is accepted when it is whole, since R
  writes `20260901` as a double. A negative, fractional or non-numeric one is
  refused before the chain, by the run and by `bayests check` alike; `check`
  prints the seed it read.

  The seed is read in `src/io/hdf5/` (`read_model_seed()`) and applied in the
  command line (`src/model_seed.cpp`). Nothing under `src/core/` or
  `include/bayests/` changes: seeding is the host's business, and an embedded
  host keeps seeding its own generator, R through `set.seed()`. There is
  nothing for the vendoring packages to propagate.

  **Draws are unchanged** for every file without a seed, which is every file
  written before this: the generator is not touched unless the attribute is
  there. The golden harness calls the model entry points directly and seeds
  them itself, so no fingerprint can move. Checked end to end as well: the two
  unseeded model files of the JSS illustration, a `VarNormalGamma` and a
  `VarTvpStochvol` with 20,000 draws each, were run through `bayests posterior`
  built before and after, and every `/posterior` dataset is bit-identical. All
  291 registered tests pass.
  `agents.recipes` gains a `seed` scenario, which checks that a seeded file gives
  identical `/posterior` datasets across two runs, when its stages run
  separately, and for two copies in one directory walk. It also checks that a
  different seed gives different draws and that a negative one exits 1 naming
  `/model/seed`.

- **`bayests check`, which says whether a run would accept a model file and how
  it read it, without running anything.** Most fields are read through a
  default, so a misspelled attribute or the wrong `error` spelling is not an
  error but a different model, and a run that exits 0 cannot tell you which.
  `check` opens the file read-only and passes each model through the reader and
  `validate()` its run would use. It adds the checks a forecast makes on its
  regressors, which otherwise come only after the chain has run. It then prints
  the dimensions and switches the file resolved to, and warns about every
  dataset the model never opened and every `/model` attribute no model reads.
  It takes the same `--group` and `--all-groups` flags as the other commands.
  Exit 0 means every model would be accepted, warnings or not; 1 means a model
  would be refused, with the reason on stderr.

  Two refusals are ones a run makes only at the forecast stage, after the
  chain: missing or too few forecast regressors, and a VAR whose
  `/data/train/z` is a different width from what its dimensions make.

  Each front-end implements `BaseModel::check()` by handing its own reader to a
  template in `src/models/model_check.h`, so the check cannot drift from the
  readers. A `check.*` test beside the `golden.*` test of each single-model fixture
  fails if the check refuses a file a run accepts. `agents.recipes` requires every documentation
  example to pass it without warnings, and requires it to refuse the files a run
  refuses. That turned up one more documentation mistake: the time-varying
  example left the VAR's `/initial/u_sigma_inv` in place, and `VarTvpGamma`
  never reads it.

  No sampler is touched, so draws are unchanged by construction.

- **Documentation for coding agents, in `agents/`.** The file format is the
  whole interface and fails quietly, so an assistant working from the README
  alone writes files that run and mean something else. `agents/` has a
  vendor-neutral `AGENTS.md`, and a skill covering the model file, the twenty
  algorithms, the run order, the results, and complete VAR, VEC and factor model
  examples.

  It reaches users three ways. The repository is a Claude Code plugin
  marketplace (`/plugin marketplace add franzmohr/BayesTS`). The documentation
  site serves the same files, indexed by `llms.txt`. An installed package
  carries them under `share/doc/BayesTS/agents/`.

  The new test `agents.recipes` runs every Python example against the built
  binary and checks the shapes the text states. It is registered when CMake
  finds a Python with h5py (`BAYESTS_TEST_AGENT_DOCS`, default `AUTO`), and the
  Linux CI jobs require it. Writing it turned up these mistakes, each now fixed:
  - A factor model's `/posterior/lambda/coeffs` holds the whole loading matrix
    per draw, not only the free loadings.
  - The forecast regressors only need the first horizon's lag block *for one
    lag*. In general, lag `j` of horizon `i` is read while `j > i`.
  - The VEC example inherited an unrestricted intercept from the VAR, which
    contradicted the regressor width it stated.
  - The VEC example flattened `beta` row by row where `vec(beta)` is column by
    column, which is wrong from rank 2 up.
  - Two results examples read a file that was already closed or never opened.

  No sampler is touched, so draws are unchanged by construction.

- **An informative marginal prior on a time-varying cointegration space.** The
  three time-varying VECs -- `VecTvpWishart`, `VecTvpGamma` and `VecTvpStochvol`
  -- read an optional `/priors/beta/p_tau`, `TvpCointSpacePrior::p_tau`, and use
  it as the transition of the state equation with rho taken out:

      beta_t = rho (I_r kron P_tau) beta_{t-1} + eta_t,   eta_t ~ N(0, I).

  This is eq. 12 of the working paper version of Koop, Leon-Gonzalez and
  Strachan (2011). With P_tau = H H' + tau H_perp H_perp' the part of beta along
  sp(H) keeps rho and the part off it decays at rho tau, so the mode of the
  marginal distribution of the space is sp(H) at every t while the space at t
  stays centred between the space at t - 1 and sp(H). The published version of
  the paper, and everything here until now, is the identity: a uniform marginal
  prior. P_tau goes into the transition and not the innovation variance, which
  stays the identity and keeps pinning beta's scale against alpha's.

  `p_tau` is `k_beta` square and has to be symmetric with eigenvalues in
  [0, 1] -- a tau per direction is accepted, one along H and below one off it
  being the paper's case. The prior on the state before the sample stays
  `initial_state`, supplied by the file; the paper's is the stationary
  distribution the transition implies, N(0, I_r kron P_tau* / (1 - rho^2)) with
  tau* = (1 - rho^2) / (1 - rho^2 tau^2), and that is what a host writing the
  file should put there.

  The transition enters in three places, all changed: the smoother's transition
  and the mean of the first state, rho P beta_0; the posterior of the state
  before the sample, whose precision picks up rho^2 P'P and whose mean rho P'
  beta_1; and the draw of rho, which regresses beta_t on P beta_{t-1}. The last
  is still an exact truncated normal.

  **Draws are unchanged** for every file without `p_tau`, which is every file
  written before this. Verified with the fingerprint comparison over the whole
  suite: all 88 existing fixtures print their previous fingerprints digit for
  digit. Under P = I the new arithmetic is the old one to the bit -- products
  with an identity and sums formed in the same order -- and
  `unit.vec_tvp_coint` asserts it directly: a `VecTvpWishart` chain with rho
  drawn and `p_tau` set to the identity equals the chain without it, exactly,
  from the same seed. The same test checks that an informative P_tau changes the
  chain and that `validate()` refuses a P_tau of the wrong size, an asymmetric
  one, and one with an eigenvalue outside [0, 1]. Three new fixtures --
  `VecTvpWishart-rho-ptau`, `VecTvpGamma-rho-ptau` and `VecTvpStochvol-rho-ptau`
  -- run P_tau with rho drawn, the one configuration that goes through all three
  places.

### Changed

- **`bayests` runs one thread unless `OMP_NUM_THREADS` says otherwise.** It
  used to take OpenMP's default, one thread per logical core, and hand the same
  count to OpenBLAS. On an 8-core, 16-thread Ryzen 7 7700 that made a
  264-coefficient `VarNormalGamma` 9 times slower than one thread (143 s against
  16.5 s) and a 588-coefficient one 1.8 times slower, the BLAS threads
  contending for the physical cores. The one small model timed, the
  `VarTvpStochvol-plain` fixture, was 8% faster threaded.
  `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` work as before, so a run that sets
  either is unaffected. The README now recommends running models side by side,
  one process each, which on the same machine took eight `VarTvpStochvol` runs
  from 33 s to 6 s. `cli.refusals` checks the one-thread default.

  *Draws are unchanged* for any run that sets both variables, which covers every
  test and fingerprint recording; no sampler is touched. A run that set neither
  used to draw with a multi-threaded BLAS, whose draws were not reproducible from
  one run to the next; it now gets the single-threaded chain, the one the
  fingerprints record. Hosts embedding the core are unaffected: this is the
  command line's `main()`, which is not vendored.

- **VEC forecasts simulate their states forward too.** `VecTvpWishart`,
  `VecTvpGamma`, `VecTvpStochvol` and `VecNormalStochvol` read
  `/model/forecast_states` like the VARs and factor models. Under `simulate`, the
  default, each draw's loadings and short-run coefficients take a step of their
  random walk per horizon (a BVS-excluded one staying at zero), the cointegration
  vectors a step of their state equation `beta = rho (I_r kron P_tau) beta + eta`,
  `eta ~ N(0, I)` -- with the chain's `rho` where it was drawn and the prior's
  otherwise -- and Psi and the log-volatilities steps of theirs. A VEC forecast is
  its level VAR's, and the level coefficients are not linear in those states
  (`A_1 = A_0 + alpha beta' + Gamma_1`), so the level VAR is rebuilt from the
  stepped states at every horizon rather than converted once;
  `core::simulate_vec_forecast()` in `core/models/vec_support.h` does that for all
  four. `/data/forecast/x` stays in the level layout. `hold` is the old forecast,
  converted once and simulated by `VarNormalWishartSampler`, exactly as before.
  `VecKlgs2010`, `VecNormalWishart` and `VecNormalGamma` have nothing that drifts.

  A host calling `forecast()` directly has to hand over, besides the last-period
  `a` and `beta`, `a_sigma`, `a_lambda`, `rho` where drawn, and as the model has
  them the last-period `psi`, `psi_sigma`, `psi_lambda`, `u_omega_inv` and
  `h_sigma` -- or set `hold`.

  **Draws change**, in `/posterior/forecast` only, for those four models with
  `h > 0` under `simulate`. Verified with the fingerprint comparison on one build:
  with the default temporarily `hold`, no VEC fixture moved against the
  recordings taken before the change; switching to `simulate` moved exactly the
  23 forecasting fixtures of the four models (beside the 20 VAR and factor model
  ones the default already moved), in the forecast alone, and none of the four
  new VEC `-hold` rows. `unit.forecast_states` checks the simulated VEC forecast
  against closed-form moments: the variance a drifting cointegration vector and a
  drifting loading add at the first horizon, the mean `rho` and a drawn `rho`
  imply, and the volatility accumulated by a VEC with no cointegration.

- **Time-varying VAR and factor model forecasts simulate their states forward.**
  `VarTvpWishart`, `VarTvpGamma`, `VarTvpStochvol` and `VarNormalStochvol` used
  to forecast from each draw's coefficients, Psi and volatilities at the last
  sample period, held for all `h` horizons, and `DfmNormalStochvol`,
  `DfmTvpGamma` and `DfmTvpStochvol` did the same with their loadings,
  transition and two volatilities. That is the forecast of a model whose
  drift stops where the sample does, not of the model estimated, and it is wrong
  in more than width:
  - its intervals left out the drift;
  - a held log-volatility understated the expected variance at every step past
    the first, since `E[exp(h_{T+i})] = exp(h_T + i sigma / 2)`;
  - past `h = 1` the point forecast moved as well, because the mean of a
    product of drifting coefficient matrices is not the product of held ones.

  Each draw's random walks now take one step per horizon, before the observation
  that step generates, by the innovation variances the chain drew for them. A
  coefficient or a Psi element BVS excluded stays at zero, and only a factor
  model's free loadings move -- the identifying block does not, in the forecast
  as in the sample. The VEC forecasts do the same, as the entry above describes.
  `DfmNormalGamma` and `FavarNormalWishart` have nothing that drifts.

  `/model/forecast_states` chooses between the two: `simulate`, the default, or
  `hold`, which reproduces the old forecast exactly. The attribute is new, so
  every existing file now reads as `simulate`. A host calling `forecast()`
  directly gets `simulate` too from `VarSpec::forecast_states`. It then has to
  hand over the draws each sampler's header lists — `a_sigma`, `a_lambda`,
  `psi`, `psi_sigma`, `psi_lambda`, `u_omega_inv`, `h_sigma`, and for a factor
  model `lambda_sigma`, `u_h_sigma` and `v_h_sigma`, as the model has them — or
  set `hold`, which reads what it always read. `bayests check` prints
  which of the two a forecast will use.

  **Draws change**, in `/posterior/forecast` only, and only for those seven
  models with `h > 0` under `simulate`. Verified with the fingerprint comparison
  on one build, in two steps, first for the VARs and then for the factor models:
  - With the default temporarily `hold`, every existing fixture was unchanged
    against the recording taken before the change, but for the new
    `u_sigma_inv/sigma` line of the two stochastic volatility factor models,
    which went from `absent` to written. So the held forecast is the old one
    digit for digit, and storing the variances moves nothing.
  - Flipping the default to `simulate` then moved exactly the 20 forecasting
    fixtures of the seven models, in `/posterior/forecast` alone. The seven new
    `-hold` rows and every `-nofcst` row stayed put.

  `unit.forecast_states` checks the closed-form moments of the simulated
  forecast to 5 percent: `i s` for a drifting intercept, `1 + i s` for a
  drifting Psi element, `exp(i s / 2)` for a drifting log-volatility, and no
  spread from a coefficient BVS excluded; for the factor models, `exp(i s / 2)`
  through either volatility, `i s` for a drifting free loading with the
  identifying one unmoved, and `1 + 2 s` for a drifting transition at the
  second horizon. On the generator's fixtures run to
  2000 draws, the forecast standard deviation of the first variable at `h = 4`
  grows by 3 to 12 percent:

  | Fixture | Held | Simulated |
  | --- | --- | --- |
  | `VarNormalStochvol-plain` | 1.64 | 1.70 |
  | `VarTvpGamma-bvs-covar` | 1.63 | 1.78 |
  | `VarTvpWishart-plain` | 1.96 | 2.18 |
  | `VarTvpStochvol-covar` | 2.19 | 2.45 |

- **The VAR forecasts factorise the error covariance once per draw rather than
  once per horizon.** `VarNormalWishart`, `VarNormalGamma`, `VarNormalStochvol`,
  `VarTvpWishart`, `VarTvpGamma` and `VarTvpStochvol` redid the inverse and the
  eigendecomposition of the same precision at every horizon of every draw, and
  every VEC forecast runs through one of them. The factorisation draws nothing,
  and the product that scales the shocks is unchanged, so draws are unchanged:
  the fingerprint comparison over all 91 fixtures shows none moved.

- **The documentation says what a time-varying forecast holds fixed.** Every
  `Tvp` and `Stochvol` model forecasts from the coefficients and volatilities of
  the last sample period, held for all `h` horizons rather than simulated
  forward, so its intervals leave out the drift the model allows over the
  horizon. The README said so only for the two stochastic volatility factor
  models, and `agents/` not at all. `results.md` now has a section on it, and
  the README's command table states it. Documentation only.

- **The time-varying models hold their Psi path and error precisions one block
  per period.** All four time-varying models with a covariance block kept Psi as
  a `(k tt)` square block diagonal. `VarTvpStochvol` and `VecTvpStochvol` also
  kept the error precision and the volatilities that way, formed
  `Psi' Omega Psi` over the whole diagonal as a dense product on every draw, and
  scored their BVS candidates against a `(k tt)` square quadratic form. At
  `k = 6`, `tt = 500` each such matrix is 72 MB of mostly zeros, and the product
  is of order `(k tt)^3` a draw. All three are now `k x k` blocks stacked per
  period, `stacked_identity()` in `model_support.h`, the layout `VarTvpGamma`
  already used for its precision. The product is `tt` small ones, and the
  quadratic forms are sums over periods.

  *Draws are unchanged* for `VarTvpGamma` and `VecTvpGamma`: their fingerprints
  are byte-identical, since only where Psi is stored moved. *Draws change by a
  rounding error* for `VarTvpStochvol` and `VecTvpStochvol` with a covariance
  block, where the product is now summed block by block. The worst relative
  difference over their four fixtures is 6.7e-13. Without a covariance block they
  are byte-identical.

- **The README says `seed` fixes the draws on a single thread only.** The table
  of `/model` attributes said a seed "fixes the run's draws". The program uses
  every core unless told otherwise, and the samplers are only reproducible with
  `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. The agent documentation
  already said so.

- **A release carries every package, not only the source.** `release.yml` used
  to create a new draft release with the packages attached. A release made in the
  web page for the same tag stayed empty beside it. The workflow now uploads to
  the tag's release when one exists, and drafts one only when none does. It also
  attaches a `.sha256` for every file, which the upload used to leave behind. Two
  packages are new:
  - a `.deb`, built on Ubuntu 24.04 against the distribution's libraries. It is
    installed and run in a clean container before it is accepted. Build one
    yourself with `-DBAYESTS_PACKAGE_DEB=ON`, which is off by default because a
    vcpkg build would produce a package with the wrong dependencies.
  - the snap, built by `snap.yml`, which `release.yml` now calls.

  A manual run can attach the packages to a tag's existing release. No sampler is
  touched, so draws are unchanged by construction.

- **The Windows installer upgrades in place, under one start menu folder.** It
  used to propose `%ProgramFiles%\BayesTS` even when an earlier version was
  installed elsewhere, because it removed that version — and the registry entry
  recording where it was — before choosing a directory. It now reads that entry
  first and proposes the same folder. `/D=` still overrides it. The start menu
  folder is `BayesTS` rather than `BayesTS 0.1.0`, so a release no longer adds
  a folder of its own. The 0.1.0 installer recorded its folder under the
  versioned name, and an upgrade from it offers that name again only if you
  decline removing 0.1.0 first. No sampler is touched, so draws are unchanged
  by construction.

- **A path that does not exist exits 2, not 1.** Exit 1 means a run started and
  something in it failed, and 2 that the command line could not be acted on. A
  missing path started nothing: no file was opened and no walk begun. It belongs
  with a `--group` that cannot name a group, and now exits the same way, still
  printing "Path does not exist". A script that treated 1 as "check the path"
  needs updating. A path that exists but is not an HDF5 file still exits 1, as
  does any failure inside a walk. No sampler is touched, so draws are unchanged
  by construction.

- **The out-of-sample regressors are the compact layout.** `ForecastData::x` is
  `h` rows by `VarSpec::n_x()` columns, one period per row, where
  `ForecastData::z` was `h * k` rows by `k * n_x` columns — the same numbers
  kroneckered up with `I_k`. `/data/forecast/x` is the dataset that carries it.

  A forecast applies one `k x n_x` coefficient matrix to one regressor column
  per period, so the SUR spelling held no information the compact one does not,
  at `k^2` the memory and `k` times the multiplications — all of the extra ones
  against a structural zero. The six samplers that iterate a path now reshape a
  draw's coefficients once per draw instead of multiplying a wide `z` once per
  horizon. It is `k` that decides how much that is worth: for a three-variable
  VAR nothing anyone would measure, for the 174-variable global VAR that
  prompted it, half a gigabyte of regressors down to 17 KB over twelve periods.

  `VarNormalGamma` and `VarTvpGamma` gained the width check the other four
  forecasts already had. `arma::reshape()` zero-pads or truncates instead of
  throwing, so regressors that do not divide into `k` rows would have produced a
  plausible path from the wrong coefficients where the SUR product used to fail
  on the mismatch.

  *Draws change by a rounding error.* The arithmetic is the same sum with the
  structural zeros left out, but BLAS blocks a 184-term reduction differently
  from a 32,016-term one. All 88 fixtures were compared with
  `record_fingerprints.sh`: 52 moved, every one of them in `/posterior/forecast`
  and nowhere else, with a worst relative difference of 9.4e-16 — about four
  ulps. Coefficients, log likelihood, precisions and `beta` are bit-identical
  throughout.

  **Model files written before this still forecast.** A file carrying
  `/data/forecast/z` and no `/data/forecast/x` is compacted on the way in by
  `read_forecast_regressors()`, which is exact — it subscripts the kron rather
  than averaging it — and refuses a `z` whose dimensions are not multiples of
  `k` rather than inventing regressors from one. `unit.forecast_regressors_io`
  covers both spellings, the precedence when a file has both, and that refusal.

### Fixed

- **A time-varying coefficient path leaves its starting values.** Every
  time-varying block drew its path with the simulation smoother centred on the
  previous draw of the state before the sample, `a0`, with the random walk's own
  innovation variance as the prior covariance of the first period, then drew
  that variance, then `a0` given the path. Each step is a valid conditional, but
  together they tie the first period and `a0` to each other with the innovation
  variance, and a path meant to move slowly has a small one. At a prior rate of
  1e-12 the chain could not move the level of a path at all: two chains of
  `VarTvpGamma` started at 0 and at 5 ended exactly 5 apart, and what the
  samplers reported as the posterior of the coefficients was the starting path.
  A time-varying VEC kept its loadings where the host had initialised them while
  its cointegration vectors moved, which in the global VEC models of a downstream
  package left sub-models without error correction and the solved models
  explosive.

  The path is now drawn with `a0` integrated out of the first period's prior,
  `N(mu_0, V_0 + Sigma)`, and `a0` is drawn from its conditional on that path
  before the innovation variance, which conditions on it. The order is part of
  the fix: this is a partially collapsed Gibbs sampler, which preserves the
  posterior only in that order. The smoother itself is unchanged.
  `initial_state_variance()` in `src/core/models/model_support.h` sets it out.
  It covers the coefficient and covariance blocks of `VarTvpAld`, `VarTvpGamma`,
  `VarTvpStochvol`, `VarTvpWishart`, `VecTvpGamma`, `VecTvpStochvol` and
  `VecTvpWishart`, and the loadings and transition of `DfmTvpGamma` and
  `DfmTvpStochvol` through `draw_random_walk_state()`, which now draws the state
  before the sample first. The cointegration block, whose innovation variance is
  fixed at the identity, and the log-volatilities are not changed.

  **The prior precision of the state before the sample must now be positive
  definite** for every time-varying block (`/priors/a`, `/priors/psi`,
  `/priors/lambda`), since the draw takes its inverse. `validate()` refuses a
  zero, singular or indefinite one. `agents/` says so.

  **Draws change** for every time-varying model and for nothing else. Verified
  with `record_fingerprints.sh` before and after on the same machine and build:
  43 of 91 fixtures moved, every one of them a `VarTvp*` (18), `VecTvp*` (21) or
  `DfmTvp*` (4) fixture, and all 48 fixtures of constant-coefficient models are
  unchanged. The new numbers are the right ones because the old ones did not
  depend on the data where the innovation variance was small: the new
  `unit.tvp_initial_state` runs `VarTvpGamma` with its covariance block and
  `VecTvpWishart` from starts 5 apart and requires the posterior means to agree
  to 0.25 and, for the VAR, to lie within 0.25 of the coefficients the data were
  simulated from. Against the previous core it fails all seven of its checks
  that concern the draw or the refusal.

  **Downstream:** the vendored core in bvartools needs the refresh, and the
  DFM subset vendored by dfmtools does too.

- **A boolean attribute written from R is read as what it holds.** HDF5 has no
  boolean type. h5py and HighFive store one as an enumeration with the members
  FALSE and TRUE, while R's hdf5r stores a logical as an enumeration over an
  unsigned byte with a third member, NA. Read through HighFive's conversion to
  `bool`, the R enumeration came back false whatever it held, so
  `/model/structural = TRUE` written by bvartools reached the samplers as false.
  - A structural VAR ran with its contemporaneous columns as ordinary
    regressors, and `bayests check` reported `structural: no`.
  - A structural VEC was refused, because its `z` had `k(k-1)/2` columns more
    than its dimensions describe.

  `get_attribute_bool()` now reads an enumeration in its own type and decides by
  the name of its member, reads an integer as non-zero, and refuses R's NA.
  `unit.bool_attributes` covers HighFive's encoding, R's (NA included) and
  integers.

  *Draws are unchanged* for every file whose booleans were written by h5py or
  HighFive: the fingerprint comparison over the full suite (91 fixtures, 1,564
  fingerprints) found none moved. *Draws change* for structural models written
  from R, which now run as structural.
  - Structural VARs: the coefficient and error precision draws are those of the
    same regression either way, since the flag only splits `A_0` off for the
    forecast. So the forecasts change, and the chains do not.
  - Structural VECs: they run at all.

- **The constant VECs sample the posterior of their cointegration space prior
  when the cointegration term has more rows than the model has equations.**
  `VecNormalWishart`, `VecNormalGamma`, `VecNormalStochvol` and `VecKlgs2010`
  use the collapsed Gibbs sampler of Koop, León-González and Strachan (2010):
  draw `alpha` given `beta`, change to `A = alpha (alpha' alpha)^{-1/2}` and
  `B = beta (alpha' alpha)^{1/2}`, draw `B` given `A` from a normal. The paper
  derives that normal for `alpha` and `beta` with the same number of rows `n`
  (Proposition 1 and its proof), and its Section 4 notes that the algorithm does
  not apply when the dimensions differ. With deterministic terms restricted to
  the cointegration space or unmodelled variables in it, `k_beta > k`: the
  exponents of the MACG density of `beta` and of the normaliser of
  `alpha | beta` no longer cancel, nor do the two polar Jacobians, and the prior
  in `(A, B)` is the normal kernel times `|B' P_tau^{-1} B|^{-(k_beta - k)/2}`.
  Drawing from the normal alone overstated `|Pi|`. In a simulation-based
  calibration of `VecNormalWishart` with `k = 2`, rank one, `T = 40`,
  `v_inv = 4` and 1000 replications, the true `|Pi|` had a mean rank of 0.39
  among the posterior draws with a restricted constant, 0.31 with a restricted
  constant and trend, and 0.42 with one unmodelled random walk, against 0.50
  (chi-square p from 1e-16 to 1e-101); with `k_beta = k` it was calibrated.

  The draw is now exact. Before every draw of `B` the loadings are given the
  `k_beta - k` rows they lack, drawn from
  `N(0, c^{-1} (beta' Q beta)^{-1} kron gamma^{-1} I)` independently of
  everything else — `c = v` and `Q = P_tau^{-1}` under a proper prior, `c = 1`
  and `Q = I` under the flat one — so that `alpha` and `beta` have the same
  number of rows and Proposition 1 holds as written. `B` is then normal given the
  augmented `A`, the data enter through its first `k` rows, and the auxiliary
  rows are discarded after the draw. This is the new `augment_loadings()` in
  `core/models/vec_support.h`, called by all four samplers. `unit.coint_jacobian`
  compares the posterior means of `Pi` and `|Pi|` from `VecNormalWishart` and
  `VecKlgs2010`, in a one-equation model with a restricted constant, with an
  exact integration on a grid, under a proper and under the flat prior: they
  agree within 3.6 Monte Carlo standard errors, where the posterior without the
  factor is 34 to 164 away. Rerun on the new draw, the calibration above gives
  mean ranks between 0.49 and 0.51 for every statistic in the three
  configurations with `k_beta > k`, with the smallest chi-square p 0.07.

  As first committed, the normal draw was instead a Metropolis–Hastings proposal
  corrected by that factor, in `accept_coint_draw()`. It was right in
  distribution and kept so few proposals on real models that the chains barely
  moved: on the 26 country models of Dees, di Mauro, Pesaran and Smith (2007) as
  the bgvars GVEC vignette sets them up, with `k_beta - k` from four to seven,
  the median model repeated `beta` in 92% of its draws and one never left its
  starting values in 10,000. The new `unit.coint_dees_us` runs `VecNormalWishart`
  and `VecKlgs2010` on the US model of that paper (`k = 6`, `k_beta = 10`, rank
  2, flat priors; the data are in `test/dees2007_us.h`, written by
  `test/make_dees2007_us.R`). `beta` now moves in every draw, where it repeated in
  62% of them; the median effective sample size of the elements of `Pi` is about
  1,000 of 5,000 draws, where it was 682 of 10,000; the two samplers agree within
  1.6 Monte Carlo standard errors; and the paper's Table B11 effects of foreign
  output and inflation lie inside the 90% posterior intervals. What does not
  improve is the slowest element, `Pi`'s column on the restricted trend, which
  mixes no faster per draw than before, with a first-order autocorrelation of
  0.9. Scaling the auxiliary rows by anything from 1e-4 to 1e4 did not change
  that either.

  *Draws change* for every constant VEC with `k_beta > k`: the fingerprint
  comparison over the full suite moves exactly the 19 constant-VEC fixtures, all
  of which restrict a constant to the cointegration space, and leaves the other
  72 unchanged. All 301 tests pass. With `k_beta = k` no random number is drawn
  and the draws are bit-identical, checked through bvartools from the same seed
  for `VecNormalWishart` and `VecNormalGamma`. `VecNormalGamma` still leaves the
  cointegration space prior out of its precision draw, as documented there.

- **The constant VECs no longer stop partway through a chain with
  `sqrtmat_sympd(): transformation failed`.** Every draw is split into a
  semi-orthogonal factor and a scale, `alpha (alpha' alpha)^{-1/2}` and
  `Beta (Beta' Beta)^{1/2}`, and both were computed from the eigendecomposition
  of the cross product. That squares the condition number: a full-rank
  `VecNormalWishart` with BVS on real data (k = 6, rank 6, k_beta = 12) drew a
  loading matrix whose `alpha' alpha` had eigenvalues from 1e-16 to 259, and
  Armadillo refuses any that rounds below zero — one of those models stopped at
  draw 346, another at draw 4680, of 6000. One that rounded just above zero was
  inverted into a factor that was not semi-orthogonal, silently. Both factors
  now come from the thin SVD, `X = U S V'`, as `U V'` and `V S V'`: the same
  matrices, resolved to working precision, in `reparameterise_alpha()` and the
  new `normalise_beta()` in `core/models/vec_support.h`. `VecNormalWishart`,
  which spelt both out inline, now calls them like `VecNormalGamma`,
  `VecNormalStochvol` and `VecKlgs2010`. Both models run to completion.
  *Draws change by a rounding error* in those four models: the fingerprint
  comparison over the full suite moves exactly their 19 fixtures, by at most a
  relative 1.2e-13 (`VecNormalStochvol-structural`), and leaves the other 72
  unchanged.

- **`OPENBLAS_NUM_THREADS` is respected.** The binary set the OpenBLAS thread
  count to the OpenMP one on every start, overwriting the variable, so
  `OPENBLAS_NUM_THREADS=1` on its own still ran BLAS on every core; only
  `OMP_NUM_THREADS` had any effect. The OpenMP count is now used only when
  `OPENBLAS_NUM_THREADS` is unset. A run that sets `OMP_NUM_THREADS`, or both,
  gets the thread counts it got before. This holds for an OpenBLAS built with
  pthreads, as vcpkg's and Ubuntu's default are; one built with OpenMP, as
  MSYS2's is, ignores `OPENBLAS_NUM_THREADS` whatever the binary does. The
  startup line now names the threading model, as in
  `OpenBLAS threads: 1 (pthreads)`. Draws are unchanged: the samplers are not
  touched, and the fingerprint comparison over all 91 fixtures, which pins both
  variables to one, shows none moved. `cli.refusals` checks the count the binary
  reports against a pthreads OpenBLAS and skips the check on other builds. As
  first committed it asserted the count on every OpenBLAS, and failed the
  Windows CI job, which links MSYS2's OpenMP build.

- **Two misordered command lines exit 2 with the right reason.** A flag
  directly after `--group`, as in `--group --all-groups`, was taken as the name
  of a group, looked for in the file, and failed as a run with exit 1. A flag in
  the place of the path, as in `bayests check --all-groups model.h5`, exited 2
  but called the real path "a second path". Both are now refused as the command
  lines they are, before any file is opened, and `cli.refusals` covers both.

- **`--all-groups` finds every model in a file that also has one at its root.**
  The search stopped at the first group holding a model, so a file with a
  model at its root and more under `/models` ran the root model alone, printed
  nothing about the others and exited 0; the same models were found when the
  search started at `/models`. It now skips only a model's own `model`, `data`,
  `priors`, `initial` and `posterior` and searches every other group below it.
  Files whose models all sit beside each other under groups find the same
  models as before. `unit.model_group` covers a root model with models below
  it, and the README and `pipeline.md` state the rule. Draws are unchanged: the
  samplers are not touched, and the fingerprint comparison over all 91
  fixtures shows none moved.

- **`model-file.md` describes the files the stochastic volatility and
  time-varying DFMs read.** Its tables gave `/priors/v_sigma` only the gamma
  layout, though `DfmNormalStochvol` and `DfmTvpStochvol` read a full stochastic
  volatility group there at the width of the factors. They left the shapes of
  `u_h` and `v_h` blank and never named `u_h_init` or `v_h_init`, which both
  models read. They credited the loadings' random walk starting values to
  `DfmTvpGamma` alone, without shapes, though `DfmTvpStochvol` reads them too,
  and gave the loadings only as a flat row where the time-varying DFMs read a
  `(tt, n_lambda)` path. Every row now names the models that read it and the
  shape the fixtures carry, and the FAVAR's loading count is stated beside the
  DFM's. Documentation only.

- **The documentation lists `/posterior/u_scale/coeffs`.** Both `*Ald` models
  write the asymmetric Laplace scale there, `(k, iterations)`, and `loglik`
  reads it back, but neither `results.md` nor the README's list of posterior
  datasets named it. `results.md` now has a row for it, and its
  `u_omega_inv` row says that for the quantile models the `k*tt` draws are the
  precision of the normal mixture, `1 / (tau^2 w_t u_scale)` with
  `tau^2 = 2 / (q (1 - q))`, which moves with
  the latent scales rather than with a volatility. Documentation only.

- **The generated VEC fixtures no longer carry a dataset their model never
  reads, and a `check.*` test now fails on any warning.** The fixture generator
  wrote both `/data/train/z` and `/data/train/x` into every VEC file, so
  `bayests check` warned that the model never reads one of them on 40 of the 91
  fixtures, and nothing failed on it. Each file now carries only the layout its
  model reads, both still built from the same levels. The `check.*` tests fail
  on a `warning:` line, as `agents.recipes` already did. Draws are unchanged:
  the fingerprint comparison over all 91 fixtures shows none moved, and all 300
  tests pass.

- **The agent documentation states the posterior shapes and refusals the
  binary has.** `results.md` gave `/posterior/psi/coeffs` as `k*k` wide, but the
  four time-varying models with a covariance block write one block per period,
  `k*k*tt`. It listed `/posterior/u_omega_inv/coeffs` for the gamma and quantile
  models only, but the stochastic volatility models write it too, at `k*tt`. The
  refusal table in `algorithms.md` now lists the six `validate()` refusals it
  lacked: more factors than series, a factor transition order not below the
  number of periods, a FAVAR with no observed factor, a FAVAR state Wishart
  prior with fewer degrees of freedom than state elements, fewer than two
  periods on a time-varying or stochastic volatility model, and a VEC of
  positive rank with no regressors. Documentation only; no sampler is touched.

- **A flag the command does not know, or a second path, exits 2 rather than
  running something else.** Either used to be a warning and a run with exit code
  0. `bayests check a.h5 --gruop /models/3` checked the model at the root of the
  file, a misspelled `--all-groups` ran one model instead of all of them,
  `bayests check a.h5 b.h5` never looked at `b.h5`, and a step flag such as
  `--no-loglik` on `coefficients` was ignored. Each now prints the reason and
  the usage line and exits 2, the code documented for a command line that
  cannot be acted on. `cli.refusals` runs each case against the built binary.

- **A directory walk that meets an entry it cannot read no longer ends the
  program.** The walk used `std::filesystem::recursive_directory_iterator`,
  which throws on the first entry it cannot read, and nothing caught it. A
  directory holding a junction whose target was gone ended in `std::terminate`,
  with exit code -1073740791 on Windows, no line naming the entry, and every file
  after it unprocessed. The walk now goes directory by directory with error
  codes. An entry that cannot be read is reported and counts as a failure (exit
  1), since it may have held models. A link whose target does not exist is
  skipped with a warning. Either way the walk carries on, and the files it finds
  run in sorted order. A path that does not exist still exits 2. `cli.refusals`
  covers the broken junction on Windows and a dangling symbolic link elsewhere.

- **A time-varying starting path of the wrong length is refused rather than
  padded.** `read_path()` was an `arma::reshape`, which pads a short path with
  zeros and cuts a long one. `validate()` then saw a matrix of exactly the shape
  it asks for, whatever the file held: a `VarTvpGamma` file with 264 of its 288
  starting coefficients passed `bayests check` and a run with exit code 0. The
  element count is now checked first, for `/initial/a`, `psi`, `beta` and
  `lambda` in every time-varying VAR, VEC and DFM, and the message names the
  dataset and both counts. `unit.read_path` covers a path short, long and at the
  wrong width.

- **Forecasting a time-varying model from a file with no training sample is
  refused with a message.** The forecast readers took the last period as
  `tt - 1` on an unsigned `tt`, so a file holding a fitted posterior but no
  `/data/train/y` asked for the largest index there is and failed inside
  Armadillo. The DFM readers guarded against it and the rest did not.
  All of them, the constant-coefficient stochastic volatility readers included,
  now go through `last_sample_period()` in `model_io_common.h`, which says what
  is missing.

- **`bayests check` warns about a covariance-block selection attribute that
  nothing reads.** `/model/priors/psi`'s `varsel` is read only by the four
  time-varying models with a covariance block, and only with the block switched
  on. Anywhere else the file asked for a selection that was never made, and
  `check`, which scans only the attributes of `/model`, said nothing.

  None of these five touches a sampler. *Draws are unchanged*: all 91 fixtures'
  fingerprints are byte-identical before and after, and all 300 tests pass,
  `agents.recipes` and every `check.*` among them.

- **The time-varying gamma models draw their covariance block under the current
  error variances.** `VarTvpGamma` and `VecTvpGamma` inverted the *starting*
  error precision once, before the chain, and drew every psi path under that
  inverse. The precision itself was redrawn each iteration but never reached
  psi, while the selection step on the same block scored its candidates against
  the current one. A comment beside it declined to refresh the inverse because
  doing so "would move the posterior". Each psi draw now uses the diagonal of the
  current error precision, which is what its full conditional is conditioned on.

  *Draws change* for both models with a covariance block (`error = gamma+covar`),
  with or without selection. Of the 91 fixtures these four moved:
  `VarTvpGamma-covar`, `VarTvpGamma-bvs-covar`, `VecTvpGamma-covar` and
  `VecTvpGamma-bvs-covar`. That comparison covers this entry and the next four
  together; each fixture that moved is accounted for by one of them.

- **VarNormalAld's variable selection let an excluded coefficient back in on the
  prior alone.** Its likelihood callback multiplied each candidate by the mask a
  second time, with the indicators the sweep was still updating. The candidate
  that switched an excluded coefficient on was therefore zeroed, and its return
  was decided by the prior inclusion probability without the data. The other 13
  samplers with BVS score candidates against the unmasked regressors, and this
  one now does too.

  `unit.bvs` estimates a median regression with one regressor the data depend on
  and one they do not, under a prior inclusion probability of 0.5. The code this
  replaces included the irrelevant regressor in 78.5% of draws, and the test
  fails against it. The sampler now includes it in 9.3%, and the relevant one in
  all of them.

  *Draws change* for `VarNormalAld` with `varsel = bvs` (`VarNormalAld-bvs`).
  `VarTvpAld` scored against the unmasked regressors already and is unchanged.

- **VarNormalStochvol no longer fails on an observation far out in the tails.**
  It carried its own copy of the ten-component mixture draw, with both faults
  `stochvol_mixture.h` describes. The component probabilities were formed as
  densities, which all underflow to zero for an observation far from every
  component. The indicator index was never clamped, so the NaN row that left
  indexed one past the end of the table. A fixture with a single value of 1e30
  in `y` exited with "Mat::elem(): index out of bounds"; a host that defines
  `ARMA_NO_DEBUG` read past the table instead. The model now calls
  `stochvol_ocsn_2007` and `draw_stochvol_state`, as the other four stochastic
  volatility samplers do. The shared code forms the probabilities in logs, clamps
  the index, and draws each path with a banded Cholesky rather than a dense
  `tt x tt` factorisation. `unit.var_normal_stochvol` runs the outlier and checks
  that the chain finishes with finite precisions.

  *Draws change* for every `VarNormalStochvol` configuration (its six fixtures).
  The model is the same one: the same mixture, the same conditionals, and the
  same order of blocks. The random numbers are consumed differently, and
  indicators that underflowed before are now drawn.

- **Prior values no model can mean are refused, where they used to run.**
  `validate()` checked shapes and not values. A negative gamma rate passed
  `bayests check` and a chain with exit code 0, and produced plausible numbers,
  since the posterior rate stays positive while the data outweigh it. An
  inclusion probability of 1.5 held every coefficient it covered at zero for the
  whole chain, because `log(1 - 1.5)` is a NaN and a NaN excludes. A run and
  `bayests check` now refuse, naming the element:
  - a gamma `shape` or `rate` that is negative or not finite; zero is allowed, as
    an improper prior the sample makes proper;
  - a log-volatility `offset`, SSVS `tau0`/`tau1`, or initial log-volatility
    innovation variance that is not greater than zero;
  - an `inprior` outside `[0, 1]`;
  - an asymmetric normal prior precision or Wishart scale, to the tolerance
    `p_tau_inv` is already held to, for the reason it is;
  - a non-diagonal starting precision that the sampler redraws only the diagonal
    of. That is `u_sigma_inv` of `VarNormalGamma` and `VecNormalGamma`,
    `u_omega_inv` of the time-varying gamma models, and the random walks'
    `a_sigma_inv`, `psi_sigma_inv` and `lambda_sigma_inv`. Its off-diagonal
    elements used to stay in the chain from the first draw to the last.

  `unit.input_values` checks each refusal against an input otherwise accepted.
  `agents/` states the rules.

  *Draws are unchanged*: the other 80 fixtures are byte-identical, and every
  `check.*` test and `agents.recipes` passes, so no file the project writes is
  refused.

- **A VAR forecast refuses regressors without exactly one row per horizon, and
  `bayests check` agrees with it.** Five of the six VAR forecasts never checked
  the height of `/data/forecast/x`. With a row short they read and wrote past its
  end: "Mat::submat(): indices out of bounds" in a checked build, and memory
  corruption under `ARMA_NO_DEBUG`. `VarNormalWishart`'s forecast, which every
  VEC's goes through, required exactly `h` rows. `bayests check` required at
  least `h`, so a file with an extra row passed the check and failed the
  forecast. All six now call `require_forecast_horizons()` in `model_support.h`,
  and the check requires exactly `h`. `unit.input_values` covers a row short and
  a row over.

  *Draws are unchanged* (the same comparison as above).

  bvartools and dfmtools vendor the core, so the changes these five entries make
  under `src/core/` have to reach them as well.

- **BVS draws its inclusion indicators from their posterior.** `bvs_sweep()`
  set an indicator to one when `l1 - l0` exceeded the log of a uniform, which is
  inclusion with probability `min(1, exp(l1 - l0))` rather than
  `exp(l1) / (exp(l0) + exp(l1))`. It also visited a position only with the
  prior probability of the state it was already in, and scored every candidate
  against the mask as it stood before the sweep began. The chain that made did
  not have the posterior as its stationary distribution. At a prior inclusion
  probability of 0.5 and a likelihood ratio of one, a position once included
  stayed in for ever, where the posterior is 0.5. At a prior of 0.1 it spent
  53% of its draws included, where the posterior is 10%. The rule came over
  from bvartools' original `bvs.cpp`.

  Each indicator is now drawn from its full conditional (Korobilis, 2013): every
  selected position in every sweep, in random order, conditioned on the mask as
  it stands at that point, including the positions visited earlier in the same
  sweep. The logistic is taken in logs, so a log likelihood difference in the
  hundreds cannot overflow. `unit.bvs` runs the sweep against a likelihood whose
  posterior over two dependent indicators is known in closed form. Both scopes
  reproduce that four-cell table to within 0.006, and a flat likelihood returns
  its prior.

  *Draws change* for every model configured with `varsel = bvs`: the six VARs,
  the two quantile VARs and the six VECs that offer it, on the coefficients and,
  where there is one, on the covariance block. The indicators, and through the
  mask every block drawn after them, now follow the posterior they are named
  for. Of the 91 fixtures, the 22 that select by BVS moved and the other 69 are
  byte-identical. That comparison covers this entry and the next one together.
  bvartools vendors `core/algorithms/bvs.h` and has to take the same change.

- **BVS over the covariance block ignored the data in the four
  constant-coefficient models that have one.** `VarNormalGamma`,
  `VarNormalStochvol`, `VecNormalGamma` and `VecNormalStochvol` copied the psi
  block's regressors before the chain started, while they were still zero, and
  scored every selection candidate against that copy. Every candidate had the
  same likelihood, so the indicators were a function of the prior alone. The
  psi draw also ignored them and was only masked afterwards. The four
  time-varying models, which keep the copy per draw, were not affected.

  The regressors are now kept once they hold the current draw's errors, and the
  psi draw is made against them masked by the indicators, as the coefficient
  block in the same files already was. In `unit.bvs`, against errors whose
  correlation the data leave no doubt about, a prior inclusion probability of
  0.1 now gives a posterior inclusion of 1.000. Against independent errors a
  prior of 0.5 now gives 0.029. Both models used to return their prior.

  *Draws change* for those four models configured with both `bvs` and a
  covariance block (`gamma+covar` or `sv+covar`).

- **The time-varying models scored every period of their log likelihood under
  the last period's error precision.** `VarTvpStochvol` and `VecTvpStochvol`
  always, and `VarTvpGamma` and `VecTvpGamma` when a covariance block makes the
  precision drift. Their readers handed the log likelihood the precision of
  period `tt` alone, and every observation was scored under it. That is the
  likelihood of a model whose volatility or covariance does not move, not of
  the model that was estimated, so a WAIC or LOO computed from
  `/posterior/loglik` compared the wrong thing. The stochastic volatility DFMs
  and the constant-coefficient stochastic volatility models already scored each
  period under its own precision.

  The readers now pass the whole stored path, and each period is scored under
  its own block. The forecast still starts from the last period, as it should.
  A precision of a height that is neither one matrix nor one per period is
  refused rather than silently truncated by a reshape.

  *Draws are unchanged*; the log likelihood changes. Across the fixtures only
  `/posterior/loglik` moved: substantively in the 14 `VarTvpStochvol` and
  `VecTvpStochvol` fixtures and the four `*TvpGamma` ones with a covariance
  block, by a relative difference of up to 1.2 in a summary statistic.

- **The determinant term of the Gaussian log likelihoods no longer underflows.**
  Thirteen samplers formed `-log(det(Sigma)) / 2` by inverting the precision and
  taking the log of the determinant of the result. That is a product of `k`
  variances, and it reaches zero, and the log minus infinity, once `k` and the
  scale of the data are modest together. It is now `log|Sigma^-1| / 2` through a
  Cholesky in logs, `half_log_det_precision()` in `model_support.h`, which also
  refuses a precision that is not positive definite by name.

  *Draws are unchanged*; the log likelihood changes by a rounding error. In the
  24 fixtures this entry moves on its own, `/posterior/loglik` differs by at most
  2.2e-16 relative. It was measured in the same comparison as the previous
  entry, whose fixtures it also touches.

- **SSVS includes a coefficient far from zero instead of excluding it.**
  `ssvs_sweep()` formed the spike and slab densities at the current draw and
  divided one by their sum. For a coefficient many slab widths from zero both
  are zero in double precision. The inclusion probability was then NaN, the
  draw against it failed, and the coefficient the data most want in was
  excluded, its prior precision set to the spike's. The log odds are now formed
  in logs and put through the same logistic BVS uses,
  `inclusion_probability()` in `core/algorithms/inclusion_probability.h`.
  `unit.ssvs` checks the inclusion frequency against the closed form, and that a
  coefficient at 50 slab widths is included in every sweep.

  The sweep also visited the indicators in a fresh random order. Given the
  coefficients they are independent, so the order changed nothing but which
  uniform went to which position, and cost a permutation per draw. They are now
  drawn in the order listed.

  *Draws change* for the four models configured with `varsel = ssvs`
  (`VarNormalWishart`, `VarNormalGamma`, `VecNormalWishart`, `VecNormalGamma`).
  Where no density underflowed, only the assignment of random numbers to
  positions differs and the posterior is the same. Where one did, the
  coefficient is now included as it should be. Of the 91 fixtures, the five that
  select by SSVS moved and the other 86 are byte-identical.

- **A cointegration prior matrix that is not symmetric is refused.** A constant
  VEC checked only that `/priors/beta/p_tau_inv` was `k_beta` square, so a
  square but asymmetric one ran under `bayests posterior` and exited 0 without a
  word. It was found in a Dees et al. (2007) GVEC reproduction. The samplers read
  the matrix as if it were symmetric, and not consistently. kron(., P_tau^-1)
  enters the posterior precision of beta, whose draw factorises the upper
  triangle only. beta' P_tau^-1 beta enters the prior precision of the loadings
  through the whole matrix. So such a file stood for a prior that was neither
  the matrix nor its transpose, and not the same prior in the two blocks.
  bvartools' `add_priors()` already refuses such a matrix. A file written any
  other way, from h5py or by hand in R, reached BayesTS unchecked.

  `validate()` now refuses it in all four constant VECs (`VecNormalWishart`,
  `VecNormalGamma`, `VecNormalStochvol`, `VecKlgs2010`), and so does
  `bayests check`. The time-varying VECs get the same check on
  `/priors/beta/v_inv`, the prior precision of the state before the sample,
  which had the same gap: its draw factorises v_inv + rho^2 P'P through the upper
  triangle and multiplies the prior mean by the whole of v_inv.

  `/priors/beta/p_tau` was already required to be symmetric. It now goes through
  the same check, so all three matrices share one tolerance. A matrix counts as
  symmetric when its largest |m(i,j) - m(j,i)| is at most `1e-8` times its
  largest absolute element. The tolerance is relative because rounding scales
  with the entries, and loose because rounding is either nothing, as for an outer
  product or bvartools' (P + P') / 2, or about condition number times machine
  epsilon, as for a matrix inverted through a general LU. It passes every matrix
  bvartools' `isSymmetric()` accepts up to 670 rows, and an actual mistake is off
  by orders of magnitude more. For `p_tau` this replaces an absolute bound of
  `1e-10` times the larger of one and its largest element. For every matrix with
  eigenvalues in [0, 1], the only ones accepted, the new bound is looser when
  that element is at least 0.01 and tighter below.

  The error message names the matrix and gives the asymmetry and the largest
  element it was measured against. The new `unit.coint_prior_symmetry` checks
  each of the three matrices on every VEC that reads it: exactly symmetric,
  a few ulps off and 1e-10 off are accepted; 1e-6 off and 0.1 off are refused
  with that message. Checked end to end as well: a `VecNormalWishart` file given
  an asymmetric `p_tau_inv` runs and exits 0 under the binary built before this
  change, and exits 1 under `check` and `posterior` after it, while a copy off
  by 1e-12 still runs.

  **This is a core change.** It touches `src/core/inputs.cpp` alone, and the
  vendoring packages need to propagate it to refuse such files too.

  **Draws are unchanged** for every file the check accepts, which includes every
  file that is symmetric up to rounding: it only adds a refusal and does not
  touch what reaches the samplers. Verified with `record_fingerprints.sh` before
  and after, on the same machine and build: all 91 fixtures unchanged, none
  moved. All 293 registered tests pass. `agents.recipes` is not registered in
  this build, which has no Python with h5py. `agents/` now states the
  requirement and the tolerance under `/priors`.

- **A file that is not HDF5 is refused in one line.** Opening one used to print
  the HDF5 library's error stack, a dozen lines of internals such as
  "minor: Not an HDF5 file", on stderr ahead of BayesTS's own message. That message
  already carries what the stack says, since HighFive walks the stack into the
  exception, so the stack is now silenced while the file is opened, and only
  then. A handler installed beforehand is restored, so a host that set its own
  keeps it. Exit codes are unchanged, and no sampler is touched, so draws are
  unchanged by construction.

- **A written `h = 0` skips the forecast, as a missing `h` does.** The eighteen
  non-quantile front-ends skipped the forecast only when `/model` had no `h`
  attribute and handed any written value to the sampler, which refuses a
  horizon below one. So a file carrying `h = 0` — the natural way for R and
  Python callers to write "no forecast" — ran the whole Gibbs chain under
  `bayests posterior` and then exited 1 with "forecast horizon (h) must be
  positive". Reproduced for `VarNormalGamma`, `DfmNormalGamma` and
  `FavarNormalWishart`; `VarNormalAld` and `VarTvpAld` were unaffected, their
  `forecast()` being a no-op. The front-ends now read the attribute through its
  default and skip silently for any `h <= 0`, which is what the README and
  `agents/` already promised: having nothing to do is not failing.

  `bayests check` (above) refused such a file up front for the same reason.
  That refusal is removed, and `agents.recipes` now requires `posterior` to exit
  0 on a written `h = 0` and `check` to accept it. Only exit codes change: no
  sampler is touched, so draws are unchanged by construction.

- **The two golden tests over one multi-model fixture no longer race under
  `ctest -j`.** `golden.VarNormalGamma-multi_submodels_US` and `_JP` run
  `bayests_golden` over the same file, and each staged its copy at
  `<temp>/bayests_golden/<file name>` — the same path — so one could overwrite
  the copy the other had open, and HDF5 failed with "unable to synchronously
  open file". Intermittent in parallel, never serially. The same clash was
  waiting for two recorded fixtures sharing a basename, or two build trees
  testing at once.

  Each invocation now stages into a directory of its own under
  `<temp>/bayests_golden/`, removed when the run passes and kept, with its path
  printed, when it fails. The fixture is still only read. Test harness only: no
  sampler is touched and draws are unchanged.

- **The local Docker harness copies out the packages it just built**, and not
  whatever else is lying in the build tree. `docker/ci.sh` globbed
  `*.tar.gz`, `*.zip` and `*.sha256` out of the Release build directory; with
  the `bayests-ci-work` named volume that directory survives between runs, so
  after the 0.0.1 → 0.1.0 bump the previous version's archives were still
  sitting there and were copied into `/out` again, restamped with the new run's
  time. A directory presented as the output of one build described two.

  The copy now names `CPACK_PACKAGE_FILE_NAME`, read back out of the
  `CPackConfig.cmake` the configure wrote — the same string CPack names the
  files with, so it cannot disagree with what is on disk the way a version
  parsed out of `CMakeLists.txt` could. Nothing about the build, the tests or
  the packages themselves changes; no sampler is touched and no draws move.

  `docker/README.md` now also says that `ci.sh` is copied into the image rather
  than read from the mounted checkout, so editing it does nothing until
  `docker build` runs again. That rebuild is seconds — the `COPY` is after the
  vcpkg layer.

## 0.1.0 — 2026-09-12

First tagged release, and the version the JSS manuscript describes. The
leading zero is deliberate: twenty algorithms, the model-file layout, the
four commands and the public headers in `include/bayests/` are all in place
and tested, but the interface is not yet being promised as stable, and a
later release may change it without the ceremony a 1.x would owe. A 1.0.0
is what that promise will be spelled as, once the model set has stopped
moving and the manuscript describing it has been through review.

Everything below was developed before that line was drawn.

### Added

* **Release metadata: `CITATION.cff` and `.zenodo.json`.** Documentation only,
  so draws are unchanged by construction.

  Both exist so that an archived copy of this repository is attributed to a
  person rather than to a string. Without them Zenodo builds its record from
  what it can infer from GitHub, which gets the author from the account name
  and the description from the repository blurb; `CITATION.cff` does the same
  job for GitHub's own "Cite this repository" box. They carry the title, the
  BSD-3-Clause licence, the keyword set and the author's ORCID, which was
  checked against both its ISO 7064 check digit and the public registry before
  it was written down. Both files are parsed as part of the commit that adds
  them, because a malformed one fails silently at archive time rather than
  loudly here.

  `.zenodo.json` deliberately carries no version field: Zenodo takes that from
  the tag, and a version in two places is a version that will disagree with
  itself.

* **A prior on `rho`**, the autoregression of the cointegration state equation,
  so that the three time-varying VECs -- `VecTvpWishart`, `VecTvpGamma` and
  `VecTvpStochvol` -- estimate it rather than hold it at whatever the file says.
  This is the block Koop, Leon-Gonzalez and Strachan (2011) add to their sampler
  and the one piece of their model that was missing here.

  Write `/priors/beta/rho_min` and `/priors/beta/rho_max` to turn it on. The two
  are the support of a uniform prior and must be given together; `/priors/beta/rho`
  is then the value the chain starts at rather than the value it keeps, and it
  has to lie inside that support. The draws land in `/posterior/beta/rho`, one
  per iteration. A file that names neither bound runs exactly as it did.

  **Draws are unchanged** for every file that does not put a prior on rho, which
  is every file written before this. Verified over the whole fixture suite: with
  the state equation fix below reverted, all 85 existing fixtures print their
  previous fingerprints digit for digit, this block being unreachable without
  the two bounds. Three new fixtures -- `VecTvpWishart-rho`, `VecTvpGamma-rho`
  and `VecTvpStochvol-rho` -- cover the new block, one per model. The fix below
  does move the three models' draws, and says so.

  The block is a Gibbs step and not the Metropolis-within-Gibbs one the paper
  needs, and the difference is in the model rather than in the algorithm. With
  the state innovation variance fixed at the identity the path contributes a
  normal likelihood in rho, and a uniform prior makes the conditional that
  normal truncated to the prior's interval -- an exact draw. What makes it
  non-standard in the paper is their initial condition: their beta at the start
  of the sample is drawn from the state equation's own stationary distribution
  `N(0, I / (1 - rho^2))`, which puts rho where no conjugacy survives. BayesTS
  gives beta before the sample a normal prior of its own, read from
  `/priors/beta/mu` and `/priors/beta/v_inv` and free of rho, so it drops out of
  the conditional. Anyone comparing the two samplers should know that this is
  the piece that differs: the prior over the cointegration space at the start of
  the sample, not the draw of rho itself.

  The truncated normal it draws from is new, in
  `src/core/algorithms/truncated_normal.cpp`, and covered by
  `unit.truncated_normal`. It is rejection sampling from three envelopes rather
  than an inverse of the truncated CDF, because the interval here is routinely
  both narrow and far out in a tail -- a prior support a thousandth of a unit
  wide, against a conditional mean that can sit tens of standard deviations
  outside it -- which is where the quantile route returns the nearer endpoint
  every time and looks like a chain stuck at a boundary.

* **`--all-groups`**, which runs every model in a file rather than the one
  `--group` names. With it `--group` becomes the root to search under, so
  `--all-groups` alone covers the whole file and `--group /submodels
  --all-groups` covers what is below that group. Accepted by all four
  subcommands, and it composes with directory mode: each file in the walk is
  expanded in turn, so one invocation covers a directory of files that each hold
  several models.

  **Draws are unchanged.** No sampler was touched: this is discovery and
  iteration in the command line, above the io layer. Verified with the whole
  fixture suite, and additionally by a new `VarNormalGamma-multi` fixture that
  writes two models into one file — both print the `VarNormalGamma-plain` row's
  fingerprints, digit for digit, as `VarNormalGamma-grouped` already did for one
  model under a group.

  A model is found by having a `model` subgroup with an `algorithm` attribute —
  what `get_algorithm_type()` reads — and the search stops there rather than
  descending into its `data`, `priors` and `posterior`. Results are sorted, so
  the processing order and the order failures are reported in do not depend on
  HDF5's link order.

  Without the flag nothing changes: one model per file, `--group` naming it, and
  a `--group` that names nothing still an error rather than an empty walk. A
  well-formed root that holds no model is not a failure — it is reported and
  exits 0, the rule already in force for a directory with no HDF5 files in it.

  What this is for: a caller that keeps many models in one file — the sub-models
  of a GVAR, say — no longer needs one invocation per model. Note that results
  are written in place over unlinked datasets, which HDF5 does not reclaim, so a
  file that is re-run often enough is worth an occasional `h5repack`.

* **`VarNormalAld` and `VarTvpAld`**, Bayesian quantile VARs -- the nineteenth
  and twentieth registered algorithms, and the first models here that estimate a
  conditional *quantile* rather than a conditional mean.

  **Draws are unchanged** for every existing model. Verified with the fingerprint
  comparison in CONTRIBUTING.md over the whole suite: 78 fixtures recorded before
  and after, 78 unchanged and none moved. (The raw comparison reports all 78 as
  moved, because `test/golden_models.cpp` gained one row -- the new
  `/posterior/u_scale/coeffs`, recorded as `absent` for every model that does not
  write it. With that row filtered out the two recordings are identical.)

  Minimising the quantile loss at `q` is maximising the likelihood of an
  asymmetric Laplace distribution, and that distribution is a scale mixture of
  normals. With `theta = (1 - 2q) / (q(1 - q))` and `tau2 = 2 / (q(1 - q))`,

  ```
  y_it = x_t' a_i + theta w_it + e_it,   e_it ~ N(0, tau2 s_i w_it),
  w_it ~ Exp(1 / s_i),
  ```

  has its `q`-th conditional quantile at `x_t' a_i`. Conditional on the latent
  scales `w` that is an ordinary weighted normal regression, which is why these
  are the stochastic volatility samplers with a different rule for where the
  per-period variance comes from rather than a new kind of model: the coefficient
  block is `var_normal_stochvol.cpp`'s and `var_tvp_stochvol.cpp`'s unchanged,
  fed `1 / (tau2 s_i w_it)` instead of `exp(-h_it)` and a response carrying the
  offset `theta w_it`.

  The quantile is a new `/model` attribute, `quantile`, and a new `double` field
  on `VarSpec` -- its first non-integer member. One file is one quantile; a grid
  of them is a list of models, which is how a quantile grid parallelises.

  New: `src/core/algorithms/inverse_gaussian.cpp`, the draw the latent scales
  need (their conditional is generalised inverse Gaussian at index 1/2, which is
  the reciprocal of an inverse Gaussian), and `src/core/models/ald_support.h`,
  which holds the two blocks and the density. `optional_attribute_double()` is
  new in `src/io/hdf5/model_io_common.h`; there had been no double-valued
  attribute before.

  **Three things these models do not have, each on purpose.**

  *No covariance block.* `Psi` is a triangular rotation of the errors, and
  conditional on `w` the equations are independent. Rotating them is exactly what
  stops the estimand being a quantile: the rotated residual is a combination of
  equations, and the `q`-th quantile of a combination is not the combination of
  `q`-th quantiles. `validate()` rejects `covar`.

  *No forecast.* The `h` step quantile is not the quantile of the iterated one
  step quantiles, so there is no path to simulate that could be read as one.
  `validate()` rejects a non-zero horizon -- so the file is refused before a chain
  is spent on it -- and `forecast()` throws with the reason if it is reached
  anyway. These are the only two models here that do not forecast.

  *No calibrated intervals.* The asymmetric Laplace is a working likelihood, not
  a claim about the data. The posterior locates the quantile, but its spread
  needs the sandwich adjustment of Yang, Wang and He (2016), which is not
  applied. Read the spread as a diagnostic rather than as a credible interval.
  This is stated in the class comments, the README and here, because it is a
  property of the estimator that a user cannot see in the output.

  Structural models *are* allowed: contemporaneous terms are regressors like any
  other, and adding them to an equation leaves its quantile reading intact. BVS
  is available; SSVS is not, as for the stochastic volatility models.

  Tested by `test/unit_var_ald.cpp` -- the first unit test of a VAR sampler here,
  because these two are the first whose failure mode is silent. A quantile model
  that has lost its skew term estimates the median instead and passes every smoke
  test there is, so the test checks the property that defines the estimand: the
  share of fitted residuals below zero is `q`. It comes out at 0.2500, 0.5000 and
  0.8050 for `q` of 0.25, 0.5 and 0.8. `test/unit_inverse_gaussian.cpp` covers the
  new draw against its first two moments and the reciprocal identity the caller
  relies on.

  Kozumi, H., & Kobayashi, G. (2011). Gibbs sampling methods for Bayesian
  quantile regression. *Journal of Statistical Computation and Simulation,
  81*(11), 1565-1578.

  Yang, Y., Wang, H. J., & He, X. (2016). Posterior inference in Bayesian
  quantile regression with asymmetric Laplace likelihood. *International
  Statistical Review, 84*(3), 327-344.

* **`FavarNormalWishart`**, a factor augmented VAR — the eighteenth registered
  algorithm, and the first model here whose state vector is part data.

  ```
  x_t = Lambda_f f_t + Lambda_y y_t + e_t,   e_t ~ N(0, R),  R diagonal,
  s_t = sum_{j=1..p} Phi_j s_{t-j} + v_t,    v_t ~ N(0, Q),  s_t = (f_t', y_t')',
  ```

  for `k` panel series, `n_factors` unobserved factors and `n_obs_factors`
  observed ones, after Bernanke, Boivin and Eliasz (2005). Five Gibbs blocks:
  the factor path, the loadings, the idiosyncratic precisions, the state
  innovation precision and the transition.

  *Draws are unchanged* for every existing model. The fingerprint comparison in
  CONTRIBUTING.md over the whole suite reports `76 unchanged, 0 moved` beside the
  two new `FavarNormalWishart` fixtures; the shared code this touched —
  `spec.h`/`spec.cpp`, `data.h`, `read_spec()` and `dfm_support.h` — only gained
  members and functions, and none of the seventeen existing samplers reads one.

  **What it is.** `DfmNormalGamma` with observed variables added to the state.
  The observed factors are the variables the model is about — a policy rate,
  output — and the panel is there to measure the common component they move
  with. They sit *in* the state rather than beside it because the transition is
  a VAR over both blocks jointly: the factors respond to the policy variable and
  the policy variable responds back, and that coupling is the model.

  **The path draw conditions rather than draws.** A dynamic factor model's state
  is unobserved throughout and is drawn whole; half of this one is data, so
  `chan_jeliazkov_2009_conditional` partitions the assembled precision into the
  drawn rows and the observed ones and solves `K_FF f = b_F - K_FY y` — the same
  band one block size narrower. That is exact. Adding the observed factors to the
  measurement with a small error variance is the usual shortcut and is not
  available to a precision based sampler at all, exact observation being infinite
  precision, and it would also be a different model. Conditioning keeps the
  information the observed factors' own equations carry about the lagged
  factors, which a sampler that drew the factor block alone would discard.

  **Why the Wishart, and what it forces.** A factor model's idiosyncratic `R`
  must stay diagonal — errors free to correlate leave the factors nothing to
  explain — which is why no `Dfm*` offers a Wishart. `Q` is a different object:
  a VAR's innovation covariance, whose observed block is an ordinary one and
  whose cross block is the correlation between the factor innovations and the
  shock to the observed variables. That cross block is what a FAVAR is estimated
  to measure, and forcing it to zero would assert the policy shock is orthogonal
  to every factor innovation. So the third part of a `Favar*` name refers to `Q`,
  and `R` is gamma-diagonal throughout the family — the one place the naming rule
  differs from the DFM row.

  That choice fixes the identification and the two cannot be picked separately.
  A rotation `F -> C F` is invisible in the measurement if the loadings absorb
  it. A DFM rules it out with a unit lower triangular loading block *and* a
  diagonal `V`, which together admit only `C = I` by the uniqueness of an LDL
  factorisation. With `Q` free the second half is gone, so the leading
  `n_factors` square loading block is the **identity** here rather than a unit
  triangle, and the observed columns of those rows are zero: the first
  `n_factors` panel series are the factors plus idiosyncratic noise and carry no
  free loading at all. The two identifications cost the same `n(n-1)/2`
  restrictions — a FAVAR spends them on the loadings instead of on `V`. Taking
  the DFM's rule over instead leaves a model that runs, produces plausible
  numbers, and has loadings free to wander along a ridge; that was caught in
  development by a recovery check, whose loading error fell from 0.25 to 0.07
  once the block was tightened.

  **On disk**, the tree is a DFM's with two changes, both from half the state
  being data. `/data/train/f_obs` holds the observed factors, `tt` by
  `n_obs_factors`, beside the panel in `/data/train/y`; `/model/n_obs_factors`
  names their count and a file that omits it describes a dynamic factor model,
  which `validate()` says by name rather than estimating. `/priors/v_sigma`
  carries `df`/`scale` rather than `shape`/`rate`, and `/initial/v_sigma_inv` is
  an `n_state` square matrix rather than a diagonal — that pair is what a file
  written against a DFM gets wrong, and both are rejected by shape.

  `/posterior/factors/coeffs` holds the unobserved factors alone: the observed
  half is the caller's own input, and a copy of it per draw would be the largest
  thing in the file. `/posterior/forecast` is the one forecast here wider than
  `k` — `h * (k + n_obs_factors)`, the panel of a horizon followed by the
  observed factors of the same horizon, which are what the model is forecast for
  and have no other dataset to go in.

  `VarSpec` gains `n_obs_factors`, `uses_obs_factors()`, `n_state()`,
  `n_favar_lambda()` and `n_favar_a()`. The last two are paired with
  `n_lambda()` and `n_factor_a()` the way `n_non_structural_vec()` is paired with
  `n_non_structural()`, and `n_favar_lambda()` agrees with `n_lambda()` at no
  dimension at all — the identifications differ. `TrainData` gains `f_obs`, a
  member of its own rather than a reuse of `x`: the observed factors are not
  regressors, they appear on the left of the transition as well as the right.

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

* **`chan_jeliazkov_2009_conditional()`**, a second entry point to the banded
  state path draw that holds the trailing elements of every state column at
  observed values instead of drawing them.

  *Draws are unchanged.* No sampler calls it yet — it is the algorithm a factor
  augmented VAR needs, landed on its own so that the refactor underneath it can
  be checked in isolation. The existing entry point was split into an assembly,
  the conditioning step and the draw, with the moved bodies left textually
  identical; `test/record_fingerprints.sh` before and after the split reports
  `76 unchanged, 0 moved` over all 1140 fingerprints of a clean-clone suite,
  which is every sampler that reaches the band — the four DFMs and every
  time-varying parameter VAR and VEC.

  What it is for. A dynamic factor model draws its whole state; a factor
  augmented VAR cannot, because half of its state is data — the transition is a
  VAR over the unobserved factors and the observed variables jointly. Adding
  those variables as measurements with a tiny error variance is the usual
  shortcut and is not available to a precision based sampler at all, exact
  observation being infinite precision. Conditioning is exact instead, and cheap:
  partitioning the assembled precision into the drawn rows F and the observed
  rows Y gives `K_FF f = b_F - K_FY y`, which is the same band one block size
  narrower — still banded, because the observed positions are the same rows of
  every block, and still positive definite, a principal submatrix of a positive
  definite matrix being one.

  Two things a caller should know. The measurement is given against the *whole*
  state, `[Lambda_f Lambda_y]`, and the observations are passed unmodified:
  subtracting `Lambda_y y_t` from the data by hand as well would subtract it
  twice, since that term is one of the cross terms the conditioning removes.
  And there is no trailing column to drop — the state column past the end of the
  sample would be half data and half not, so it is never built. That is exact,
  not an approximation: the term it contributes is a normalised Gaussian density
  in that column alone. `unit_chan_jeliazkov.cpp` checks it as linear algebra,
  marginalising the last block out of a dense system and recovering the
  truncated one to 3.6e-15, and checks the draw itself against the dense
  conditional's mean and standard deviation over 20000 draws.

* **The Linux CI jobs as a Docker image**, in `docker/`. One image carries the
  same toolchain and dependency set the Ubuntu runner uses -- gfortran, an
  upstream CMake, Ninja, and Armadillo and HDF5 from vcpkg's
  `x64-linux-dynamic` triplet, with HighFive at the ref `.github/highfive-version`
  pins -- and `docker/ci.sh` runs the steps of `ci.yml`, `docs.yml` and
  `fingerprints.yml` against it. The point is the platform gap: this project is
  developed on Windows, so the Linux jobs were previously unrunnable before
  pushing.

  The checkout is mounted read-only and mirrored inside the container, so a
  Linux configure never meets the Windows build tree, and what gets built is the
  working tree with its uncommitted changes. Nothing about the library changes:
  this is a second way to run what the workflows already run, and the workflows
  themselves are untouched. **Draws are unchanged** -- no source file was
  edited.

  `docker/README.md` records where it is deliberately not the runner, of which
  the one that matters is that a fingerprint recorded in the container is
  comparable to another recorded in the container and to nothing else, the same
  rule that applies to any two machines.

### Changed

* **The sampler and test counts in the three documentation files are brought
  back to the code.** Documentation and comments only — no sampler, no header
  and no I/O code was touched, so draws are unchanged by construction. `ctest`
  is 183/183 and a clean rebuild is warning-free after the change.

  * "All eighteen samplers are covered from a clean clone" predated
    `VarNormalAld` and `VarTvpAld`; it is twenty, which is what
    `src/models/model_factory.cpp` registers and what the distinct model column
    of `test/CMakeLists.txt`'s `bayests_add_model_fixture` calls covers.
    `CLAUDE.md` had it both ways, "eighteen" in the build section against
    "Twenty registered algorithms" in the taxonomy. Corrected in `CLAUDE.md`,
    `CONTRIBUTING.md` and `README.md`.
  * The suite is 183 tests from a clean clone, not 169: fifteen `unit.*` and
    eighty-four `fixture.*`/`golden.*` pairs. The sizes quoted beside it had
    drifted with it — `ctest -V` is about 1.1 MB and a `record_fingerprints.sh`
    reduction of it about 115 KB (1344 fingerprints, sixteen per fixture), and
    one golden test's own block is about 12 KB rather than the 7 KB claimed.
    All measured on one run of the suite.

* **`obeject` and "Stop of log likelihood" in the twelve older `BaseModel`
  front-ends.** Comment typos, fixed to the wording the `Dfm*` and `Favar*`
  front-ends already use. Nothing executable changed.

* **Two comments that had drifted from what they describe.** No behaviour and no
  fixture row changed; `ctest` is 183/183 either side.

  * `test/CMakeLists.txt`'s VEC preamble had come loose from the VEC rows and
    fused with the DFM comment, ten lines above a block about factor models. It
    is moved down to where the VEC fixtures actually start and corrected on two
    points: it claimed `VecNormalWishart` had no generator, which it has had
    since the Wishart fixtures were added, and it claimed SSVS reaches
    `VecNormalGamma` alone, when it also reaches `VecNormalWishart` — the two
    VEC models with constant coefficients and no stochastic volatility, which is
    what the SSVS note at the head of the matrix already says. Its structural
    count is four, not five.
  * `.github/dependabot.yml` named
    `.github/actions/setup-linux-deps/action.yml` as the place the HighFive tag
    is pinned. It is `.github/highfive-version`, which that action, the Windows
    jobs of `ci.yml` and `release.yml`, and the check in `snap.yml` all read.

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

* **The model-file orientation is now stated for both kinds of reader,
  everywhere it appears.** `README.md`, `CLAUDE.md` and `CONTRIBUTING.md` all
  carried the R-centric wording; all three now give both orientations.
  Documentation only — no sampler, no header and no I/O code was touched, so
  draws are unchanged by construction.

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

  The README's algorithm count was stale in the same pass and was corrected
  with it: the sentence said fourteen, and enumerated one dynamic factor model,
  from before `DfmNormalStochvol` was added above.

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

* **A time-varying VEC centred the cointegration space of the first period over
  `beta_0` rather than over `rho beta_0`.** The simulation smoother's sixth
  argument is the prior mean of the state the *first* observation loads on, and
  it does not put the transition matrix through it; all three of
  `VecTvpWishart`, `VecTvpGamma` and `VecTvpStochvol` passed `/initial/beta_init`
  there unchanged. That is the random walk's answer and is exactly right at
  `rho = 1`. Below one the smoother was centring `beta_1` over `beta_0` while
  the Gibbs block for `beta_0` a few lines further down was centring it over
  `rho beta_0` -- two different models, one per block.

  **Draws change**, for those three models and only where `rho` is below one --
  which is the default, 0.999. Everything else in the file is untouched: of the
  85 fixtures, the 15 time-varying VEC rows move and the other 70 print their
  previous fingerprints digit for digit. The move is small at a `rho` near one,
  a tenth of a percent on the summary statistics of `VecTvpWishart-plain` at the
  fixtures' `rho = 0.99`, and it is not a rounding error: it is one period of
  the state equation that was being skipped.

  The new numbers are the right ones because `/initial/beta_init` and
  `/priors/beta/mu` are documented, and read by every other block, as the
  cointegration space of the period *before* the sample. Carrying it into the
  sample is one application of the transition.

  Found while adding the uniform prior that lets `rho` be drawn rather than
  pinned just below one, and the reason that could not be added on top of this:
  a `rho` that moves makes the discrepancy as large as the draw wanders.

* **`bayests` exited 0 when a run started and failed.** Every `BaseModel`
  front-end wrapped its body in `catch (const std::exception &e) { std::cerr <<
  e.what(); }` and returned `void`, so the subcommand above it saw nothing: a
  model file the sampler rejected, a singular matrix, a missing prior — all
  exited 0, with the datasets simply absent. README and `CLAUDE.md` have always
  documented exit 1 for "the run started and something failed", and a script
  looping over model directories could not tell a rejected file from a fitted
  one.

  The 58 catch blocks are gone. A stage that cannot do what it was asked throws;
  the subcommands, which already had a handler that returns 1, catch it and
  report `Error processing <file>[:<group>]: <reason>` — the location included,
  which the front-ends' own handler never printed. A directory walk still visits
  every file and still exits 1 if any of them failed.

  **Having nothing to do is not failing, and still exits 0.** Output that is
  already present; a forecast on a model whose file carries no horizon; and a
  quantile model's `forecast()`, since `validate()` refuses it a horizon in the
  first place and `bayests forecasts` over a directory should not fail because
  some of the models in it are quantile models. Five front-ends used to print
  `Error processing ...: Forecast horizon h is missing.` where the other
  thirteen returned in silence, and `VarTvpWishart` alone announced an existing
  forecast the same way; all eighteen are quiet about it now, because none of
  them has failed.

  **Draws are unchanged.** Verified with the fingerprint comparison in
  CONTRIBUTING.md over the whole suite: 84 fixtures before and after, 84
  unchanged and 0 moved. Exercised end to end besides — `forecasts` and `loglik`
  on a file with no draws exit 1 and name the file; a full `posterior`, a second
  `posterior` over finished output, an `h = 0` model and `forecasts` on a
  quantile model all exit 0; a directory holding one healthy file and one
  without draws visits both, fails one and exits 1.

  `test/golden_models.cpp` catches each of the three stages itself, so all three
  are still attempted and the fingerprints still printed whatever happened — what
  a fixture wrote before it failed is most of the evidence for why. Its
  empty-stage check is unchanged and still the thing that fails the test.

* **`n_favar_lambda()` claimed it could never be confused with `n_lambda()`,
  and it can.** Three comments — `VarSpec::n_favar_lambda()` in
  `include/bayests/spec.h`, the `lambda` check in `FavarNormalWishartInput::
  validate()`, and the taxonomy section of `CLAUDE.md` — said the DFM's loading
  count and the FAVAR's "do not coincide at any dimension". They differ by

  ```
  n_lambda() - n_favar_lambda() = n(n - 1)/2 - (k - n) n_obs
  ```

  which is zero on a family rather than nowhere. With one observed factor it
  vanishes at every `k = n(n + 1)/2` — `(k, n, n_obs)` of `(3, 2, 1)`,
  `(6, 3, 1)`, `(10, 4, 1)`, `(15, 5, 1)` and up — and elsewhere wherever
  `(k - n) n_obs` hits `n(n - 1)/2`, as `(4, 3, 3)` and `(7, 4, 2)` do. Checked
  against the header itself, not derived on paper.

  That matters because the claim was doing work: `validate()` checks
  `/priors/lambda` by *length*, and the comment told the reader a substitution
  would be caught. At `k = 3, n = 2, n_obs = 1` — a plausible first FAVAR — a
  lambda laid out to the DFM's unit lower triangular convention is three
  elements, exactly what a FAVAR asks for, and is then read row by row as
  something else. The chain runs to completion on a different model.

  Comments only; no count, no check and no sampler changed, so draws are
  unchanged by construction. The corrected comments state the difference, the
  family it vanishes on, and that length is not evidence of convention. The
  length check itself is unchanged — catching this needs the file to say which
  convention it was written to, which is a format question, not a comment.

* **`VarTvpWishart` and `VecTvpWishart` set `spec.covar` on a model that has no
  psi block.** Their readers passed `"wishart"` as `read_spec()`'s `covar_error`
  argument, so any file carrying `error = "wishart"` — which is what
  `make_model_fixture.cpp` writes for them, and what a real one carries — came
  back with `spec.covar == true`. Nothing consumed it: neither input struct
  defines `use_psi()`, neither sampler reads `n_psi()`, and `validate()` says so
  in as many words. But the flag means "this model has a covariance block", these
  two do not, and the next `n_psi()` added anywhere near them would have sized a
  block off it. Both now pass `nullptr`, which is what the two `*NormalWishart`
  readers beside them have always done; all five Wishart-family readers and all
  twelve without a psi block now agree.

  **Draws are unchanged.** Verified with the fingerprint comparison in
  CONTRIBUTING.md over the whole suite: 84 fixtures recorded before and after on
  one machine and one build, 84 unchanged and 0 moved.

  `gamma+covar` and `sv+covar` are now the only two values of the `error`
  attribute that switch anything on. `CLAUDE.md` and README had documented
  `wishart` as a third, the README noting that on those two models it "turns
  nothing on and is there for symmetry" — true of the outcome, not of the flag.
  Both are corrected, as is the comment in `make_model_fixture.cpp` that credited
  only `VarNormalWishart`'s reader with ignoring the attribute.

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
