# The model file

One HDF5 file holding the data, the priors, the starting values and the
`/model/algorithm` attribute that names the sampler. Everything below is a path
*within one model*. They are paths from the root of the file unless the run was
given `--group`, in which case they all hang under that group instead — which is
how one file holds many models.

**Only four things are required**: `/model`'s `algorithm`, `k`, `iterations` and
`burnin`. Everything else is read through a default, so a file written for a
simpler model still describes a valid one. The corollary is that a *missing*
field is usually not an error — it is a different model.

## Orientation, once, for both audiences

Every dataset below is described in **HDF5 dataspace terms**: one row per
quantity, one column per draw or per observation. That is what `h5py` reports
and what a C or Python reader should expect.

R's HDF5 readers reverse the dimension order, R being column-major where HDF5 is
row-major, so an R session sees the transpose of each shape written here. For
the posterior that means draws in rows, which is what `coda` expects of an
`mcmc` object and why `start`, `end` and `thin` are written alongside.

Practically:

| Writing from | Do this |
| --- | --- |
| Python / h5py | Store the transpose: `f["/data/train/z"] = Z.T` for the `(tt*k, nparams)` matrix `Z` |
| R | Store the matrix as it reads; R's reversal produces the right dataspace |
| C++ (in-tree) | `write_armadillo_matrix_to_hdf5` transposes for you |

A **vector** is stored as a single HDF5 row: `(1, n)` in dataspace, `(n, 1)`
seen from R.

## `/model` — attributes

| Attribute | Type | Default | Meaning |
| --- | --- | --- | --- |
| `algorithm` | string | *required* | Which of the twenty-two algorithms runs. The only thing that decides |
| `k` | int | *required* | Endogenous variables. For a factor model, the number of **observed** series |
| `iterations` | int | *required* | Draws kept. Must be positive. On the two `*TvpDiscount` models there is no chain, so it is how many i.i.d. draws a forecast takes |
| `burnin` | int | *required* | Draws discarded before the first kept one. Must be 0 on a `*TvpDiscount` model, which iterates nothing |
| `thin` | int | 1 | Keep one draw in `thin` after the burn-in, the last of each block. The chain runs `burnin + iterations * thin` draws and every result still holds `iterations`. At least 1, and exactly 1 on a `*TvpDiscount` model |
| `p` | int | 0 | Lags. For a factor model, the lag order of the transition. For a VEC, a **level** order |
| `forecast_states` | string | `simulate` | `simulate` or `hold`. Whether the forecast of a model with time-varying coefficients or volatilities — VAR, VEC or factor model — carries its random walks over the horizon or keeps them at the last sample period — see `results.md`, *What a forecast does with drifting states*. Any other spelling throws |
| `m` | int | 0 | Exogenous variables |
| `s` | int | 0 | Lags of the exogenous variables. For a VEC, a **level** order |
| `n` | int | 0 | Deterministic terms entering unrestricted |
| `h` | int | 0 | Forecast horizon. Absent or 0 means no forecast was asked for |
| `varsel` | string | `none` | `none`, `ssvs` or `bvs`. Any other spelling throws |
| `structural` | bool | false | Move the last `k(k-1)/2` entries of `a` to contemporaneous coefficients. Read from an HDF5 boolean enumeration (h5py, HighFive, or an R logical, whose NA is refused) or an integer (non-zero is true) |
| `error` | string | `""` | Descriptive, except for `gamma+covar` and `sv+covar` — see below |
| `quantile` | double | 0.5 | The quantile an `*Ald` model estimates. Must lie in `(0, 1)`. Ignored by every other model |
| `delta_beta` | double | 1.0 | `VarTvpDiscount` and `VecTvpDiscount`: how fast the coefficients drift. In `(0, 1]`; 1 holds them constant. Ignored by every other model |
| `delta_sigma` | double | 1.0 | The same for the error covariance. In `(0, 1]`, and `1/(1 - delta_sigma)` must exceed `k` — the degrees of freedom settle there whatever the prior says |
| `rank` | int | 0 | A VEC's cointegration rank. Zero means "not a cointegrated model" |
| `k_beta` | int | 0 | Rows of the cointegration matrix. `rank` may not exceed it |
| `n_restricted` | int | 0 | A VEC's deterministic terms restricted to the cointegration space. Disjoint from `n` |
| `n_factors` | int | 0 | Unobserved factors. Zero for every model that is not a factor model |
| `n_obs_factors` | int | 0 | A FAVAR's observed factors. Needs `n_factors > 0` as well |
| `seed` | int | none | Seeds the random number generator for this model's run, so its draws depend on the file alone. A non-negative whole number; anything else is refused. Absent leaves the generator as it is — see `pipeline.md`, *Reproducibility*. A `*TvpDiscount` model's estimation consumes no random numbers at all, so a seed reaches only its forecast |

`error` is load-bearing for exactly two values, and the spelling is
model-specific: `gamma+covar` switches the covariance block on for the gamma
models, `sv+covar` for the stochastic volatility models. The fourteen readers
of models with no `psi` block never compare it.

## `/model/priors/psi` — attribute

| Attribute | Meaning |
| --- | --- |
| `varsel` | The selection scheme for the covariance block alone, read by the four time-varying models that have one, with the block switched on. The `/model` attribute governs the coefficients. `bayests check` warns when the model does not read it |

## `/data/train`

| Dataset | Dataspace shape | Contents |
| --- | --- | --- |
| `y` | `(1, tt*k)` | The response, already stacked: `vec(Y')`, all `k` variables of period 1, then period 2, and so on |
| `z` | `(nparams, tt*k)` | The SUR regressor matrix. On paper it is `(tt*k, nparams)`, one block of `k` rows per period |
| `w` | `(k_beta, tt)` | A VEC only: the error correction term. On paper `(tt, k_beta)` |
| `x` | `(n_x, tt)` | The regressors in the compact layout, one column per regressor on paper. Read by `VecKlgs2010`, `VarTvpDiscount` and `VecTvpDiscount` in place of `z`. For `VecTvpDiscount` it holds the short-run blocks alone, `n_x_vec` of them: the `rank` error correction columns are built from `w` and `/initial/beta` |
| `f_obs` | `(n_obs_factors, tt)` | A FAVAR only: the observed factors. On paper `(tt, n_obs_factors)`. The observed half of the state vector, not regressors |

`tt` is never stored. It is recovered as `len(y) / k`, and a `y` whose length is
not a multiple of `k` is refused.

A factor model has **no `z`**: its regressors are the unobserved factors, drawn
rather than given, so `/data/train/y` is all the data there is.

## `/data/forecast`

| Dataset | Dataspace shape | Contents |
| --- | --- | --- |
| `x` | `(n_x, h)` | Out-of-sample regressors in the compact layout, `(h, n_x)` on paper. **Required when `h > 0`**, with exactly `h` horizons: a run and `bayests check` refuse any other number |
| `z` | `(k*n_x, h*k)` | The older SUR spelling of the same thing, `kron(x, I_k)`. Still read, and compacted on the way in |

A factor model needs neither: the horizon alone drives its forecast.

## `/data/test`

| Dataset | Dataspace shape | Contents |
| --- | --- | --- |
| `y` | `(k, h)` | What the horizon realised, `(h, k)` on paper: one row per period and one column per variable. Optional |

For a VAR that is the variable order of `/data/train/y`. **For a VEC it is the
levels**, the series `/data/forecast/x` carries the lags of, not the differences
`/data/train/y` holds: a VEC is forecast and scored in its level
parameterisation. See `results.md`.

The one thing in `/data` that no sampler reads. It is not a sample anything is
estimated on but the observations a forecast is scored against, and it lives in
the file so that one window of an expanding window exercise carries what it is
to be judged by -- which is what lets a folder of them be scored a file at a
time, instead of the caller holding the next window in memory to supply the
observation this one predicted.

**Fewer than `h` periods is fine** and means what it says: an expanding window
whose last windows run past the end of the sample realises fewer periods than it
forecasts, and those are the ones that can be scored. More than `h` is refused,
since the file then describes a horizon other than the one `/model` asks for.
The stacked spelling `/data/train/y` also accepts is refused here rather than
guessed at: once `h` is unknown, `h*k` numbers in one column could be in either
order and scoring against the wrong one would be a reshuffle of the right
numbers.

`bayests forecasts` scores the forecast against it where the algorithm can be
scored, writing `/posterior/forecast/loglik`; see `results.md`. `bayests check`
reports how many periods are there, says where the algorithm cannot use them
yet, and warns where there are some but `/model` asks for no horizon.

## `/priors`

| Group | Datasets | Read by |
| --- | --- | --- |
| `/priors/a` | `mu` `(1, nparams)`, `v_inv` `(nparams, nparams)` | Every constant-coefficient model |
| `/priors/a` | `shape`, `rate`, `mu`, `v_inv` | Every time-varying model: `shape`/`rate` on the innovation precision of the random walk, `mu`/`v_inv` on the state before the sample. `v_inv` must be positive definite -- the samplers integrate that state out of the first period's prior, which takes its inverse -- so a flat prior of zeros is refused. The same holds for `/priors/psi` and `/priors/lambda` |
| `/priors/a` | `inprior` `(1, nparams)`, `include` (one-based ints), `tau0`, `tau1` | Added when `varsel` is on. `tau0`/`tau1` for `ssvs` only. With `bvs` the tightness of `v_inv` above decides whether selection can work at all — see the skill's `varsel` section |
| `/priors/a` | `mean` `(k, n_x)`, `cov` `(n_x, n_x)` | The two `*TvpDiscount` models, and **not** `mu`/`v_inv`: a matrix normal prior, `mean` being the `n_x` by `k` coefficient matrix on paper and `cov` the *regressor* side of its covariance. The equation side is the error covariance the Wishart prior carries, which is what makes the posterior conjugate. A file bringing `mu`/`v_inv` instead is read as having no coefficient prior, and `bayests check` warns that nothing read them |
| `/priors/psi` | The same shapes at width `k(k-1)/2` | The models with a covariance block switched on. Every `psi` vector, here and under `/initial`, is the strict lower triangle of `Psi` **row by row** — `(1,0) (2,0) (2,1) (3,0) ...` — not column by column as R's `m[lower.tri(m)]` gives it. The two orders agree up to `k = 3`, so a writer that gets it wrong still passes on three variables |
| `/priors/u_sigma` | `df` (scalar), `scale` `(k, k)` | The Wishart models |
| `/priors/u_sigma` | `shape` `(1, k)`, `rate` `(1, k)` | The gamma models |
| `/priors/u_sigma` | `offset`, `sigma`, `shape`, `rate`, `mu` `(1, k)` and `v_inv` `(k, k)` | The stochastic volatility models |
| `/priors/a`, `/priors/psi`, `/priors/u_sigma` | `omega_v` in place of `shape`/`rate` | `VarTvpStochvol`, and `VarTvpGamma` for `a` and `psi`: the non-centred parameterisation of that block's random walk. See below |
| `/priors/u_scale` | `shape` `(1, k)`, `rate` `(1, k)` | The two `*Ald` models: the scale of the asymmetric Laplace |
| `/priors/beta` | `v_inv` (scalar), `p_tau_inv` `(k_beta, k_beta)` | The constant VECs: the cointegration space prior |
| `/priors/beta` | `mu`, `v_inv`, optional `rho`, optional `rho_min`/`rho_max`, optional `p_tau` `(k_beta, k_beta)` | The time-varying VECs — a state equation rather than a shrinkage. See below |
| `/priors/beta` | nothing | `VecTvpDiscount` reads no prior over the space: it conditions on the one in `/initial/beta` rather than drawing it |
| `/priors/lambda` | `mu`, `v_inv` (and `shape`/`rate` where the loadings drift) | The factor models: the free loadings |
| `/priors/v_sigma` | `shape` `(1, n_factors)`, `rate` `(1, n_factors)` | `DfmNormalGamma` and `DfmTvpGamma`: the factor innovation precisions |
| `/priors/v_sigma` | `offset`, `sigma`, `shape`, `rate`, `mu` `(1, n_factors)` and `v_inv` `(n_factors, n_factors)` | `DfmNormalStochvol` and `DfmTvpStochvol`: the factor innovations' log-volatilities, the same group `/priors/u_sigma` is for the series, at the width of the factors |
| `/priors/v_sigma` | `df`, `scale` `(n_state, n_state)` | `FavarNormalWishart`: its state innovation precision is a matrix |

### The non-centred random walk and the test for time variation

`VarTvpStochvol` reads each of its three random walks -- the coefficients
(`/priors/a`), the covariance block (`/priors/psi`) and the log-volatility
(`/priors/u_sigma`) -- in one of two parameterisations, block by block.
`VarTvpGamma` does the same for its two, the coefficients and the covariance
block; its error precision does not move, so `/priors/u_sigma` keeps its gamma
prior there:

- **Centred**, the default: `shape` and `rate`, an inverse gamma on the variance
  of the innovations.
- **Non-centred**: `omega_v` `(1, n)`, with `n` the block's width, in place of
  `shape` and `rate`. It is the variance of a normal prior on the *signed
  standard deviation* of the innovations, `omega ~ N(0, omega_v)` with the
  variance equal to `omega^2` (Frühwirth-Schnatter and Wagner 2010), so
  `omega_v` is also the prior mean of that variance. It must be finite and
  greater than zero.

A block that has both `omega_v` and `shape`/`rate` is refused rather than read
one way or the other. The starting values are the centred ones: the chain starts
`omega` at the square root of the variance `/initial/a_sigma_inv`,
`/initial/psi_sigma_inv` or `/priors/u_sigma/sigma` gives, which therefore has to
be strictly positive.

What the non-centred form buys is that "this state does not move" is
`omega = 0`, a point *inside* the prior rather than at the edge of it, so its
Bayes factor is a Savage-Dickey density ratio that one run of the time-varying
model estimates (Chan 2018). The draws for it are written beside the block's
`sigma`; `results.md` has them and the formula.

Only `VarTvpStochvol` and `VarTvpGamma` read `omega_v` so far. On any other model it is a dataset
nothing reads, which `bayests check` warns about, and the model runs centred.

`rho` is the autoregression of a time-varying VEC's cointegration state
equation. It must lie in `(0, 1]`; 1 is the random walk. Giving **both**
`rho_min` and `rho_max` turns it into a drawn parameter with that uniform prior,
and `rho` is then the value the chain starts at and must lie inside the support.
One end without the other is refused rather than guessed at.

The matrices under `/priors/beta` must be symmetric: `p_tau_inv` on the
constant VECs, and `v_inv` and `p_tau` on the time-varying ones. A matrix is
refused, by a run and by `bayests check`, when its largest difference from its
transpose is more than `1e-8` times its largest absolute element. That lets
through the rounding a computed matrix carries and stops one that is not the
matrix meant: the samplers read parts of these through one triangle and parts
through the whole matrix, so an asymmetric one would run as neither. `p_tau` must
also have its eigenvalues in `[0, 1]`.

Values are checked as well as shapes, by a run and by `bayests check`:

- Every `v_inv` and every Wishart `scale` must be symmetric, to the same
  tolerance.
- A gamma `shape` or `rate` must be finite and at least zero. Zero is an
  improper prior the sample makes proper; a negative value is refused.
- A log-volatility `offset`, SSVS `tau0` and `tau1`, and the initial variance of
  a log-volatility's innovations must be finite and greater than zero.
- `inprior` must lie in `[0, 1]`.
- Under `ssvs`, at every selected position: `/priors/a/mu` must be zero,
  `/priors/a/v_inv` must be zero off the diagonal in that row and column, and
  `tau0` must be smaller than `tau1`. The same for `/priors/psi`. The diagonal
  of `v_inv` there is read for the first draw only; the sweep replaces it with
  the spike or slab precision from then on.
- Nothing checks a prior precision for being tight enough for `bvs` to select
  anything, and nothing compares `include` against the indicators `/initial`
  starts them at. Both are ways of getting a run that finishes and a posterior
  that looks ordinary while the model estimated was not the one intended; the
  skill's `varsel` section has each.
- A starting precision the sampler redraws only the diagonal of must be
  diagonal: `u_sigma_inv` of `VarNormalGamma` and `VecNormalGamma`, `u_omega_inv`
  of the time-varying gamma models, `a_sigma_inv` and `psi_sigma_inv` of the
  time-varying VARs and VECs, and `lambda_sigma_inv` and `a_sigma_inv` of the
  time-varying DFMs. Anything off the diagonal would otherwise stay in the chain
  from the first draw to the last, or be ignored without a word.

## `/initial`

Starting values, at the widths the priors imply.

| Dataset | Shape | For |
| --- | --- | --- |
| `a` | `(1, nparams)` | Constant coefficients |
| `a` | `(tt, nparams)` | Time-varying coefficients: the whole path, one period per column on paper. Exactly `tt` periods of `nparams`; a path of any other size is refused rather than padded, and so are `psi`, `beta` and `lambda` below |
| `a_sigma_inv`, `a_init` | `(nparams, nparams)`, `(1, nparams)` | The random walk's innovation precision and the state before the sample |
| `a_lambda` | `(1, nparams)` | Inclusion indicators, when `varsel` is on. Only the positions `/priors/a/include` names are ever redrawn; every other one keeps the value it starts with from the first draw to the last. Ones unless an outright restriction is meant — a zero outside `include` masks that coefficient out of the whole run silently |
| `psi`, `psi_sigma_inv`, `psi_init`, `psi_lambda` | The same at width `k(k-1)/2` | The covariance block |
| `u_sigma_inv` | `(k, k)` | The Wishart models, and `VarNormalGamma` and `VecNormalGamma`. **A factor model's is `(1, k)`** — diagonal by assumption, so it is stored flat — and only `DfmNormalGamma`, `DfmTvpGamma` and `FavarNormalWishart` read it |
| `u_omega_inv` | `(k, k)` | The time-varying gamma models |
| `h`, `h_init` | `(k, tt)`, `(1, k)` | Stochastic volatility: the log-volatility path and its start |
| `w`, `u_scale` | `(k, tt)`, `(1, k)` | The `*Ald` models: the latent scales (strictly positive) and the scale of the asymmetric Laplace |
| `beta` | `(1, k_beta*rank)` | A constant VEC's cointegration matrix, flat as `vec(beta)` — the first column of `beta`, then the second. `VecTvpDiscount` reads it from here too, in the same layout, but **not as a starting value**: it iterates nothing, so this is the cointegration space the whole run conditions on. Copy a file, rewrite this one dataset, and you have the same model over another candidate space |
| `beta`, `beta_init` | `(tt, k_beta*rank)`, `(1, k_beta*rank)` | A time-varying VEC: the whole path, and the state before the sample |
| `lambda` | `(1, n_lambda)` | A constant-loading factor model's free loadings, flat. `FavarNormalWishart` counts them differently, `(k - n_factors) * n_state` — see `VarSpec::n_favar_lambda()` |
| `lambda` | `(tt, n_lambda)` | `DfmTvpGamma` and `DfmTvpStochvol`: the whole path of the free loadings, held to exactly `tt` periods like `a` |
| `lambda_sigma_inv`, `lambda_init` | `(n_lambda, n_lambda)`, `(1, n_lambda)` | `DfmTvpGamma` and `DfmTvpStochvol`: the loadings' random walk innovation precision and the state before the sample. Their `a`, `a_sigma_inv` and `a_init` are the rows above at the transition's width, `n_factors * n_factors * p` |
| `v_sigma_inv` | `(1, n_factors)` | `DfmNormalGamma` and `DfmTvpGamma`: the factor innovation precisions, diagonal so stored flat |
| `v_sigma_inv` | `(n_state, n_state)` | `FavarNormalWishart`: a matrix, not a diagonal |
| `u_h`, `u_h_init` | `(k, tt)`, `(1, k)` | `DfmNormalStochvol` and `DfmTvpStochvol`: the series' log-volatility path and its start, in place of `u_sigma_inv` |
| `v_h`, `v_h_init` | `(n_factors, tt)`, `(1, n_factors)` | The same two models: the factors' log-volatility path and its start, in place of `v_sigma_inv` |

The two `*TvpDiscount` models read almost none of this group: no coefficient
start, no state precision, no error start, nothing being iterated. `/initial/beta`
above is the only entry either of them opens.

## `/posterior`

Written by the run, not by you. See `results.md`.

## Two conventions worth repeating

**Variable-selection positions are one-based.** `include` counts from 1, the way
R and the file format count, and is converted on read. A position below 1 is
refused outright rather than wrapping into an index nothing would catch.

**Results are written back in place, and a dataset that is already there is
unlinked first.** HDF5 does not reclaim that space, so a file grows a little
every time it is re-run — barely worth noticing on a file holding one model,
worth a periodic `h5repack` on one holding a hundred.
