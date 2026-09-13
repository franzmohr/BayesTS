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
| `algorithm` | string | *required* | Which of the twenty samplers runs. The only thing that decides |
| `k` | int | *required* | Endogenous variables. For a factor model, the number of **observed** series |
| `iterations` | int | *required* | Draws kept. Must be positive |
| `burnin` | int | *required* | Draws discarded before the first kept one |
| `p` | int | 0 | Lags. For a factor model, the lag order of the transition. For a VEC, a **level** order |
| `m` | int | 0 | Exogenous variables |
| `s` | int | 0 | Lags of the exogenous variables. For a VEC, a **level** order |
| `n` | int | 0 | Deterministic terms entering unrestricted |
| `h` | int | 0 | Forecast horizon. Absent or 0 means no forecast was asked for |
| `varsel` | string | `none` | `none`, `ssvs` or `bvs`. Any other spelling throws |
| `structural` | bool | false | Move the last `k(k-1)/2` entries of `a` to contemporaneous coefficients |
| `error` | string | `""` | Descriptive, except for `gamma+covar` and `sv+covar` — see below |
| `quantile` | double | 0.5 | The quantile an `*Ald` model estimates. Must lie in `(0, 1)`. Ignored by every other model |
| `rank` | int | 0 | A VEC's cointegration rank. Zero means "not a cointegrated model" |
| `k_beta` | int | 0 | Rows of the cointegration matrix. `rank` may not exceed it |
| `n_restricted` | int | 0 | A VEC's deterministic terms restricted to the cointegration space. Disjoint from `n` |
| `n_factors` | int | 0 | Unobserved factors. Zero for every model that is not a factor model |
| `n_obs_factors` | int | 0 | A FAVAR's observed factors. Needs `n_factors > 0` as well |

`error` is load-bearing for exactly two values, and the spelling is
model-specific: `gamma+covar` switches the covariance block on for the gamma
models, `sv+covar` for the stochastic volatility models. The twelve readers of
models with no `psi` block never compare it.

## `/model/priors/psi` — attribute

| Attribute | Meaning |
| --- | --- |
| `varsel` | The selection scheme for the covariance block alone, read by the four time-varying models that have one. The `/model` attribute governs the coefficients |

## `/data/train`

| Dataset | Dataspace shape | Contents |
| --- | --- | --- |
| `y` | `(1, tt*k)` | The response, already stacked: `vec(Y')`, all `k` variables of period 1, then period 2, and so on |
| `z` | `(nparams, tt*k)` | The SUR regressor matrix. On paper it is `(tt*k, nparams)`, one block of `k` rows per period |
| `w` | `(k_beta, tt)` | A VEC only: the error correction term. On paper `(tt, k_beta)` |
| `x` | `(n_x, tt)` | The regressors in the compact layout, one column per regressor on paper. Read by `VecKlgs2010` in place of `z` |
| `f_obs` | `(n_obs_factors, tt)` | A FAVAR only: the observed factors. On paper `(tt, n_obs_factors)`. The observed half of the state vector, not regressors |

`tt` is never stored. It is recovered as `len(y) / k`, and a `y` whose length is
not a multiple of `k` is refused.

A factor model has **no `z`**: its regressors are the unobserved factors, drawn
rather than given, so `/data/train/y` is all the data there is.

## `/data/forecast`

| Dataset | Dataspace shape | Contents |
| --- | --- | --- |
| `x` | `(n_x, h)` | Out-of-sample regressors in the compact layout, `(h, n_x)` on paper. **Required when `h > 0`** |
| `z` | `(k*n_x, h*k)` | The older SUR spelling of the same thing, `kron(x, I_k)`. Still read, and compacted on the way in |

A factor model needs neither: the horizon alone drives its forecast.

## `/priors`

| Group | Datasets | Read by |
| --- | --- | --- |
| `/priors/a` | `mu` `(1, nparams)`, `v_inv` `(nparams, nparams)` | Every constant-coefficient model |
| `/priors/a` | `shape`, `rate`, `mu`, `v_inv` | Every time-varying model: `shape`/`rate` on the innovation precision of the random walk, `mu`/`v_inv` on the state before the sample |
| `/priors/a` | `inprior` `(1, nparams)`, `include` (one-based ints), `tau0`, `tau1` | Added when `varsel` is on. `tau0`/`tau1` for `ssvs` only |
| `/priors/psi` | The same shapes at width `k(k-1)/2` | The models with a covariance block switched on |
| `/priors/u_sigma` | `df` (scalar), `scale` `(k, k)` | The Wishart models |
| `/priors/u_sigma` | `shape` `(1, k)`, `rate` `(1, k)` | The gamma models |
| `/priors/u_sigma` | `offset`, `sigma`, `shape`, `rate`, `mu` `(1, k)` and `v_inv` `(k, k)` | The stochastic volatility models |
| `/priors/u_scale` | `shape` `(1, k)`, `rate` `(1, k)` | The two `*Ald` models: the scale of the asymmetric Laplace |
| `/priors/beta` | `v_inv` (scalar), `p_tau_inv` `(k_beta, k_beta)` | The constant VECs: the cointegration space prior |
| `/priors/beta` | `mu`, `v_inv`, optional `rho`, optional `rho_min`/`rho_max`, optional `p_tau` `(k_beta, k_beta)` | The time-varying VECs — a state equation rather than a shrinkage. See below |
| `/priors/lambda` | `mu`, `v_inv` (and `shape`/`rate` where the loadings drift) | The factor models: the free loadings |
| `/priors/v_sigma` | `shape` `(1, n_factors)`, `rate` | The DFMs: the factor innovation precisions |
| `/priors/v_sigma` | `df`, `scale` `(n_state, n_state)` | `FavarNormalWishart`: its state innovation precision is a matrix |

`rho` is the autoregression of a time-varying VEC's cointegration state
equation. It must lie in `(0, 1]`; 1 is the random walk. Giving **both**
`rho_min` and `rho_max` turns it into a drawn parameter with that uniform prior,
and `rho` is then the value the chain starts at and must lie inside the support.
One end without the other is refused rather than guessed at.

## `/initial`

Starting values, at the widths the priors imply.

| Dataset | Shape | For |
| --- | --- | --- |
| `a` | `(1, nparams)` | Constant coefficients |
| `a` | `(tt, nparams)` | Time-varying coefficients: the whole path, one period per column on paper |
| `a_sigma_inv`, `a_init` | `(nparams, nparams)`, `(1, nparams)` | The random walk's innovation precision and the state before the sample |
| `a_lambda` | `(1, nparams)` | Inclusion indicators, when `varsel` is on |
| `psi`, `psi_sigma_inv`, `psi_init`, `psi_lambda` | The same at width `k(k-1)/2` | The covariance block |
| `u_sigma_inv` | `(k, k)` | Wishart and gamma error precisions. **A factor model's is `(1, k)`** — diagonal by assumption, so it is stored flat |
| `u_omega_inv` | `(k, k)` | The time-varying gamma models |
| `h`, `h_init` | `(k, tt)`, `(1, k)` | Stochastic volatility: the log-volatility path and its start |
| `w`, `u_scale` | `(k, tt)`, `(1, k)` | The `*Ald` models: the latent scales (strictly positive) and the scale of the asymmetric Laplace |
| `beta` | `(1, k_beta*rank)` | A constant VEC's cointegration matrix, flat as `vec(beta)` — the first column of `beta`, then the second |
| `beta`, `beta_init` | `(tt, k_beta*rank)`, `(1, k_beta*rank)` | A time-varying VEC: the whole path, and the state before the sample |
| `lambda` | `(1, n_lambda)` | A factor model's free loadings, flat |
| `v_sigma_inv` | `(1, n_factors)` | A DFM's factor innovation precisions, diagonal so stored flat |
| `v_sigma_inv` | `(n_state, n_state)` | `FavarNormalWishart`: a matrix, not a diagonal |
| `u_h`, `v_h` | | A DFM under stochastic volatility |
| `lambda_sigma_inv`, `lambda_init`, `a_sigma_inv`, `a_init` | | `DfmTvpGamma`, where `lambda` and `a` are paths |

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
