# Reading the results

A run writes back into the same file, under `/posterior` (or under
`<group>/posterior` when `--group` was given).

## What gets written

| Dataset | Dataspace shape | Written by |
| --- | --- | --- |
| `/posterior/a/coeffs` | `(nparams, iterations)`, or `(nparams*tt, iterations)` when the coefficients drift | `coefficients` |
| `/posterior/a/lambda` | `(nparams, iterations)` | `coefficients`, when selection is on: the inclusion indicators |
| `/posterior/a/sigma` | `(nparams, iterations)` | `coefficients`, for a time-varying model: the random walk's innovation variances |
| `/posterior/psi/coeffs` | `(k*k, iterations)` | `coefficients`, when the covariance block is on. **Not `k(k-1)/2`**: each column is the whole lower-triangular `Psi`, vectorised, even though only the free elements were drawn |
| `/posterior/psi/lambda` | `(k*k, iterations)` | The same widening applies |
| `/posterior/psi/sigma` | `(k(k-1)/2, iterations)` | But **not** here: the random walk's innovation variances are one per *free* element |
| `/posterior/u_sigma_inv/coeffs` | `(k*k, iterations)`, or `(k*k*tt, iterations)` when the precision moves with time | `coefficients`. One vectorised precision matrix per draw. **This is the dataset every stage checks for** |
| `/posterior/u_omega_inv/coeffs` | `(k, iterations)`; `(k*tt, iterations)` for the two `*Ald` models | `coefficients`, for the gamma and `*Ald` models: the **diagonal** of the precision, which is the part actually drawn |
| `/posterior/beta/coeffs` | `(k_beta*rank, iterations)`, widened over `tt` for a time-varying VEC | `coefficients`, for a VEC of positive rank |
| `/posterior/beta/rho` | `(1, iterations)` | `coefficients`, where a time-varying VEC put a prior on the state autoregression |
| `/posterior/lambda/coeffs` | `(k*n_factors, iterations)`; `(k*n_state, iterations)` for a FAVAR; widened over `tt` where the loadings drift | `coefficients`, for a factor model. **Not `n_lambda`**: each column is the whole `k x n_factors` loading matrix, `vec`'d column by column, fixed block included, even though only the free loadings were drawn |
| `/posterior/lambda/sigma` | `(n_lambda, iterations)` | `coefficients`, where the loadings drift |
| `/posterior/factors/coeffs` | `(n_factors*tt, iterations)` | `coefficients`, for a factor model. A FAVAR stores the **unobserved** factors alone |
| `/posterior/v_sigma_inv/coeffs` | `(n_factors, iterations)`, or `(n_factors*tt, iterations)` under stochastic volatility; `(n_state*n_state, iterations)` for a FAVAR | `coefficients`, for a factor model |
| `/posterior/forecast` | `(h*k, iterations)`; `(h*(k + n_obs_factors), iterations)` for a FAVAR | `forecasts` |
| `/posterior/loglik` | `(tt, iterations)` | `loglik` |

A factor model's `u_sigma_inv` is diagonal by assumption, so it stores `k` per
draw rather than `k*k`, and `k*tt` where it moves with time.

## The `mcmc` attributes

Datasets under `/posterior/<block>/` carry `start`, `end` and `thin` attributes,
so an R session can hand them straight to `coda` as an `mcmc` object.

`/posterior/forecast` and `/posterior/loglik` do **not** carry them.

## Orientation

Every dataset above is **one row per quantity and one column per draw** in HDF5
dataspace terms — the shapes in the table are what `h5py` reports. An R session
sees the transpose of each, which is draws in rows.

That is not a quirk to work around; it is why `/posterior/loglik` comes out
right for `loo` in R without transposing. R sees it as `(iterations, tt)`, which
is the `S x N` layout `loo::waic()` and `loo::loo()` expect.

## From Python

```python
import h5py
import numpy as np

with h5py.File("model.h5", "r") as f:
    k = f["/model"].attrs["k"]
    iterations = f["/model"].attrs["iterations"]

    a = f["/posterior/a/coeffs"][:]          # (nparams, iterations)
    loglik = f["/posterior/loglik"][:]       # (tt, iterations)

post_mean = a.mean(axis=1)                   # one number per coefficient
post_sd = a.std(axis=1, ddof=1)
q05, q95 = np.quantile(a, [0.05, 0.95], axis=1)
```

Reshaping a time-varying path: the quantity dimension holds the whole path,
period blocks of `nparams` values in order, so

```python
import h5py

with h5py.File("model.h5", "r") as f:
    k = f["/model"].attrs["k"]
    tt = f["/data/train/y"].shape[1] // k    # tt is never stored
    a = f["/posterior/a/coeffs"][:]          # (nparams*tt, iterations)

nparams = a.shape[0] // tt
a_path = a.reshape(tt, nparams, -1)          # period, coefficient, draw
```

A forecast is stacked the same way as `/data/train/y` — period-major, `k`
values per period:

```python
import h5py

with h5py.File("model.h5", "r") as f:
    k = f["/model"].attrs["k"]
    h = f["/model"].attrs["h"]
    fcst = f["/posterior/forecast"][:]       # (h*k, iterations)

fcst = fcst.reshape(h, k, -1)                # horizon, variable, draw
```

## From R

R's readers reverse the dimension order, so the same datasets arrive with draws
in rows and need no transposing for `coda` or `loo`:

```r
a <- rhdf5::h5read("model.h5", "/posterior/a/coeffs")   # iterations x nparams
colMeans(a)

loglik <- rhdf5::h5read("model.h5", "/posterior/loglik") # iterations x tt
loo::waic(loglik)
```

## What a green run does not prove

A run that exits 0 wrote *something*. It does not prove the sampler did what you
meant:

- A stage that had nothing to do returns 0 (see `pipeline.md`).
- A dataset another model owns is legitimately absent, so "absent" cannot be
  read as a failure on its own.
- A file whose dimensions are self-consistent but wrong estimates a different
  model and reports plausible numbers.

Run `bayests check` on the file before the run, and read its dimensions and
warnings: that catches the fields a model never reads, which is how most
self-consistent wrong files show. Then check the shapes against the dimension
arithmetic in `SKILL.md` before reading anything into the values.
