# Reading the results

A run writes back into the same file, under `/posterior` (or under
`<group>/posterior` when `--group` was given).

## What gets written

| Dataset | Dataspace shape | Written by |
| --- | --- | --- |
| `/posterior/a/coeffs` | `(nparams, iterations)`, or `(nparams*tt, iterations)` when the coefficients drift | `coefficients` |
| `/posterior/a/lambda` | `(nparams, iterations)` | `coefficients`, when selection is on: the inclusion indicators |
| `/posterior/a/sigma` | `(nparams, iterations)` | `coefficients`, for a time-varying model: the random walk's innovation variances |
| `/posterior/psi/coeffs` | `(k*k, iterations)`, or `(k*k*tt, iterations)` for the four time-varying models, whose `Psi` moves with time | `coefficients`, when the covariance block is on. **Not `k(k-1)/2`**: each column is the whole lower-triangular `Psi`, vectorised, even though only the free elements were drawn |
| `/posterior/psi/lambda` | `(k*k, iterations)` | The same widening applies |
| `/posterior/psi/sigma` | `(k(k-1)/2, iterations)` | But **not** here: the random walk's innovation variances are one per *free* element |
| `/posterior/u_sigma_inv/coeffs` | `(k*k, iterations)`, or `(k*k*tt, iterations)` when the precision moves with time | `coefficients`. One vectorised precision matrix per draw. **This is the dataset every stage checks for** |
| `/posterior/u_sigma_inv/sigma` | `(k, iterations)` | `coefficients`, for every stochastic volatility model — `VarNormalStochvol`, `VarTvpStochvol`, `VecNormalStochvol`, `VecTvpStochvol`, `DfmNormalStochvol` and `DfmTvpStochvol`: the variance of the log-volatility innovations, one per variable. A forecast under `forecast_states = simulate` steps the volatility by it, so a posterior drawn before it was written has to be re-drawn or forecast with `hold` |
| `/posterior/v_sigma_inv/sigma` | `(n_factors, iterations)` | The same for a stochastic volatility factor model's factor innovations |
| `/posterior/u_omega_inv/coeffs` | `(k, iterations)` for the gamma models; `(k*tt, iterations)` for the stochastic volatility and `*Ald` models | `coefficients`, for the VARs and VECs other than the Wishart ones: the **diagonal** of the precision, which is the part actually drawn. For the two `*Ald` models it is the precision of the normal mixture the sampler draws through, `1 / (tau^2 w_t u_scale)` period by period with `tau^2 = 2 / (q (1 - q))` at `quantile` `q`, and moves with the latent scales `w_t` rather than with any volatility |
| `/posterior/u_scale/coeffs` | `(k, iterations)` | `coefficients`, for the two `*Ald` models: the scale of the asymmetric Laplace, one per equation. **`loglik` reads it**, since the density depends on it and not on the mixture precision |
| `/posterior/beta/coeffs` | `(k_beta*rank, iterations)`, widened over `tt` for a time-varying VEC | `coefficients`, for a VEC of positive rank |
| `/posterior/beta/rho` | `(1, iterations)` | `coefficients`, where a time-varying VEC put a prior on the state autoregression |
| `/posterior/lambda/coeffs` | `(k*n_factors, iterations)`; `(k*n_state, iterations)` for a FAVAR; widened over `tt` where the loadings drift | `coefficients`, for a factor model. **Not `n_lambda`**: each column is the whole `k x n_factors` loading matrix, `vec`'d column by column, fixed block included, even though only the free loadings were drawn |
| `/posterior/lambda/sigma` | `(n_lambda, iterations)` | `coefficients`, where the loadings drift |
| `/posterior/factors/coeffs` | `(n_factors*tt, iterations)` | `coefficients`, for a factor model. A FAVAR stores the **unobserved** factors alone |
| `/posterior/v_sigma_inv/coeffs` | `(n_factors, iterations)`, or `(n_factors*tt, iterations)` under stochastic volatility; `(n_state*n_state, iterations)` for a FAVAR | `coefficients`, for a factor model |
| `/posterior/forecast/forecasts` | `(h*k, iterations)`; `(h*(k + n_obs_factors), iterations)` for a FAVAR | `forecasts` |
| `/posterior/loglik` | `(tt, iterations)` | `loglik` |

A factor model's `u_sigma_inv` is diagonal by assumption, so it stores `k` per
draw rather than `k*k`, and `k*tt` where it moves with time.

## The `/posterior/forecast` group

`forecasts` is the only member `bayests` writes. The group is the place for
everything the forecast periods produce, and two more names are reserved for a
host that has the observations those periods realised:

| Dataset | Shape | Holds |
| --- | --- | --- |
| `forecasts` | `(h*k, iterations)` | The simulated paths |
| `errors` | `(h*k, iterations)` | Realised minus forecast |
| `loglik` | `(n, iterations)` | The log predictive density of what `/data/test/y` realised, `n` of its periods |

The leaves are named after what they hold rather than one of them being called
`draws`, because all three are draws. `/posterior/loglik` stays outside the
group and is a different statistic, not the same one over other periods: it
evaluates each in-sample observation under states that have already seen it,
which is what WAIC and PSIS-LOO want and what a forecast score must not do.

`forecasts` is the input to `errors`, so rewriting it invalidates them. Delete
the group rather than one dataset in it.

## The score

`loglik` is written by `forecasts` wherever the file carries `/data/test/y` and
the algorithm can be scored, which today is `VarNormalWishart` and
`VarNormalGamma`. One column per realised period, which may be fewer than `h`.

**Each column conditions on the realised observations before it**, not on the
path the forecast simulated. Column `i` is therefore the one step ahead
predictive density of period `T+i` given everything known up to `T+i-1`, and

```python
import h5py
import numpy as np

with h5py.File("model.h5", "r") as f:
    ll = f["/posterior/forecast/loglik"][:]      # (n, iterations)

# log of the mean of exp(), taken per period and shifted to keep it finite
top = ll.max(axis=1, keepdims=True)
lpd = np.log(np.mean(np.exp(ll - top), axis=1)) + top.ravel()
lpl = lpd.sum()                                  # log p(y*_{T+1..T+n} | data)
```

is the log predictive likelihood of the whole realised stretch: the joint,
factorised. Scoring against a simulated history instead would give the marginal
density of each horizon on its own — a defensible quantity, and a different one,
since marginals do not sum to a joint.

With the history realised rather than simulated the regressors of the scored
periods do not depend on the draw, so the score is the model's own pointwise log
likelihood over those periods, on a sample whose lag blocks are the realised
values. There is one density per algorithm, not two that can drift apart, and
`unit.predictive_score` pins the two against each other.

**What cannot be scored yet.** Every algorithm that lets its coefficients or its
error precision move over the horizon — the `Tvp` and `Stochvol` families, and
the factor models — needs each draw's state carried forward before the density
can be taken, which is the forecast's own machinery and is not wired into the
score. Those refuse rather than produce a number of the right size. `hold` does
not rescue them: it would score them under a model whose drift stops where the
sample does, which is a different density. A structural model refuses for good,
its regressors holding the contemporaneous observations and its density the
Jacobian of `A_0`. The two quantile models never reach it, having no forecast at
all. `bayests check` says which side of that line a file is on.

Before 0.3.0 the paths were a dataset at `/posterior/forecast` itself. Rerun
`bayests forecasts`: it replaces the old dataset with the group. The name is
freed, the space is not, so pass the file through `h5repack` to get it back.

## What a forecast does with drifting states

Every model whose coefficients or error precision move with time — the `Tvp`
and `Stochvol` algorithms — starts its forecast from the **last in-sample
period** of each draw. What happens after that depends on the model and on
`/model/forecast_states`.

**Every model with something that drifts** reads the attribute: the four
time-varying VARs (`VarTvpWishart`, `VarTvpGamma`, `VarTvpStochvol`,
`VarNormalStochvol`), the four time-varying VECs (`VecTvpWishart`,
`VecTvpGamma`, `VecTvpStochvol`, `VecNormalStochvol`) and the three drifting
factor models (`DfmNormalStochvol`, `DfmTvpGamma`, `DfmTvpStochvol`):

- `simulate`, the default: each draw's random walks take one step per horizon,
  before the observation that step generates. The coefficients step by
  `/posterior/a/sigma`, `Psi` by `/posterior/psi/sigma`, a factor model's free
  loadings by `/posterior/lambda/sigma`, and the log-volatilities by
  `/posterior/u_sigma_inv/sigma` and, for the factor innovations,
  `/posterior/v_sigma_inv/sigma`. A coefficient or `Psi` element that BVS
  excluded stays at zero, and so does a factor model's identifying block of
  loadings. `/posterior/forecast/forecasts` is then the predictive distribution of the
  estimated model, and its spread widens with the horizon.
- `hold`: the states stay at period `tt` for all `h` horizons. The spread
  carries parameter uncertainty and future errors at the period-`tt` precision,
  but not the drift the model allows over the horizon. Read it as conditional on
  no drift after the end of the sample. Every forecast was this before the
  attribute existed.

A file with no `forecast_states` reads as `simulate`. A stochastic volatility
model whose posterior lacks `/posterior/u_sigma_inv/sigma` (or, for a factor
model, `/posterior/v_sigma_inv/sigma`) — drawn by an older BayesTS — makes
`forecasts` exit 1 under `simulate`. Delete `/posterior` and run again, or set
the attribute to `hold`.

**A VEC simulates in levels.** Under `simulate` its loadings and short-run
coefficients step by `/posterior/a/sigma`, its cointegration vectors by their
state equation `beta_t = rho (I_r kron P_tau) beta_{t-1} + eta_t`,
`eta_t ~ N(0, I)` — with `/posterior/beta/rho` where the chain drew `rho` and
`/priors/beta/rho` otherwise, and `/priors/beta/p_tau` — and the level VAR they
imply is rebuilt at every horizon. `/data/forecast/x` is in the level layout
either way.

`VecKlgs2010`, `VecNormalWishart`, `VecNormalGamma`, `DfmNormalGamma` and
`FavarNormalWishart` have nothing that drifts, so the attribute changes nothing
for them.

## The `mcmc` attributes

Datasets under `/posterior/<block>/` carry `start`, `end` and `thin` attributes,
so an R session can hand them straight to `coda` as an `mcmc` object. They count
iterations after the burn-in: `start` is `/model/thin`, `end` is
`iterations * thin`, and `thin` is `/model/thin` again — so without thinning they
run from 1 to `iterations`.

`/posterior/forecast/forecasts` and `/posterior/loglik` carry them too, and have
since 0.3.0. Files written before that have them on the block datasets alone.

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
    fcst = f["/posterior/forecast/forecasts"][:]   # (h*k, iterations)

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
- `iterations` counts stored draws, not independent ones. A slowly mixing chain
  can report a posterior mean that a second `/model/seed` moves well beyond its
  apparent precision. Run it longer with `/model/thin` (see `recipes.md`) until
  two seeds agree.

Run `bayests check` on the file before the run, and read its dimensions and
warnings: that catches the fields a model never reads, which is how most
self-consistent wrong files show. Then check the shapes against the dimension
arithmetic in `SKILL.md` before reading anything into the values.
