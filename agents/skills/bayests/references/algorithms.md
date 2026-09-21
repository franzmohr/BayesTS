# The twenty-two algorithms

`/model/algorithm` selects the sampler and is the only thing that does. The name
must be one of these twenty-two, spelled exactly.

Twenty of them are samplers. The two `*TvpDiscount` rows are not: their
posterior is closed form, and the section on them below says what that changes
in the file.

## Catalogue

| `algorithm` | Coefficients | Error precision | Covariance block | Variable selection | Structural | Forecast |
| --- | --- | --- | --- | --- | --- | --- |
| `VarNormalWishart` | Constant | Wishart | — | SSVS, BVS | refused | yes |
| `VarNormalGamma` | Constant | Independent gamma | optional | SSVS, BVS | yes | yes |
| `VarNormalStochvol` | Constant | Stochastic volatility | optional | BVS | yes | yes |
| `VarTvpWishart` | Random walk | Wishart | — | BVS | refused | yes |
| `VarTvpGamma` | Random walk | Independent gamma | optional, time-varying | BVS | yes | yes |
| `VarTvpStochvol` | Random walk | Stochastic volatility | optional, time-varying | BVS | yes | yes |
| `VarNormalAld` | Constant | Asymmetric Laplace | refused | BVS | yes | refused |
| `VarTvpAld` | Random walk | Asymmetric Laplace | refused | BVS | yes | refused |
| `VarTvpDiscount` | Random walk under a discount; not sampled | Discounted Wishart | — | refused | refused | yes |
| `VecNormalWishart` | Constant | Wishart | — | SSVS, BVS | refused | yes, in levels |
| `VecNormalGamma` | Constant | Independent gamma | optional | SSVS, BVS | yes | yes, in levels |
| `VecNormalStochvol` | Constant | Stochastic volatility | optional | BVS | yes | yes, in levels |
| `VecTvpWishart` | Random walk, beta included | Wishart | — | BVS | refused | yes, in levels |
| `VecTvpGamma` | Random walk, beta included | Independent gamma | optional, time-varying | BVS | yes | yes, in levels |
| `VecTvpStochvol` | Random walk, beta included | Stochastic volatility | optional, time-varying | BVS | yes | yes, in levels |
| `VecTvpDiscount` | Random walk under a discount, beta fixed; not sampled | Discounted Wishart | — | refused | refused | yes, in levels |
| `VecKlgs2010` | Constant, non-SUR | Wishart | — | refused | refused | yes, in levels |
| `DfmNormalGamma` | Constant loadings and transition | Independent gamma | — | refused | refused | yes |
| `DfmNormalStochvol` | Constant loadings and transition | Stochastic volatility | — | refused | refused | yes |
| `DfmTvpGamma` | Random walk, loadings and transition | Independent gamma | — | refused | refused | yes |
| `DfmTvpStochvol` | Random walk, loadings and transition | Stochastic volatility | — | refused | refused | yes |
| `FavarNormalWishart` | Constant loadings and transition | Wishart on the state innovations, gamma on the idiosyncratic errors | — | refused | refused | yes |

All seventeen VARs and VECs take exogenous regressors, deterministic terms and a
pointwise log likelihood — laid out for WAIC and PSIS-LOO, except on the
discounted pair, whose parameters are integrated out exactly and whose
`/posterior/loglik` is therefore one row rather than one per draw. The factor models have
one too, with the caveat in the DFM section below.

## The refusal table

Each of these throws when the file is read, before a chain is spent on it. They
are refusals with reasons, not unimplemented features.

| Combination | Why |
| --- | --- |
| `ssvs` with stochastic volatility | Not implemented; use `none` or `bvs` |
| `ssvs` with time-varying coefficients | Not implemented; use `none` or `bvs` |
| `ssvs` with either `*Ald` model | Not implemented; use `none` or `bvs` |
| Any selection on `VecKlgs2010` | Selection acts on the columns of the SUR matrix this sampler declines to build |
| Any selection on a factor model | Not implemented |
| `ssvs` with a non-zero prior mean at a selected position | The spike and slab are both centred at zero, and the indicators are scored as if they were. See the `varsel` section of `SKILL.md` |
| `ssvs` with a prior precision that couples a selected position to any other | Each indicator is drawn from its own coefficient alone, which is the posterior only when the selected coefficients are a priori independent of everything else |
| `ssvs` with `tau0 >= tau1` at a selected position | The spike must be the narrower component, or every indicator reads backwards |
| Selection positions inside a VEC's first `k*rank` entries | Excluding a loading is a change in the rank of `Pi`, which nothing downstream models. Restrict `include` to the positions after them |
| `structural` with a Wishart precision | `A_0` is not identified against an unrestricted `Sigma` |
| `structural` with a covariance block | Same: `Psi` is a second contemporaneous matrix doing the same job as `A_0`, and only their composition is pinned down |
| `structural` on a factor model | Not implemented |
| A covariance block on either `*Ald` model | `Psi` rotates the equations into each other, and the q-th quantile of a combination is not the combination of q-th quantiles |
| `h > 0` on either `*Ald` model | The h-step quantile is not the quantile of the iterated one-step quantiles |
| `quantile` outside `(0, 1)` | At 0 or 1 the loss has no minimiser and the scale is infinite |
| `n_obs_factors > 0` with `n_factors == 0` | A model with no unobserved factor is a VAR |
| `n_factors > k` on a factor model | The identifying block of the loadings needs one observed series per factor to pin the rotation with |
| A factor transition order `p` not below the number of periods | The transition regresses on `p` lags of the factors, and there has to be a sample left after the longest |
| `FavarNormalWishart` with `n_obs_factors == 0` | A model with no observed factor is a dynamic factor model; `DfmNormalGamma` estimates it |
| `FavarNormalWishart` whose state innovation Wishart prior has `df` below `n_factors + n_obs_factors` | A Wishart on an `n`-square matrix needs at least one degree of freedom per state element |
| Fewer than two periods on a time-varying or stochastic volatility model | The random walk differences the path against its own lag, so one period leaves nothing to difference |
| `rank > k_beta` | The rank cannot exceed the rows of the cointegration matrix |
| A VEC of positive `rank` with no regressors | The beta block is drawn inside the coefficient block, so without regressors it would never run |
| `y` whose length is not a multiple of `k` | A ragged sample would misalign silently |
| A `z` whose columns disagree with the dimensions (VEC) | The sampler would size everything off `z` and run to completion on a model the attributes do not name |
| `rho` outside `(0, 1]`, or a `rho` outside its own prior support | The state path would reverse sign or grow without bound |
| One of `rho_min`/`rho_max` without the other | Which end is missing changes the model rather than a detail of it |
| Any selection on either `*TvpDiscount` model | There are no draws for an inclusion indicator to be drawn alongside |
| `structural` on either `*TvpDiscount` model | The inverse Wishart posterior leaves `Sigma` unrestricted, so `A_0` is not identified |
| `delta_beta` or `delta_sigma` outside `(0, 1]` | Zero is not a model and above one the covariance shrinks each period instead of growing |
| `delta_sigma` with `1/(1 - delta_sigma) <= k` | The degrees of freedom settle there whatever the prior says, and below `k` the inverse Wishart is improper |
| A non-zero `burnin` or a `thin` other than 1 on a `*TvpDiscount` model | Nothing is iterated, so a file asking for either is asking for something that does not happen |
| `/data/train/z` and no `/data/train/x` on a `*TvpDiscount` model | These read the compact layout; the SUR matrix is the one thing they never build |

### Why `structural` needs a diagonal covariance

The data determine the reduced form and nothing else: the coefficients
`A_0^-1 A_i`, and the reduced-form covariance `Omega = A_0^-1 Sigma A_0^-T`,
which has `k(k+1)/2` free elements. `A_0` is unit lower triangular and
contributes `k(k-1)/2` of its own. Leave `Sigma` unrestricted — another
`k(k+1)/2` — and the structural side carries `k^2` parameters mapping onto
`k(k+1)/2`, so a `k(k-1)/2`-dimensional set of `(A_0, Sigma)` pairs produces the
same `Omega` and the likelihood is flat along it. Make `Sigma` diagonal and the
count is exactly `k(k+1)/2`: `A_0` and `diag(Sigma)` are then the LDL factor of
`Omega`, which is unique, and the model is the recursive SVAR.

Two things leave `Sigma` unrestricted, and the second is easy to miss. A Wishart
prior does, obviously. So does a covariance block:
`Sigma^-1 = Psi' Omega^-1 Psi` with `Psi` unit lower triangular and `Omega`
diagonal is a full `Sigma`.

It is rejected rather than warned about, because the reason to set the flag at
all is to read `A_0`, and a draw of it on the ridge is the prior plus wherever
the chain wandered.

## What each family needs in the file

### VAR

`k`, `p`, and whichever of `m`, `s`, `n` apply. `/data/train/y` and
`/data/train/z`. `/priors/a` and `/priors/u_sigma`. `/initial/a` and the error
starting values. `/data/forecast/x` when `h > 0`.

### VEC

Everything a VAR needs, plus `rank`, `k_beta`, `n_restricted`,
`/data/train/w`, `/priors/beta` and `/initial/beta`.

Each VEC differs from the VAR beside it in exactly one place: the first
`k*rank` regressors are `beta' w_{t-1}`, a function of the current draw rather
than data, so a Gibbs block for `beta` is added.

**`p` and `s` are level orders.** Differencing costs one lag block, so a VEC of
level order 1 carries no `Gamma` at all, and `n_x_vec = k*max(p-1, 0) + m*s + n`.

**All six forecast in levels**, by rewriting the draws as the level VAR they
imply. Their `/data/forecast/x` is in the level layout, not the differenced one
`/data/train/z` uses.

`VecKlgs2010` is the exception to the family: it reads `/data/train/x`, the
compact regressors, in place of `z`. The posterior is the same one
`VecNormalWishart` targets — the k equations of a VEC share their regressors, so
the posterior precision factors and the Gram product to form is `n_x` square
rather than `k*n_x` square. The choice between the two is a choice about cost.
What it gives up is variable selection, which acts on the columns of the matrix
it declines to build.

### The discounted pair

`VarTvpDiscount` and `VecTvpDiscount` are the matrix normal dynamic linear model
of West and Harrison (1997, ch. 16) with Uhlig's (1997) discounted Wishart on
the error precision. **They have an answer rather than a chain**, and everything
below follows from that.

Two `/model` attributes drive them, both in `(0, 1]` and both defaulting to 1:

- `delta_beta` discounts the coefficient covariance each period. `R_t =
  C_{t-1}/delta` is exactly the predicted covariance of a random walk whose
  innovation covariance is `((1 - delta)/delta) C_{t-1}`.
- `delta_sigma` discounts the Wishart, which is a stochastic volatility law of
  its own: multiplicative beta shocks to the precision, where the samplers'
  `Stochvol` is a Gaussian autoregression in the log variance.

Both at 1 is the conjugate posterior of a constant coefficient model, computed
one period at a time. It is a meaningful setting, not a no-op.

What the file looks like:

- **`/data/train/x`, not `/data/train/z`.** The compact layout, `tt` rows by one
  column per regressor. For `VecTvpDiscount` it holds the short-run blocks
  alone; the `rank` error correction columns are built from `w` and `beta`.
- **`/priors/a/mean` and `/priors/a/cov`, not `mu` and `v_inv`.** A matrix
  normal prior: `mean` is `n_x` by `k`, `cov` is `n_x` square and is the
  *regressor* side of a covariance, the equation side being the error covariance
  the Wishart prior already carries. Bringing `mu`/`v_inv` along instead is read
  as having no coefficient prior, and `bayests check` warns that nothing read
  them.
- **No `/initial` group for the VAR.** Nothing starts anywhere.
  `VecTvpDiscount` reads `/initial/beta`, but not as a starting value: it is the
  cointegration space the run *conditions on*, and it does not move.
- **`burnin` must be 0 and `thin` 1.** `iterations` keeps its name and means
  how many i.i.d. draws a forecast takes.
- **`seed` has nothing to repeat.** Estimation consumes no random numbers, so
  two runs agree to the bit. The forecast draws, and that is what a seed fixes.

What comes back is in results.md: a posterior, not draws, and deliberately **no
`/posterior/a/coeffs`**.

**`VecTvpDiscount` holds the cointegration space fixed, and cannot do
otherwise.** A VEC's measurement is `alpha_t beta_t' w_t`, two latent blocks
multiplying each other, which is why the three sampling VECs alternate two
smoother passes rather than filtering once. Beyond that, the innovation variance
of `beta_t` in those models is fixed at the identity because that is what pins
beta's scale against alpha's — a discount would replace it with something that
moves with the data, and alpha and beta would slide against each other along a
ridge. So use this model to ask how the *adjustment* to a given long-run
relation has moved, and `VecTvpWishart` and its two siblings to ask whether the
relation itself moved.

The fixed space buys back a number the samplers cannot give cheaply: the sum of
`/posterior/loglik` is the exact marginal likelihood of the sample given that
space, the rank and the two discounts. One pass each, so a grid over candidate
spaces, ranks or discounts is a list of files and no chain is run for any of
them.

### DFM

```
x_t = Lambda f_t + u_t,                u_t ~ N(0, U),  U diagonal,
f_t = sum_{j=1..p} A_j f_{t-j} + v_t,  v_t ~ N(0, V),  V diagonal,
```

`k` is the number of **observed** series, `n_factors` the unobserved ones, `p`
the lag order of the transition. `m`, `s`, `n`, `rank` and `n_restricted` are
all zero.

Three consequences, each visible in the file:

- **No `/data/train/z`.** The regressors are drawn rather than given, so
  `/data/train/y` is all the data there is and the horizon alone drives the
  forecast.
- **A whole `tt`-period factor path is part of every draw** and is written to
  `/posterior/factors/coeffs`. The forecast and the log likelihood read it back
  rather than re-filtering, which makes the reported likelihood the
  **conditional** one, `p(x_t | f_t, Lambda, U)`.
- **`Lambda` is identified only up to a rotation and a scale**, so its leading
  `n_factors` square block is fixed unit lower triangular and only
  `n_factors*(2k - n_factors - 1)/2` elements are drawn.

### FAVAR

```
x_t = Lambda_f f_t + Lambda_y y_t + e_t,
s_t = sum_j Phi_j s_{t-j} + v_t,        s_t = (f_t', y_t')'
```

`n_obs_factors` observed factors arrive in `/data/train/f_obs`. They are data
and not a state to draw, and they sit in the state vector regardless: the
transition is a VAR over both blocks jointly, and that coupling is the whole of
what separates this from the DFM.

**The identification differs from the DFM's, and the two cannot be mixed.** A
rotation `F -> C F` is invisible in the measurement if the loadings absorb it. A
DFM rules it out with a unit lower triangular loading block *and* a diagonal
`V`, which together admit only `C = I` — the uniqueness of an LDL
factorisation. A FAVAR has no diagonal `V` to offer, its `Q` being unrestricted
by design, so its leading loading block is the **identity** instead: the first
`n_factors` series of the panel are the factors plus idiosyncratic noise and
nothing else. That leaves `(k - n_factors) * n_state` free loadings.

The two restrictions cost the same. A unit triangle leaves
`n_factors(n_factors-1)/2` loadings free and spends that many on making `V`
diagonal; the identity spends them on the loadings and leaves `Q` free.

**A length check will not catch the mix-up.** The two counts differ by
`n(n-1)/2 - (k-n)*n_obs`, which is zero on a whole family of dimensions: with
one observed factor it vanishes at every `k = n(n+1)/2` — `(k, n, n_obs)` of
`(3,2,1)`, `(6,3,1)`, `(10,4,1)`, `(15,5,1)` and on up — and beyond that
wherever `(k-n)*n_obs` happens to hit `n(n-1)/2`, as `(4,3,3)` and `(7,4,2)` do.
At those dimensions a `/priors/lambda` laid out to the DFM's convention has
exactly the length a FAVAR asks for, passes, and is then read as something it is
not: the chain runs to completion and estimates a different model.

## Choosing between them

- **Wishart or gamma?** Wishart leaves the error covariance unrestricted, which
  is the usual default and which rules out `structural`. Gamma is diagonal,
  which is what makes contemporaneous coefficients identified; add
  `error = "gamma+covar"` to put a constant covariance block back on top of it.
- **Stochastic volatility or not?** `Stochvol` lets each error variance follow
  its own log random walk. It costs the SSVS option.
- **`ssvs` or `bvs`?** Where both are offered, they answer different questions.
  BVS zeroes the regressor, so exclusion is exact and the draws it stores are
  already the restricted coefficients — Korobilis (2013). What it pays for that
  is a dependence on the coefficient prior: an excluded coefficient is drawn
  from the prior while it is out, and the flatter that prior the harder it is
  for anything to get back in, so `/priors/a/v_inv` has to be tight enough to
  select against; `bayests check` and the run itself both warn when it is not.
  See the `varsel` section of `SKILL.md`. SSVS leaves the regressors alone and moves the prior
  precision between a `tau0` spike and a `tau1` slab; exclusion is approximate,
  and the spike carries the tightness itself. BVS is the one available on every
  model that selects at all.
- **`Normal` or `Tvp`?** `Tvp` makes the coefficients a random walk, one state
  path per coefficient. It costs the SSVS option and needs a `/priors/a` with
  `shape`/`rate` on the innovation precision beside `mu`/`v_inv` on the state
  before the sample.
- **A discount instead of a sampler?** `VarTvpDiscount` or `VecTvpDiscount`
  when the drift can be described by a discount factor rather than estimated,
  and when an exact marginal likelihood is worth more than an unrestricted state
  variance. They are the only models here with no burn-in, no thinning and no
  convergence to judge. One file is one pair of discounts, so a grid over them
  is a list of models — and cheap enough to be one, each costing a single pass.
- **A quantile?** `VarNormalAld` or `VarTvpAld`, with `quantile` in `(0, 1)`.
  One file is one quantile, so a grid of them is a list of models — which is
  also what lets the grid run in parallel. Read the posterior spread as a
  diagnostic only: the asymmetric Laplace is a working likelihood, and the
  intervals are not calibrated without the adjustment of Yang, Wang and He
  (2016), which is not applied.
