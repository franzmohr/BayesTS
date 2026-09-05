# BayesTS

[![CI](https://github.com/franzmohr/BayesTS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/franzmohr/BayesTS/actions/workflows/ci.yml)
[![Docs](https://github.com/franzmohr/BayesTS/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/franzmohr/BayesTS/actions/workflows/docs.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-%E2%89%A5%203.25-blue.svg)](CMakeLists.txt)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/franzmohr/BayesTS/actions/workflows/ci.yml)

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa?logo=github-sponsors&logoColor=white)](https://github.com/sponsors/franzmohr)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/franzmohr)

Bayesian time series estimation in C++: Gibbs samplers for vector autoregressive
models, driven either from the command line or from an embedded host such as an
R package.

A *model file* — one HDF5 file holding the data, the priors, the starting values
and a `/model/algorithm` name — is the unit of work. `bayests` reads it, runs the
sampler that name selects, and writes the posterior draws, forecasts and
pointwise log likelihood back into the same file. Nothing is passed on the
command line except which file to work on and which of those three results are
wanted, so a run is fully described by the file it is given and can be repeated
from it.

The numerics are deliberately isolated. `bayests_core` links neither HDF5 nor
HighFive, prints nothing, and touches no global state beyond the Armadillo RNG:
values in, values out. That is what lets the same sampler objects serve this
command line and a host that cannot allow a library to own the files or the
console — under RcppArmadillo the RNG is R's own, so `set.seed()` reaches the
draws without the sampler knowing. Progress reporting and cancellation arrive
through a `Reporter` interface instead of being written to `stdout`.

Requires a C++20 compiler and Fortran (for LAPACK). Built on
[Armadillo](https://arma.sourceforge.net/), [OpenBLAS/LAPACK](https://www.openblas.net/),
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) and
[HighFive](https://github.com/BlueBrain/HighFive), with optional OpenMP.

## Models

The `/model/algorithm` attribute in the model file selects the sampler; it is
the only thing that decides which one runs. Twenty are registered — eight VARs,
six VECs, one alternative implementation of a VEC, four dynamic factor models and
one factor augmented VAR:

| `algorithm` | Coefficients | Error precision | Variable selection |
| --- | --- | --- | --- |
| `VarNormalWishart` | Constant, normal prior | Wishart | SSVS, BVS |
| `VarNormalGamma` | Constant, normal prior | Independent gamma, optional constant covariance block | SSVS, BVS |
| `VarNormalStochvol` | Constant, normal prior | Stochastic volatility, optional covariance block | BVS |
| `VarTvpWishart` | Random walk | Wishart | BVS |
| `VarTvpGamma` | Random walk | Independent gamma, optional time-varying covariance block | BVS |
| `VarTvpStochvol` | Random walk | Stochastic volatility, optional time-varying covariance block | BVS |
| `VarNormalAld` | Constant, normal prior | Asymmetric Laplace at a chosen quantile; no covariance block | BVS |
| `VarTvpAld` | Random walk | Asymmetric Laplace at a chosen quantile; no covariance block | BVS |
| `VecNormalWishart` | Constant, normal prior; cointegration space prior on beta | Wishart | SSVS, BVS |
| `VecNormalGamma` | Constant, normal prior; cointegration space prior on beta | Independent gamma, optional constant covariance block | SSVS, BVS |
| `VecNormalStochvol` | Constant, normal prior; cointegration space prior on beta | Stochastic volatility, optional covariance block | BVS |
| `VecTvpWishart` | Random walk, beta included | Wishart | BVS |
| `VecTvpGamma` | Random walk, beta included | Independent gamma, optional time-varying covariance block | BVS |
| `VecTvpStochvol` | Random walk, beta included | Stochastic volatility, optional time-varying covariance block | BVS |
| `VecKlgs2010` | `VecNormalWishart` drawn without the SUR system | Wishart | none |
| `DfmNormalGamma` | Constant, normal prior on the loadings and on the factor transition | Independent gamma, on the idiosyncratic errors and the factor innovations | none |
| `DfmNormalStochvol` | Constant, normal prior on the loadings and on the factor transition | Stochastic volatility, on the idiosyncratic errors and the factor innovations | none |
| `DfmTvpGamma` | Random walk, the free loadings and the factor transition both | Independent gamma, on the idiosyncratic errors and the factor innovations | none |
| `DfmTvpStochvol` | Random walk, the free loadings and the factor transition both | Stochastic volatility, on the idiosyncratic errors and the factor innovations | none |
| `FavarNormalWishart` | Constant, normal prior on the loadings and on the state transition | Wishart on the state innovations; independent gamma on the idiosyncratic errors | none |

The fourteen VARs and VECs support exogenous regressors, deterministic terms and
a pointwise log likelihood laid out for WAIC and PSIS-LOO. All but the two
`*Ald` entries also forecast; those two refuse, for the reason below. Ten of them
take a structural (contemporaneous-coefficient) form: the four Wishart models
leave the error covariance unrestricted, which leaves `A_0` unidentified, so they
refuse it — see the structural paragraph at the end of this section.
`VecKlgs2010`, the four `Dfm*` entries and `FavarNormalWishart` are the exceptions
to the rest, each in its own way — see below.

**The two `*Ald` entries estimate a conditional quantile rather than a
conditional mean.** Minimising the quantile loss at `q` is maximising the
likelihood of an asymmetric Laplace distribution, and that distribution is a
scale mixture of normals — so with a latent scale per observation the model is,
conditionally, the weighted normal regression every other row here already is.
The quantile is a `/model` attribute; one file is one quantile, and a grid of
them is a list of models.

Three things they do not have, each on purpose. There is **no covariance block**:
`Psi` rotates the equations into each other, and the `q`-th quantile of a
combination of equations is not the combination of their `q`-th quantiles. There
is **no forecast**: the `h` step quantile is not the quantile of the iterated one
step quantiles, so a horizon is refused rather than answered with a path that
cannot be read as one. And the intervals are **not calibrated** — the asymmetric
Laplace is a working likelihood rather than a claim about the data, so the
posterior locates the quantile but its spread needs the adjustment of Yang, Wang
and He (2016), which is not applied. Read the spread as a diagnostic.

`VecKlgs2010` is the one entry that is not a model of its own. It is
`VecNormalWishart` — the cointegration sampler of Koop, León-González and
Strachan (2010) — written against the compact regressors `/data/train/x` rather
than the SUR matrix `/data/train/z` they kronecker up into. The k equations of a
VEC share their regressors, so the posterior precision of the coefficient block
factors,

```
z' kron(I_tt, Sigma^-1) z = kron(W_x' W_x, Sigma^-1)   for z = kron(W_x, I_k),
```

and the Gram product left to form is `n_x` square rather than `k n_x` square:
O(tt n_x²) against O(tt k³ n_x²), with no `(tt k) x (k n_x)` matrix built at all.
The posterior is the same one — `test/unit_vec_klgs_2010.cpp` draws both from a
single seed and compares them — so the choice between the two is a choice about
cost. What the compact form gives up is variable selection, which acts on the
columns of the matrix it declines to build; `validate()` rejects either scheme
rather than ignoring it.

The four `Dfm*` entries are the models here that are not regressions:

```
x_t = Lambda f_t + u_t,                u_t ~ N(0, U),  U diagonal,
f_t = sum_{j=1..p} A_j f_{t-j} + v_t,  v_t ~ N(0, V),  V diagonal,
```

for `k` observed series and `n_factors` unobserved ones, after Chan, Koop,
Poirier and Tobias (2019). Three things follow from the factors being
unobserved, and each shows up in the file format. The regressors are drawn
rather than given, so there is no `/data/train/z` — `/data/train/y` is all the
data there is, and the horizon alone drives the forecast. A whole `tt`-period
factor path is part of every draw and is written to `/posterior/factors/coeffs`;
the forecast and the log likelihood read it back rather than re-filtering, which
makes the reported likelihood the *conditional* one, `p(x_t | f_t, Lambda, U)`.
And `Lambda` is identified only up to a rotation and a scale, so its leading
`n_factors` square block is fixed unit lower triangular and only the remaining
`n_factors(2k - n_factors - 1)/2` elements are drawn — equation by equation,
since the rows have different widths.

The factor path is drawn whole by the same `chan_jeliazkov_2009` band sampler
listed above: its posterior precision is block banded of bandwidth `p`, so the
sweep is O(`tt` `n_factors`³) against the O(`tt`³ `n_factors`³) of forming the
`tt·n_factors` square precision and factorising it.

`DfmNormalStochvol` is that model with both `U` and `V` moving with time, each
element of both log-volatilities a random walk of its own. `Normal` still names
the coefficients: the loadings and the transition are constant, exactly as in
`VarNormalStochvol` against `VarTvpStochvol`. The two placements do different
work — volatility in `u_t` reweights the series that identify the factors, so a
series that was noisy early and quiet later stops contributing on the same terms
throughout; volatility in `v_t` is the common component's own, and it is what
keeps the `k` idiosyncratic variances from jointly absorbing a shock every series
felt at once. Both error groups take the `offset`/`shape`/`rate`/`mu`/`v_inv`
priors `VarNormalStochvol` takes, at the width of the series and of the factors
respectively, and `/posterior/u_sigma_inv/coeffs` widens from `k` per draw to
`k·tt`. The forecast holds both volatilities at their last in-sample value, as
every stochastic volatility model here does.

`DfmTvpGamma` moves the drift the other way round: the errors stay
homoskedastic and it is the coefficients that follow random walks — every free
loading and every element of `[A_1 … A_p]`, so `Lambda_t f_t` in the measurement
and `A_{j,t}` in the transition. `Tvp` names the coefficients here as it does
everywhere else, and it names both blocks; a model in which only the loadings
drifted would be a different one. What it is for is the assumption a factor
model makes most often and defends least — that a series' exposure to the common
component held over the whole sample. A constant-loading model has nowhere to
put a change in exposure except the idiosyncratic variance, which then carries it
as noise the series is credited with throughout. The identifying block still does
not drift: it is not drawn, and letting it move would let the rotation and the
scale of the factors wander over the sample, so a loading path would describe the
normalisation as much as the exposure.

Both coefficient groups then take what `/priors/a` takes for a VAR whose
coefficients drift — `shape`/`rate` on the variance of the state innovations
beside `mu`/`v_inv` on the state before the sample — and `/initial/lambda` and
`/initial/a` are paths rather than vectors, beside `lambda_sigma_inv`,
`lambda_init`, `a_sigma_inv` and `a_init`. On the way out,
`/posterior/lambda/coeffs` widens from `k·n_factors` per draw to
`k·n_factors·tt`, one vectorised loading matrix per period, and
`/posterior/a/coeffs` the same way; `lambda/sigma` and `a/sigma` carry the two
state variances. The forecast holds both at their last in-sample period, as every
time-varying model here does, while the pointwise log likelihood scores every
period under its own `Lambda_t`.

Two things in the numerics are worth knowing about. The factor path needed no new
algorithm — `chan_jeliazkov_2009` already took a measurement matrix and a
transition per period — but a drifting `Lambda` costs the shortcut it takes when
the measurement does not move: `Z'U⁻¹Z` is then formed `tt` times rather than
once, which with many observed series is the dominant cost of the assembly. And
the Kronecker identity `DfmNormalGamma` leans on for its transition, the
`n_factors` equations sharing their regressors so that the posterior precision
collapses to `kron(X X', V⁻¹)`, is a statement about a single coefficient vector
and does not survive the coefficients becoming a path; the transition path is
drawn against the SUR design `kron(x_t', I)` built per period.

`DfmTvpStochvol` is the two of those at once, and the widest model here: nothing
in it is held fixed but the normalisation. It takes `DfmTvpGamma`'s two
coefficient groups and `DfmNormalStochvol`'s two error groups, adds nothing of
its own to the file format, and draws nine Gibbs blocks against seven. Both
halves are there because a model with one of them has to explain the other with
what it has: a series whose loading fell looks like a series whose idiosyncratic
variance rose, and a period of common turbulence looks like a transition that
changed. Carrying both is what lets the data say which.

It is also the one model that hands `chan_jeliazkov_2009` all four of its
per-period arguments at once — a loading matrix, a measurement covariance, a
transition and a transition covariance, each its own per period. Two of the four
describe the transition and go in shifted by a period, because the band sampler
indexes the transition that *produces* state column `t` by `t - 1` while these
models index block `t` at period `t`; the other two describe the measurement and
go in as they are. That convention lives in one place, `draw_factor_path()`, which
serves all four dynamic factor models: an unshifted argument would estimate a
model whose transitions lag their own period, which is a different model and not
a broken one, so nothing would fail.
`FavarNormalWishart` is the dynamic factor model with observed variables in its
state:

```
x_t = Lambda_f f_t + Lambda_y y_t + e_t,   e_t ~ N(0, R),  R diagonal,
s_t = sum_{j=1..p} Phi_j s_{t-j} + v_t,   v_t ~ N(0, Q),   s_t = (f_t', y_t')',
```

for `k` panel series, `n_factors` unobserved factors and `n_obs_factors` observed
ones, after Bernanke, Boivin and Eliasz (2005). The observed factors are the
variables the model is about — a policy rate, output — and the panel is there to
measure the common component they move with. They sit *in* the state rather than
beside it because the transition is a VAR over both blocks jointly: the factors
respond to the policy variable and the policy variable responds back, and that
coupling is the model. They arrive in `/data/train/f_obs`, `tt` by
`n_obs_factors`.

Half the state being data is the whole of what is new, and it lands in three
places. The path draw *conditions* on the observed half instead of drawing it:
`chan_jeliazkov_2009_conditional` partitions the assembled precision into the
drawn rows and the observed ones, and `K_FF f = b_F - K_FY y` is the same band
one block size narrower. That is exact — adding the observed factors as
measurements with a small error variance is the usual shortcut and is not
available to a precision-based sampler at all, since observing something exactly
is infinite precision — and it keeps the information the observed factors' own
equations carry about the lagged factors, which a sampler that drew the factor
block alone would throw away.

The second is `Q`. A factor model's idiosyncratic `R` must stay diagonal, which
is why no `Dfm*` has a Wishart column; `Q` is a VAR's innovation covariance and
its cross block is the correlation between the factor innovations and the shock
to the observed variables — the one thing a FAVAR exists to measure. So the third
part of the name refers to `Q`, and `R` is gamma-diagonal throughout the family.

The third follows from the second. A rotation `F -> C F` is invisible in the
measurement if the loadings absorb it, and a DFM rules it out with a unit lower
triangular loading block *and* a diagonal `V` — together they admit only `C = I`.
A FAVAR has no diagonal `V` to offer, so its leading `n_factors` square loading
block is the **identity**, and the observed columns of those rows are zero: the
first `n_factors` panel series are the factors plus idiosyncratic noise and carry
no free loading at all. The two identifications cost the same `n(n-1)/2`
restrictions; the FAVAR spends them on the loadings rather than on `V`. Its
forecast is also the one here wider than `k`: `h * (k + n_obs_factors)`, the
panel of a horizon followed by the observed factors of the same horizon, which
are what the model is forecast for and have no other dataset to go in.

Three algorithms carry the implementation weight. The time-varying coefficient
paths are drawn as a single block with the simulation smoother of Durbin and
Koopman (2002), so the whole path moves at once rather than period by period.
Stochastic volatility uses the ten-component normal mixture of Omori et al.
(2007), which turns the non-linear measurement equation into a conditionally
linear one. The `*TvpStochvol` pair combines both. The third is the band sampler
of Chan and Jeliazkov (2009) described above, which draws the DFM factor path and,
through its conditional entry point, the FAVAR's.
The two `DfmTvp*` models are the ones that reach for the first and the third at
once — the band sampler for the factors, the simulation smoother for the loading
path of each series and for the transition — and `DfmTvpStochvol` reaches for all
three.

Each VEC differs from the VAR beside it in one place, the same place every time:
the first `k * rank` regressors are `beta' w_{t-1}`, so they are not data but a
function of the current draw, and a Gibbs block is added to draw beta itself.
The constant-coefficient three draw it as one vector, splitting each draw
between `alpha` and `beta` by the normalisation of Koop, Leon-Gonzalez and
Strachan (2010); the time-varying three draw it as a state path with the same
smoother the coefficients use, and rebuild the regressors period by period. All
six forecast in levels, by rewriting the draws as the level VAR they imply —
which means their `/data/forecast/z` is in the level layout, not the differenced
one `/data/train/z` uses.

SSVS is available only for the constant-coefficient models with a gamma or
Wishart error precision; the stochastic volatility and time-varying parameter
samplers reject it rather than silently ignoring it. In a VEC, selection may not
be applied to the loadings at the front of `a` in either scheme — excluding one
is a change in the rank of Pi, which nothing downstream models.

The structural form needs a **diagonal** error covariance, and the samplers
reject the combinations where it does not have one. `A_0` is unit lower
triangular with `k(k-1)/2` free elements, and the data determine only the
reduced form — in particular `Omega = A_0^-1 Sigma A_0^-T`, which has `k(k+1)/2`.
Against a diagonal `Sigma` that is `k(k-1)/2 + k = k(k+1)/2` exactly, and `A_0`
with `diag(Sigma)` is the LDL factor of `Omega`, which is unique: the recursive
SVAR. Against an unrestricted `Sigma` it is `k^2`, leaving a `k(k-1)/2`
dimensional set of `(A_0, Sigma)` pairs that fit identically. Two things here
leave `Sigma` unrestricted — a Wishart prior, and a covariance block, since
`Psi` is then a second contemporaneous matrix doing the same job — so
`structural` is available with the `gamma` and `sv` error specifications and no
covariance block, and refused otherwise. With `sv` it is more than exactly
identified: the volatility moving over the sample identifies `A_0` through
heteroskedasticity.

## Usage

```
bayests <command> <path> [flags...]
```

`<path>` is either a single HDF5 model file or a directory, which is walked
recursively and every HDF5 file in it processed. Results are written back into
each file in place.

| Command | What it computes |
| --- | --- |
| `posterior` | All three of the below, in the order coefficients, log likelihood, forecasts |
| `coefficients` | Posterior draws of the coefficients and the error precision |
| `forecasts` | One forecast path per posterior draw |
| `loglik` | Pointwise log likelihood, draws × periods |

Every command takes `--group <path>`, the group a model's tree hangs under inside
its file. Without it the tree hangs at the root of the file, which is where a
file holding a single model puts it; with it every path in the table below is
read and written under that group instead, so one file can hold several models
side by side. In directory mode the same group is looked for in every file.

Both spellings of the flag are accepted: `--group /models/3` and
`--group=/models/3`.

`posterior` additionally runs all three steps by default, and any of them can be
switched off with `--no-coefficients`, `--no-forecasts` and `--no-loglik`.
`coefficients`, `forecasts` and `loglik` take no step flags.

```bash
# Everything
bayests posterior model.h5

# Draw the coefficients, but skip the forecast and the log likelihood
bayests posterior model.h5 --no-forecasts --no-loglik

# The model under /models/3 of a file that holds several
bayests posterior models.h5 --group /models/3

# Every model file below models/, forecasts only
bayests forecasts models/
```

`forecasts` and `loglik` read coefficient draws that are already in the file, so
they are the way to add a result to a model that has been sampled once, without
re-running the sampler.

The program reports its thread counts on startup and exits 1 on a bad path, a
non-HDF5 file, an unknown `algorithm`, or a run that started and could not
finish — a model file the sampler rejects, or a `forecasts` or `loglik` asked
for before the coefficients have been drawn. Having nothing to do is not
failing and exits 0: output that is already there, and a forecast on a model
with no horizon, the two quantile models included. In directory mode a file
that fails is
reported on `stderr` and the walk continues to the next one, but the exit status
is 1 if any file failed — so a script driving a directory of models can tell
whether everything in it was processed.

A command line that cannot be acted on at all — no arguments, a first argument
that is not one of the four commands, or a `--group` with no value or one that
cannot name an HDF5 group — prints the reason and exits 2. The two codes are
worth keeping apart in a script: 1 means the run started and something in it
failed, 2 means it never started. A `--group` that is well formed but names
nothing in the file is the first kind, not the second: the command line was
actionable, the file just did not hold that model.

### The model file

Groups and datasets the readers look for, as paths within one model. They are
paths from the root of the file unless `--group` was given, in which case they
hang under that group instead. Only `/model`'s `algorithm`, `k`, `iterations` and
`burnin` are required — everything else is read through a default, so a file
written for a simpler model still describes a valid one.

| Location | Contents |
| --- | --- |
| `/model` (attributes) | `algorithm`, `k` endogenous variables, `iterations` kept, `burnin` discarded; optional `p`, `m`, `s`, `n` (lags, exogenous variables, their lags, deterministic terms), `h` forecast horizon, `varsel` (`none`, `ssvs`, `bvs`), `structural`, `error`; `rank`, `k_beta`, `n_restricted` for a VEC, `n_factors` for a factor model and `n_obs_factors` for a FAVAR |
| `/data/train/y`, `/data/train/z` | Endogenous variables and the regressor matrix, `(tt k)` rows by `nparams` columns |
| `/data/train/w` | A VEC's error correction term, `tt` rows by `k_beta` columns |
| `/data/train/x` | The regressors in the compact layout, `tt` rows by one column each; read by `VecKlgs2010` in place of `z` |
| `/data/train/f_obs` | A FAVAR only: the observed factors, `tt` rows by `n_obs_factors` columns. The observed half of the state vector, not regressors |
| `/data/forecast/z` | Out-of-sample regressors; required when `h` > 0 |
| `/priors/a`, `/priors/psi` | Normal prior `mu` and `v_inv` for the coefficients and the covariance block, plus `inprior`, `include`, and `tau0`/`tau1` for SSVS |
| `/model/priors/psi` (attribute) | `varsel` for the covariance block on its own, read by the four time-varying models that have one; the `/model` attribute above governs the coefficients |
| `/priors/u_sigma` | `shape`/`rate` for gamma precisions, `df`/`scale` for Wishart, `mu`/`v_inv`/`sigma`/`offset` for stochastic volatility |
| `/priors/beta` | A VEC only: `p_tau_inv`, the prior precision of the cointegration space, and for the time-varying three `mu`/`v_inv` over beta before the sample and the state autoregression `rho` |
| `/priors/lambda`, `/priors/v_sigma` | A factor model only: normal `mu`/`v_inv` over the free loadings, and `shape`/`rate` for the factor innovation precisions. Under `DfmTvpGamma` the loading group is a state equation instead, `shape`/`rate` on the innovation variance beside `mu`/`v_inv` on the state before the sample, and `/priors/a` reads the same way. Under `FavarNormalWishart` the `v_sigma` group is `df`/`scale` rather than `shape`/`rate`, its state innovation precision being a matrix |
| `/initial/…` | Starting values: `a`, `psi`, `u_sigma_inv`, `u_omega_inv`, `h`, the `*_init` states and the `*_lambda`, `*_sigma_inv` blocks the samplers that need them read; `beta` for a VEC; `lambda`, `v_sigma_inv` and, under stochastic volatility, `u_h`/`v_h` for a DFM; `lambda` and `a` are paths under `DfmTvpGamma`, beside `lambda_sigma_inv`, `lambda_init`, `a_sigma_inv` and `a_init`; under `FavarNormalWishart` `v_sigma_inv` is an `n_state` square matrix rather than a diagonal |
| `/posterior/…` | Written by the run: `a/coeffs`, `a/lambda`, `a/sigma`, the matching `psi/…`, `u_sigma_inv/coeffs`, `u_omega_inv/coeffs`, `forecast` and `loglik`; `beta/coeffs` for a VEC; `lambda/coeffs`, `factors/coeffs` and `v_sigma_inv/coeffs` for a factor model, plus `lambda/sigma` where the loadings drift. Under `FavarNormalWishart` `factors/coeffs` holds the unobserved factors alone, `v_sigma_inv/coeffs` is `n_state` squared per draw, and `forecast` is `h * (k + n_obs_factors)` rows rather than `h * k` |

Two conventions are worth knowing before writing a file by hand. The first is
which way round the matrices are stored, and it is worth stating twice, because
the answer depends on which language is asking. In HDF5's own dataspace terms
every dataset above is **one row per quantity and one column per draw** — h5py
reports `/posterior/a/coeffs` of a 12-parameter, 80-draw model as `(12, 80)`,
and `/data/train/z` is `nparams × (tt k)`. R's HDF5 readers reverse the
dimension order, R being column-major where HDF5 is row-major, so an R session
sees the transpose of each: **draws in rows**, which is what R and `coda` expect
of an `mcmc` object, and why the `start`/`end`/`thin` attributes are written
alongside. Inside the samplers a draw is one column. A reader coming from
Python or C should expect the dataspace orientation, not R's.

Variable-selection positions are stored **one-based**, the way R and the file
format count, and converted on read. The `error` attribute is what turns the
covariance block on, and the spelling that does it is model-specific:
`gamma+covar` for the gamma models and `sv+covar` for stochastic volatility. Those
two are the only values that switch anything on. A model with no psi block —
the Wishart family, whose covariance is the Wishart precision alone, the two
quantile models, `VecKlgs2010` and the factor models — has its reader compare
against no spelling at all, so there the attribute describes the file without
being read back.

## Building from source

The build needs nothing but CMake and the four dependencies above. The toolchain
notes below are for Windows, which is the platform the project is developed and
exercised on; the runtime-bundling step in `CMakeLists.txt` is Windows-specific,
but nothing else is.

### Windows

**Compiler**

On my Windows set-up I use the `x64-mingw-dynamic` compiler, which I downloaded via MSYS2 using:

`pacman -S --needed base-devel mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-toolchain`

Make sure you also install gfortran, which is needed for LAPACK.

[https://code.visualstudio.com/docs/cpp/config-mingw] is an informative guide.

**Visual Studio Code**

Should not require any further tweaking, if the CMakePresets.json is configured to your system.

**Ninja**

Using Ninja as generator. Make sure that its directory is included in the environment variable `Path`.

**vcpkg for package management**

Make sure that its directory is included in the environment variable `Path`. Also add the environment variable `VCPKG_ROOT`, which should also include the directory.

Note that the `hdf5` library has problems to build. So I installed the binary from the website.

General information: [https://learn.microsoft.com/en-us/vcpkg/] and [https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-vscode?pivots=shell-powershell]

**CMake 4.2.1**

Make sure that its directory is included in the environment variable `Path`.

`CMakePresets.json` is committed and should stay machine-independent. Put your own
compiler and dependency paths in a `CMakeUserPresets.json` next to it, which is
git-ignored, and inherit from the `default` preset:

```json
{
    "version": 8,
    "configurePresets": [
        {
            "name": "my-windows-default",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_C_COMPILER": "gcc",
                "CMAKE_CXX_COMPILER": "g++",
                "CMAKE_Fortran_COMPILER": "gfortran",
                "CMAKE_PREFIX_PATH": "D:/vcpkg/installed/x64-mingw-dynamic;D:/HDF5/2.0.0;D:/HighFive",
                "HIGHFIVE_DIR": "D:/HighFive",
                "VCPKG_HOST_TRIPLET": "x64-mingw-dynamic",
                "VCPKG_TARGET_TRIPLET": "x64-mingw-dynamic"
            }
        }
    ]
}
```

Things you will want to adjust:
* The compiler variables. No change needed if you also use `gcc` and `g++`.
* `VCPKG_HOST_TRIPLET` and `VCPKG_TARGET_TRIPLET`, which seem to be `x64-windows` by default and, thus, assume that you have Visual Studio (not Visual Studio Code) installed.
* `CMAKE_PREFIX_PATH`, so that Armadillo, HDF5 and HighFive are found where `vcpkg` (or the installer) put them.

**Doxygen (optional, for the API documentation)**

Only needed to build the `docs` target. Install it from
[https://www.doxygen.nl/download.html] or, under MSYS2, with

`pacman -S mingw-w64-x86_64-doxygen mingw-w64-x86_64-graphviz`

Graphviz is optional and only adds the class and include graphs. Make sure both
directories are on `Path`, or point CMake at the installation with
`-DDOXYGEN_EXECUTABLE=...`. See the *API documentation* section below.

### Linux

Everything comes from the distribution's own packages; no vcpkg is needed. On
Debian or Ubuntu:

```bash
sudo apt install cmake ninja-build gfortran libarmadillo-dev libhdf5-dev libopenblas-dev
```

HighFive is the one exception. The archive is still on the 2.x line and this
project is written against 3.x, so clone the tag recorded in
`.github/highfive-version` and point the build at it — it is header-only, so
there is nothing to install:

```bash
git clone --depth 1 --branch "$(cat .github/highfive-version)" \
    https://github.com/BlueBrain/HighFive.git ~/src/highfive

cmake -S . -B build/bin/linux -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIGHFIVE_DIR="$HOME/src/highfive" \
    -DBAYESTS_BUILD_DOCS=OFF
cmake --build build/bin/linux
```

Neither `libarmadillo-dev` nor `libhdf5-dev` installs a CMake config package, so
configuration falls back to `FindArmadillo` and `FindHDF5` and builds the
imported targets from what those report. The configure output says which route
each dependency took:

```
-- Armadillo: FindArmadillo module, version 12.6.7, libraries /usr/lib/x86_64-linux-gnu/libarmadillo.so
-- HDF5 target: HDF5::HDF5
```

A `-- Armadillo: CONFIG package` line instead means a config package was found —
from vcpkg, or a source build — and that is used in preference.

### Configure and build

```bash
# Configure -- run from the repository root
cmake --preset my-windows-default

# Build
cmake --build build/bin/my-windows-default
```

**Build tree location**

The presets set `binaryDir` to `${sourceDir}/build/bin/${presetName}`, so each
one gets its own tree under `build/` and they do not overwrite each other.
`build/` is git-ignored and excluded from the source package.

Configuring in the repository root itself is refused with a `FATAL_ERROR`: an
in-source build scatters `CMakeCache.txt` and `CMakeFiles/` among the sources,
where they shadow every later out-of-source configure. Point `-B` at a
subdirectory, or use a preset, which already does.

### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `BAYESTS_BUILD_DOCS` | `ON` | Add the `docs` target when Doxygen is found |
| `BAYESTS_BUILD_TESTS` | `ON` | Build the regression harnesses in `test/` |
| `BAYESTS_NATIVE_ARCH` | `OFF` | `-march=native`; not redistributable, see *Packaging* |
| `BAYESTS_BUNDLE_RUNTIME_DEPS` | `ON` (Windows) | Copy the runtime DLLs next to the executable |
| `BAYESTS_RECORDED_FIXTURES` | *(empty)* | Recorded model files, `;`-separated, each registering an extra golden test; the generated suite runs without them |
| `BAYESTS_RUNTIME_DEP_DIRS` | *(empty)* | Extra directories to resolve the bundled runtime libraries from, see *Packaging* |

## Tests

`ctest` is the entry point, and needs no arguments or environment beyond a
completed build:

```bash
cmake --build build/bin/my-windows-default
ctest --test-dir build/bin/my-windows-default --output-on-failure
```

Each `fixture.*` test writes one model file into
`build/bin/<preset>/test/fixtures/`, and the `golden.*` test after it runs all
three entry points over that file. They are paired through CTest's
`FIXTURES_SETUP`/`FIXTURES_REQUIRED`, so naming a single golden test regenerates
just the input it needs:

```bash
ctest --test-dir build/bin/my-windows-default -R golden.VarTvpGamma-covar
```

Fixtures are written into the build tree, never into the source tree: they are
derived data and a full set runs to hundreds of megabytes. The tests pin
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` themselves, because the
samplers are only reproducible single-threaded, and on Windows they prepend the
dependency DLL directories to `PATH` so `ctest` works in a shell that has not
been set up by hand.

**What these tests do and do not check.** `bayests_golden` prints a fingerprint
for every posterior dataset, and fails a fixture that throws or whose run
produced nothing — no draws, no log likelihood, or no forecast where the horizon
is positive. It does not compare the numbers against an expectation, so the
suite is a smoke test across every sampler and code path rather than a numerical
regression test. Expected fingerprints are deliberately not checked in: they
shift in the last digits with the compiler, the BLAS implementation and — when
`BAYESTS_NATIVE_ARCH` is on — the CPU, so a committed baseline would fail on
every machine but the one that recorded it. To use them as a regression check,
record the output before a change and diff it against the output after, on the
same machine and with the same build:

```bash
test/record_fingerprints.sh build/bin/my-windows-default test/baselines/before.txt
# ... make the change, rebuild ...
test/record_fingerprints.sh build/bin/my-windows-default test/baselines/after.txt
test/diff_fingerprints.sh test/baselines/before.txt test/baselines/after.txt
```

`record_fingerprints.sh` keeps the fixture headers and the fingerprint lines of a
`ctest -V` run and drops the progress bars, timings and absolute paths that
differ between two runs of an unchanged build. `diff_fingerprints.sh` then
reports which *fixtures* moved, and prints one fixture's lines when given its
name as a third argument. Prefer it to a plain `diff`: a recording of the full
suite is around 115 KB and the `-V` run behind it close to 1.1 MB, so a change to
a shared algorithm produces a line diff longer than anyone reads, while the list
of fixtures it moved is short and is what says whether the blast radius matches
the intent.

`test/baselines/` is the place for these: it is git-ignored except for its own
`.gitignore`, and unlike anything under `build/` it survives a `rm -rf build`,
which a baseline recorded before a rebuild has to. Date the files or name them
for the change they precede — a baseline recorded before a *build flag* change
still diffs cleanly enough to look meaningful, which makes a stale one worse
than none.

**Coverage.** All twenty samplers are covered from a clean clone:
`test/make_model_fixture.cpp` writes a model file for every one of them, and the
suite depends on no data outside the repository.

What a generated file cannot be is a real sample under a real prior — the
numbers it invents only have to be admissible and reproducible. Recorded model
files fill that in, and any number of them can be added as extra golden tests
with `BAYESTS_RECORDED_FIXTURES`, each dispatched on its own
`/model/algorithm`. None is checked in: `*.h5` is gitignored, and a recording is
mostly posterior draws.

## Packaging

Packages are produced with CPack from the `release` preset. Like the
development presets, `release` carries no compiler or dependency paths of its
own — inherit from it in your `CMakeUserPresets.json`, listing it *before* your
own preset so its `Release` build type wins:

```json
{
    "name": "my-windows-release",
    "inherits": ["release", "my-windows-default"]
}
```

Then configure, build and package:

```bash
cmake --preset my-windows-release
cmake --build --preset my-windows-release
cpack --preset my-windows-release
```

The archive and its SHA-256 checksum land next to the build, in
`build/bin/my-windows-release/`:

```
BayesTS-0.0.1-Windows-AMD64.zip
BayesTS-0.0.1-Windows-AMD64.zip.sha256
```

It contains `bin/bayests.exe`, the runtime libraries it needs, and
`share/doc/BayesTS/` with the licence and this README. Unpack it anywhere and
`bayests` runs without MSYS2, vcpkg or HDF5 installed.

`cmake --build ... --target package` does the same thing if you prefer it to
`cpack`, and `cmake --install ...` writes the same layout into
`CMAKE_INSTALL_PREFIX` without producing an archive.

**Windows installer**

`ZIP` is the only generator configured unconditionally, because it needs
nothing beyond CMake. When [NSIS](https://nsis.sourceforge.io/) is present an
installer is built alongside it:

```
BayesTS-0.0.1-Windows-AMD64.exe
BayesTS-0.0.1-Windows-AMD64.exe.sha256
```

It installs into `%ProgramFiles%\BayesTS`, shows the licence, offers to add
`bayests` to `PATH`, and removes a previous version before installing over it.
Elevation is required, and uninstalling removes the install directory, the
registry entries and the `PATH` entry again.

For an unattended install, `/S` runs the installer without a UI and `/D=` picks
the directory — NSIS requires `/D` last and unquoted, even when the path
contains spaces:

```bat
BayesTS-0.0.1-Windows-AMD64.exe /S /D=C:\tools\BayesTS
"C:\tools\BayesTS\Uninstall.exe" /S
```

A silent install leaves `PATH` alone: the page that asks about it cannot be
answered without a UI, and the script adds nothing unless it is. Put the `bin`
directory on `PATH` yourself if an unattended install needs it.

NSIS does not have to be on `PATH`: configuration also reads the registry key
its installer writes — under both `SOFTWARE\NSIS` and `SOFTWARE\WOW6432Node\NSIS`,
since NSIS is a 32-bit application and a 64-bit CMake reading only the former
finds nothing — and falls back to the default install directories. Point at it
by hand if it is somewhere unusual:

```bash
cmake --preset my-windows-release -DBAYESTS_MAKENSIS_EXECUTABLE="D:/NSIS/makensis.exe"
```

Configuration reports which generators you get, so check for this line:

```
-- NSIS found: D:/NSIS/makensis.exe - 'package' also builds an installer
```

If it says NSIS was not found, only the ZIP is built. A generator whose tool is
missing would otherwise fail the whole `package` target, which is why the list
is assembled at configure time rather than fixed.

**Source package**

```bash
cpack --config build/bin/my-windows-release/CPackSourceConfig.cmake
```

Produces `BayesTS-0.0.1-src.zip` and `.tar.gz`. The ignore list drops the build
tree, `.git/`, `CMakeUserPresets.json` and every `*.h5`, since model files are
derived data and run to hundreds of megabytes.

**Snap package (Linux)**

`snap/snapcraft.yaml` builds `bayests` as a strictly confined snap for amd64 and
arm64. It needs [snapcraft](https://snapcraft.io/snapcraft) and LXD:

```bash
sudo snap install snapcraft --classic
sudo snap install lxd && sudo lxd init --auto

snapcraft                                       # builds for the host architecture
sudo snap install --dangerous ./bayests_0.0.1_amd64.snap
bayests
```

`snapcraft remote-build` builds both architectures on Launchpad instead, which
is the only practical way to produce the arm64 package from an x86 machine. To
publish, register the name once and upload:

```bash
snapcraft login
snapcraft register bayests
snapcraft upload --release=edge ./bayests_0.0.1_amd64.snap
```

The version is not written in `snapcraft.yaml`. It is read out of
`project(VERSION ...)` during the pull step, by the same expression
`release.yml` uses to check a tag, so the snap cannot claim a version the source
does not.

Dependencies come from the base's archive rather than vcpkg — this is the build
the `FindArmadillo` and `FindHDF5` fallbacks exist for. HighFive is cloned from
the tag in `.github/highfive-version`, which is also spelled out as `source-tag`
in `snapcraft.yaml`: **bumping the pin means editing both files.** Nothing joins
them automatically, because reading the version out of the headers cannot
distinguish the 3.0.0 betas from each other.

Confinement is strict, so `bayests` reads and writes below `$HOME` through the
`home` interface and nothing else. Data on another filesystem needs the
`removable-media` interface, which is not connected automatically:

```bash
sudo snap connect bayests:removable-media
```

The `stage-packages` sonames — `libarmadillo12`, `libhdf5-103-1t64` — are those
of `core24`, meaning Ubuntu 24.04, and are what a base bump breaks first. A name
that no longer exists fails at pull with *package not found*; `ldd
prime/usr/bin/bayests` and `dpkg -S` on what it names give the replacements.

**Portable binaries and `-march=native`**

The numeric libraries — `bayests_core` and `algorithms`, everything that links
`bayests_numeric_flags` — are compiled with `-march=native -mtune=native` only
when `BAYESTS_NATIVE_ARCH` is `ON`. Tuning for the machine that happens to be
compiling produces a binary that faults with an illegal instruction on any CPU
without the same instruction set, so the flag is off by default and the
`release` preset keeps it off. The `default` development preset turns it on:
those builds never leave your machine. Do not switch it on for anything you
intend to hand to someone else.

**Bundled runtime libraries**

`BAYESTS_BUNDLE_RUNTIME_DEPS` (on by default on Windows) copies the DLLs the
executable loads — libstdc++, libgomp, libgfortran, OpenBLAS, LAPACK, HDF5 and
the rest — into `bin/`. CMake resolves them by searching the compiler's own
`bin/` and the `bin/` of every `CMAKE_PREFIX_PATH` entry; it does *not* consult
`PATH`. If a library lives somewhere else, the install step fails with an
unresolved dependency, and the fix is to name the directory:

```bash
cmake --preset my-windows-release -DBAYESTS_RUNTIME_DEP_DIRS="D:/somewhere/bin"
```

The generated documentation is not packaged. It is built on demand from the
`docs` target, described next.

## API documentation

The API reference is generated from the doc comments in the source with
[Doxygen](https://www.doxygen.nl/). It is *not* built as part of a normal build,
so Doxygen is not required to compile the project.

```bash
# Configure as usual -- the docs target appears if Doxygen was found
cmake --preset my-windows-default

# Generate the documentation
cmake --build build/bin/my-windows-default --target docs
```

Open `build/bin/my-windows-default/docs/html/index.html` in a browser.
This README serves as the landing page.

During configuration you should see:

```
-- Doxygen 1.x.y found - build the 'docs' target; output in .../docs/html/index.html
```

If Doxygen is missing, configuration still succeeds and prints a note instead;
the `docs` target simply does not exist. To drop the target on purpose,
configure with `-DBAYESTS_BUILD_DOCS=OFF`.

Settings live in [docs/Doxyfile.in](docs/Doxyfile.in), a template that CMake
expands into the build tree. Notable choices:

* `EXTRACT_ALL = YES`, so files that carry no doc comments yet still appear in
  the file and class lists. Worth revisiting once the codebase is documented
  throughout.
* MathJax renders the LaTeX formulas in the doc comments, so no LaTeX
  installation is needed. Viewing the pages does require network access; set
  `USE_MATHJAX = NO` if the documentation has to work offline.
* `WARN_NO_PARAMDOC = YES` reports `@param` names that no longer match the
  signature.
* Class, inheritance and include graphs are drawn as interactive SVG whenever
  CMake finds Graphviz, and silently skipped when it does not. Per-function
  call graphs are off; set `CALL_GRAPH` and `CALLER_GRAPH` to `YES` if you want
  them, at the cost of a much slower run.

### Writing doc comments

Use Javadoc-style blocks on the declaration in the header, or on the definition
where there is only a `.cpp`. `src/core/algorithms/wishart.cpp` is the worked
example. Formulas go between `\f$ ... \f$` inline or `\f[ ... \f]` for display.

```cpp
/**
 * @brief One-line summary, ending with a period.
 *
 * Longer description, including the statistical model where that is what the
 * function implements.
 *
 * @param x What it holds, including its dimensions.
 * @return What comes back.
 */
```

## Multi-threading

BayesTS uses **OpenMP** and **Armadillo** with **OpenBLAS** for multi-threaded
matrix operations, which can significantly speed up computations on multi-core
CPUs. The following operations are parallelised automatically:

- Matrix multiplication (e.g. `z * a`, `arma::trans(z) * u_sigma_inv_diag * z`)
- Linear system solving (`arma::solve()`)
- Cholesky decomposition (`arma::chol()`)
- Eigenvalue decomposition (`arma::eig_sym()`)
- Kronecker products (`arma::kron()`)
- Wishart distribution sampling (`arma::wishrnd()`)

OpenMP support is picked up by CMake automatically — `-fopenmp` on GCC/MinGW,
`/openmp` on MSVC, and `libomp` where Clang needs it installed separately. A
multi-threaded OpenBLAS is assumed, which is what vcpkg provides. During
configuration you should see:

```
-- OpenMP found - enabling multi-threaded matrix operations
-- OpenMP enabled: Using X threads for matrix operations
```

### Controlling the thread count

By default the program uses all available cores, detected at startup and
reported on the console. `OMP_NUM_THREADS` overrides it:

**Windows (PowerShell)**

```powershell
$env:OMP_NUM_THREADS=4
.\bayests.exe posterior model.h5
```

**Windows (Command Prompt)**

```cmd
set OMP_NUM_THREADS=4
bayests.exe posterior model.h5
```

**Linux/Mac**

```bash
export OMP_NUM_THREADS=4
./bayests posterior model.h5
```

The count can also be fixed in the code, in [src/bayests.cpp](src/bayests.cpp):

```cpp
#ifdef _OPENMP
    omp_set_num_threads(4);  // Use 4 threads instead of auto-detection
    std::cout << "OpenMP enabled: Using 4 threads" << std::endl;
#endif
```

### Performance expectations

Speedup depends on the matrix size, the number of physical cores and the
operation — matrix multiplication and linear solves benefit most:

- **Small matrices** (< 100x100): minimal speedup, overhead dominates
- **Medium matrices** (100x100 to 1000x1000): 2-4x on 4-8 cores
- **Large matrices** (> 1000x1000): 4-8x on 8+ cores

Multi-threading can be *slower* with very small matrices, on systems with
limited memory bandwidth, or when hyperthreading gives more logical cores than
physical ones. Reducing `OMP_NUM_THREADS` to the number of physical cores is
the fix.

### Troubleshooting

*"OpenMP not found" during CMake configuration.* On GCC/MinGW, check
`gcc --version` for a recent enough compiler (>= 9.0); under MSYS2 the runtime
is `pacman -S mingw-w64-x86_64-openmp`.

*Single-threaded despite OpenMP being found.* Check that OpenBLAS itself is
multi-threaded (`vcpkg list | grep openblas`), look for the OpenMP message in
the console output at startup, and try setting `OMP_NUM_THREADS` explicitly. To
confirm threading is working, watch CPU usage during a run, or compare timings:

```bash
OMP_NUM_THREADS=1 ./bayests posterior model.h5
OMP_NUM_THREADS=8 ./bayests posterior model.h5
```

### References

- [Armadillo Documentation - OpenMP](https://arma.sourceforge.net/faq.html#openmp)
- [OpenMP API Specification](https://www.openmp.org/)
- [OpenBLAS Performance Guide](https://github.com/xianyi/OpenBLAS/wiki)

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) is the guide: how a model is split across the
four layers and what each one may touch, the order to add the files in so that
every step builds on its own, how to check a change to a sampler against the
recorded fingerprints, and the documentation conventions.

The short version of the layering, because it is the constraint most likely to
be broken by accident: `include/bayests/` holds the contract, `src/core/` the
numerics with no I/O and no console, `src/io/hdf5/` the only code that may
include HighFive, and `src/models/` the front-end the command line drives.
Treat a new dependency across those lines as a design change rather than a build
fix.

The samplers are stochastic, so a refactor is checked by pinning the RNG and
comparing fingerprints before and after — `test/golden_models.cpp` prints them
and `test/record_fingerprints.sh` and `test/diff_fingerprints.sh` compare two
runs, as *Tests* above describes. No expected fingerprint is checked in: they
shift in the last digits with the toolchain, so the baseline is one you record
yourself. Note that the `*.h5` fixtures are not in the repository either;
`test/make_model_fixture.cpp` generates the inputs.

Contributions are accepted under the BSD 3-Clause terms below. New source files
need the `SPDX-License-Identifier: BSD-3-Clause` header that every existing one
carries.

## License

BayesTS is released under the [BSD 3-Clause License](LICENSE). Every source file
carries an `SPDX-License-Identifier: BSD-3-Clause` header.

BSD-3-Clause was chosen over Apache-2.0 for downstream compatibility: BSD-3-Clause
can be combined with GPL-2.0-only code, and Apache-2.0 cannot. That matters for
R-ecosystem consumers, where `GPL-2` and `GPL (>= 2)` package licences are common.
The Apache-2.0 patent grant offers little in return for a numerical library
implementing published algorithms.

### Dependency licenses

| Dependency | License |
| --- | --- |
| Armadillo | Apache-2.0 |
| HDF5 | BSD-3-Clause style (HDF Group) |
| HighFive | Boost Software License 1.0 |
| OpenBLAS / LAPACK | BSD-3-Clause |
| libgomp (OpenMP, GCC/MinGW builds) | GPL-3 **with GCC Runtime Library Exception** |

The GCC Runtime Library Exception means that linking the OpenMP runtime does not
impose the GPL on this project or on anything built with it. The OpenMP row is
toolchain-specific: MSVC links `vcomp` (proprietary, redistributable) and Clang
links `libomp` (Apache-2.0 with LLVM exception). The dependency set is entirely
permissive under all three, and binaries may be redistributed under BSD-3-Clause
terms.

Armadillo's Apache-2.0 terms are themselves incompatible with GPL-2.0-only; a
downstream needing that combination would have to replace it. BayesTS's own
source imposes no such restriction.

### Sparse linear algebra

Sparse storage and arithmetic use Armadillo's `arma::sp_mat`. Sparse *factorisation*
is handled by the LAPACK banded routines already reachable through the linked
BLAS/LAPACK (`dpttrf`/`dpttrs` for tridiagonal precisions, `dpbtrf`/`dpbtrs` for
banded ones), which are BSD-3-Clause and add no dependency. The precision matrices
in these samplers are tridiagonal or block-banded, so a general sparse Cholesky
would pay graph-analysis overhead to rediscover a structure that is known ahead of
time.

SuiteSparse is deliberately *not* used. Its modules are not uniformly licensed —
`AMD`/`COLAMD` are BSD-3-Clause, CHOLMOD's Cholesky core is LGPL-2.1+, and the
supernodal (BLAS-3) path, `UMFPACK` and `SPQR` are GPL-2.0+ — so the fast path is
the copyleft one. Should a model with a genuinely irregular sparsity pattern be
added later, Eigen's `SimplicialLDLT` is the intended fallback; note that Eigen's
sparse Cholesky and its AMD ordering are LGPL-2.1+ rather than MPL-2.0, so that
step would be an explicit, isolated opt-in rather than a project-wide dependency.

## AI assistance

Parts of this project were written with the help of Claude (Anthropic), used as
a coding assistant for drafting implementations, refactoring and documentation.
Generated code is reviewed before it is committed and is held to the same checks
as anything hand-written: it has to respect the layering, and a change to a
sampler has to account for what it does to the fingerprints `test/golden_models.cpp`
prints, against a recording made before it.
Authorship of the project, and responsibility for the numerics, rest with the
maintainer; the licence and the copyright are unaffected. Commits carrying
assisted work are marked with a `Co-Authored-By` trailer, so the history says
which ones they are.

## References

Chan, J. C. C., & Jeliazkov, I. (2009). Efficient simulation and integrated
likelihood estimation in state space models. *International Journal of
Mathematical Modelling and Numerical Optimisation*, 1(1-2), 101-120.

Chan, J., Koop, G., Poirier, D. J., & Tobias, J. L. (2019). *Bayesian
Econometric Methods* (2nd ed.). Cambridge University Press.

Durbin, J., & Koopman, S. J. (2002). A simple and efficient simulation smoother
for state space time series analysis. *Biometrika*, 89(3), 603-615.

Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood
inference and comparison with ARCH models. *The Review of Economic Studies*,
65(3), 361-393. Implemented as `stochvol_ksc_1998` and covered by
`test/unit_stochvol.cpp`; no sampler here draws from it.

Koop, G., León-González, R., & Strachan, R. W. (2010). Efficient posterior
simulation for cointegrated models with priors on the cointegration space.
*Econometric Reviews*, 29(2), 224-242.

Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility
with leverage: Fast and efficient likelihood inference. *Journal of
Econometrics*, 140(2), 425-449.
