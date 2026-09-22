---
name: bayests
description: Write correct BayesTS code — build, run and read the HDF5 model files that drive the BayesTS Bayesian time series samplers (VAR, VEC, DFM and FAVAR Gibbs samplers in C++). Use whenever a task mentions BayesTS, a `/model/algorithm` attribute, the `bayests` command line, a model file with `/data/train/y` and `/priors/`, or any of the twenty-two algorithm names (VarNormalWishart, VecTvpGamma, DfmNormalStochvol, FavarNormalWishart, VarNormalAld, VarTvpDiscount, VecKlgs2010 and the rest). Also use when building such a file from Python (h5py) or R, when choosing an algorithm and error specification, or when reading posterior draws, forecasts or the pointwise log likelihood back out.
---

# Writing correct BayesTS code

BayesTS is a C++ library of Gibbs samplers for Bayesian time series models. The
unit of work is **one HDF5 file** — the *model file* — holding the data, the
priors, the starting values and a `/model/algorithm` attribute naming the
sampler. The command line reads it, runs that sampler, and writes the posterior
draws, forecasts and pointwise log likelihood back into the same file:

```bash
bayests posterior model.h5
```

Nothing else is passed on the command line. A run is fully described by the file
it is given, which is the single most important thing to understand: **the file
format is the entire API.** Getting a field wrong does not usually produce an
error — it produces a different model that runs to completion and reports
plausible numbers.

**So check a file before running it:**

```bash
bayests check model.h5
```

`check` reads each model through the same reader and validation a run uses,
plus the forecast's checks on its regressors, and draws and writes nothing. It
prints what the file resolved to: the dimensions, the coefficient counts, whether
the covariance block is on, the selection scheme, the horizon. It exits 1 with
the reason on stderr for a file a run would refuse. For a file a run would
accept it exits 0, but still warns about every dataset the model never opened
and every `/model` attribute no model reads. Those warnings are the silent
failures this skill is about: `/priors/psi` left unread because `error` was
spelled for another model, or `lags` written where `p` was meant. **Compare its
report with what you meant, and do not run a file with a warning you cannot
explain.**

**Working in R instead?** The bvartools (VAR, VEC) and dfmtools (DFM, FAVAR)
packages run these samplers from R objects, without a model file, and each has
its own skill. Use those, and come back here only for a model moved through
`write_to_hdf5()` to the command line.

## The five rules that prevent silent wrongness

Everything else in this skill is detail. These five are the ones that fail
quietly.

**1. Orientation: the file holds the transpose of the matrix you would write on
paper.** In HDF5 dataspace terms — what `h5py` reports — every dataset is **one
row per quantity and one column per draw or per observation**. A 12-parameter,
80-draw `/posterior/a/coeffs` is `(12, 80)`. R's HDF5 readers reverse the
dimension order, R being column-major where HDF5 is row-major, so an R session
sees the transpose: draws in rows, which is what `coda` expects of an `mcmc`
object. Both statements describe the same bytes; say which view you mean or say
neither. From Python, write `f["/data/train/z"] = Z.T`; from R, write the matrix
as it reads.

**2. `/data/train/y` is stacked by period, not by variable.** It is a single row
of `tt * k` numbers, `vec(Y')` — all `k` variables of period 1, then all `k` of
period 2, and so on. That is what makes it line up with the SUR regressor matrix
`z`, whose rows come in blocks of `k` per period. Stacking by variable instead
gives a model that estimates fine and means nothing.

**3. Run `coefficients` before `forecasts` or `loglik`.** Those two read the
posterior back and throw if `/posterior/u_sigma_inv/coeffs` is missing.
`bayests posterior` runs all three in the right order and is what you normally
want. See `references/pipeline.md`.

**4. Re-running does not re-estimate.** Every front-end skips when its output is
already in the file: `coefficients` returns immediately if
`/posterior/u_sigma_inv/coeffs` holds data — `/posterior/a/mean` for the two
`*TvpDiscount` models, which write no precision — and `forecasts` and `loglik`
do the same for theirs. To re-estimate, delete `/posterior` or write a fresh
file.

**5. Variable-selection positions are one-based.** `/priors/a/include` counts
from 1, the way R and the file format count; the samplers convert on read. A
position below 1 is rejected, and so is one that is not a whole number.

## The naming grammar

An algorithm name is `<Family><Coefficients><Errors>`, and each part names one
thing:

| Part | Values | Means |
| --- | --- | --- |
| Family | `Var`, `Vec`, `Dfm`, `Favar` | The model |
| Coefficients | `Normal`, `Tvp` | Constant with a normal prior, or a random walk |
| Errors | `Wishart`, `Gamma`, `Stochvol`, `Ald`, `Discount` | The error precision |

`VecKlgs2010` is the one name outside the grammar — it is `VecNormalWishart`
drawn against compact regressors instead of the SUR matrix they kronecker up
into, so it is a choice about cost rather than a different posterior.

`VarTvpDiscount` and `VecTvpDiscount` are inside the grammar but outside the
rest of the library: **they are not samplers.** Their posterior is closed form,
so they have no burn-in, nothing to thin and nothing to judge convergence on,
and they write a posterior rather than draws. `VecTvpDiscount` also holds the
cointegration space fixed — it is the one `Vec*` name whose `Tvp` does not reach
`beta`. See `references/algorithms.md`.

Two things the grammar does not say out loud:

- **`Normal` against `Tvp` always names the coefficients.** For a factor model
  that means the loadings *and* the transition, and `Tvp` moves both.
- **For `Favar*` the third part names `Q`, the state innovation precision**, not
  the idiosyncratic one. A factor model's idiosyncratic precision is diagonal by
  assumption and is never a choice — errors that may correlate leave the factors
  nothing to explain. That is why the FAVAR row has a `Wishart` entry and the
  DFM row cannot.

The twenty-two registered names, and what each requires and refuses, are in
`references/algorithms.md`.

## The `error` attribute

`/model/error` is a **descriptive string with exactly two load-bearing values**:

| Value | Effect |
| --- | --- |
| `gamma+covar` | Turns the covariance block on, for the gamma models only |
| `sv+covar` | Turns the covariance block on, for the stochastic volatility models only |
| anything else | Nothing at all |

The spelling is model-specific: `gamma+covar` on a stochastic volatility model
switches nothing on, and neither does `sv+covar` on a gamma model. Fourteen of
the twenty-two readers — the Wishart family, the two quantile models, the two
discounted ones, `VecKlgs2010` and the factor models — never compare the
attribute at all, so on those it is documentation and nothing more.

A covariance block also needs `k > 1`: one variable has no off-diagonal
covariance to model. When it is on, `psi` carries `k(k-1)/2` free elements and
the file needs `/priors/psi` and `/initial/psi` beside the `a` ones.

## The `varsel` attribute

`/model/varsel` is `none` (the default, and what an empty string means), `ssvs`
or `bvs`. **Any other spelling throws** rather than silently disabling
selection. Selection needs, in the prior group of the block it applies to:

- `inprior` — prior inclusion probabilities, one per coefficient of the block
- `include` — the positions selection applies to, **one-based**, and non-empty
- `tau0`, `tau1` — the two SSVS mixture components, for `ssvs` only

and `a_lambda` (or `psi_lambda`) among the starting values, one per coefficient.
The four time-varying models with a covariance block read a **separate**
`varsel` attribute at `/model/priors/psi` for that block; the `/model` one
governs the coefficients.

`include` and `a_lambda` are two different things, and the file needs both.
Selection visits only the positions `include` names; every other position keeps
the indicator `a_lambda` gave it, **for the whole run**. That is how a
coefficient is held unrestricted — leave it out of `include` and start it at
one, which is what Korobilis (2013) does with the intercepts — and it is also
how one disappears without a word: start it at zero, leave it out of `include`,
and it is masked out of every draw, with no error and nothing in the output to
say the model that ran was not the model in the file. Start `a_lambda` at all
ones unless you mean an outright restriction, and let `include` decide what
moves.

### `ssvs` takes over the prior at the selected positions

SSVS is George, Sun and Ni (2008): each selected coefficient gets the mixture
`N(0, tau0²)` (the spike, "excluded") or `N(0, tau1²)` (the slab, "included"),
and its indicator is drawn by weighing the two densities at the current draw.
That draw looks at the coefficient alone, which is only right under the paper's
prior, so three files are **refused** rather than sampled under a prior the
indicators are not scored against:

- a non-zero `/priors/a/mu` at a selected position. Both components are
  centred at zero; a Minnesota mean of 1 on an own first lag is a prior to
  shrink towards, not a coefficient to select — leave it out of `include`, or
  use `bvs`;
- a `/priors/a/v_inv` with anything off the diagonal in the row or column of a
  selected position. The indicators are independent given the coefficients only
  when the prior makes the selected ones independent of everything else;
- `tau0` not smaller than `tau1` at a selected position. Swapped, the chain
  would run and every indicator would mean the opposite of what it says.

The same holds for `/priors/psi` under a covariance block. **The diagonal of
`v_inv` at a selected position is not a prior you get to choose**: the sweep
overwrites it with `1/tau0²` or `1/tau1²` every draw, so the file's value is
used for the first coefficient draw only and then never again. Put what you
believe about the coefficient's scale into `tau1` instead.

Choosing the two: `tau0` small enough that a coefficient within about
`3·tau0` of zero makes no difference to the fit, `tau1` wide enough that
`N(0, tau1²)` covers every value you would find plausible. The paper's
semi-automatic default is to scale both by the least-squares standard error of
the coefficient, `tau0 = 0.1·se` and `tau1 = 10·se`; its simulations use the
flat pair `(0.1, 5)`. On `psi` the scale is that of BayesTS's unit triangular
factor, whose elements are the paper's `ψ_ij` divided by its `ψ_jj`: the zero
pattern is the same, so the same restrictions are selected, but `kappa` values
tuned for the paper's parametrisation do not carry over unchanged.

### `bvs` needs a coefficient prior that is not flat

BVS excludes a coefficient by zeroing its regressor, so while it is out its
draw comes from the prior alone — and the sweep decides whether to let it back
in by scoring that prior draw against the data. The flatter the prior, the
wilder that draw and the worse it scores, so the harder it is for anything to
get back in once it is out. Korobilis (2013, §3.1) puts the point where
selection stops working at a prior variance of about 100, and quotes Kuo and
Mallick's (1997) usable range of 0.25 to 25 — in `v_inv` terms a diagonal of
roughly 0.04 to 4, with the variables on a comparable scale.

Nothing refuses a flatter prior — it is a perfectly good prior, and the chain
it produces is the one the file asked for. `/priors/a/v_inv` is checked for
being square and symmetric and not for being tight, so a file with
`varsel = "bvs"` and a near-zero `v_inv` runs to the end and reports inclusion
probabilities pinned near zero. **It is warned about twice**: `bayests check`
prints it before anything runs, and the run itself says the same line before
its first draw, so a chain started without a check still tells you. Both name
the block, how many of the selected positions are affected and the worst
conditional prior variance among them, and neither refuses the file — the exit
code stays 0. **Posterior inclusion that comes back at or near zero
for every selected coefficient is this, not a finding**; compare against a run
with a tighter `v_inv` before reading anything into it.

The warning covers the constant-coefficient models only. On a time-varying one
the same argument runs through the state prior instead — `/priors/a/mu` and
`v_inv` on the state before the sample, together with `shape`/`rate` on the
innovation precision, are what decide how far an excluded path wanders — and
there is no one variance there to compare against a threshold, so the check
says nothing and the judgement is yours. SSVS is not exposed to any of this:
its excluded component is the `tau0` spike, which is tight by construction.

## `structural`

`/model/structural` is a boolean. It turns the last `k(k-1)/2` entries of `a`
into contemporaneous coefficients, which the samplers split off before a
forecast path is simulated. **It is refused wherever the error covariance is
unrestricted** — with a Wishart precision, and with a covariance block — because
`A_0` is then not identified and a draw of it is the prior plus wherever the
chain last wandered. Estimate it against a diagonal covariance instead: the
`gamma` or `sv` error specification *without* `+covar`.

## Chain length: `iterations`, `burnin`, `thin`

`iterations` is the number of draws **kept**, and every posterior dataset has
that many columns. `burnin` draws are discarded first. `thin` (default 1) keeps
the last of every `thin` draws after the burn-in, so the chain runs
`burnin + iterations * thin` draws while the file stays the size `iterations`
makes it. Reach for it when a posterior summary moves with `/model/seed`. That is
a chain too short for how slowly it mixes, and thinning lets it run longer
without a larger file. `references/recipes.md` has the example.

## Dimension arithmetic

Getting these wrong is the other common silent failure. The priors and the
starting values are checked against these counts, so a mismatch usually does
throw — but a *self-consistent* wrong count does not.

```
VAR:    n_x     = k*p + m*(s + 1) + n
        nparams = k*n_x                       (+ k(k-1)/2 if structural)

VEC:    n_x_vec = k*max(p - 1, 0) + m*s + n
        nparams = k*rank + k*n_x_vec          (+ k(k-1)/2 if structural)

psi:    k(k-1)/2                              when the covariance block is on and k > 1

DFM:    lambda  = n_factors*(2k - n_factors - 1)/2
        a       = n_factors^2 * p
FAVAR:  lambda  = (k - n_factors)*(n_factors + n_obs_factors)
        a       = (n_factors + n_obs_factors)^2 * p
```

A VEC's `p` and `s` are **level orders**, the convention the files are written
with, so differencing costs one lag block: a VEC of level order 1 has no `Gamma`
at all.

The two loading counts are **not interchangeable**, and a length check will not
save you — they coincide on a whole family of dimensions, every
`k = n(n+1)/2` with one observed factor among them. Which count applies is
decided by which model the file is, never by which one the array happens to fit.
`references/algorithms.md` says why the two identification schemes differ.

## Reference files

Read the one you need; each is written to be read on its own.

| File | Contents |
| --- | --- |
| `references/model-file.md` | Every attribute, group and dataset a reader looks for, with shapes |
| `references/algorithms.md` | The twenty-two algorithms: what each needs, and the full table of refused combinations |
| `references/pipeline.md` | The command line, the run order, idempotence, `--group`, exit codes |
| `references/results.md` | What a run writes, and how to read it back from Python and R |
| `references/recipes.md` | Worked end-to-end examples: a VAR file from h5py, a VEC, a forecast, a DFM |

## What BayesTS refuses

These are refusals with reasons, not gaps to work around. Each is stated in full
in `references/algorithms.md`:

- SSVS alongside stochastic volatility, time-varying parameters, or a quantile model
- Variable selection on a VEC's loadings, on `VecKlgs2010`, or on any factor model
- `structural` with a Wishart precision or a covariance block
- A covariance block or a non-zero horizon on either `*Ald` model
- A quantile outside `(0, 1)`
- A NaN or an infinity anywhere under `/data`, `/priors` or `/initial` — there
  is no treatment of missing values. That includes `/data/test/y` (score only
  the periods you have) and the cells of `/data/forecast/x` a forecast
  overwrites, which only need to be finite

The two `Ald` refusals look most like missing features and are not. A covariance
block would rotate the equations into each other, and the q-th quantile of a
combination of equations is not the combination of their q-th quantiles — the
file would still be named for a quantile while the numbers were not one. A
forecast would have to iterate a one-step quantile, and the h-step quantile is
not that either.
