# Worked examples

Every Python example on this page is run by the BayesTS test suite
(`agents.recipes`, in `test/check_agent_recipes.py`): it executes the example,
runs `bayests posterior` over the file it wrote, and checks the shapes stated
here against the ones the run produced. The VAR, the VEC and the factor model
are complete scripts. The covariance block, variable selection and the
time-varying model are changes to the complete VAR, and say where they go.

Run the generator (last section) whenever you want a second opinion on a layout
this page does not cover.

Shapes are stated in HDF5 dataspace terms — see `model-file.md` for why that
matters and what R sees instead.

## A complete VAR from h5py

`VarNormalGamma`, 3 variables, 1 lag, an intercept, 24 periods, forecasting 4
ahead.

```python
import h5py
import numpy as np

k, p, m, s, n = 3, 1, 0, 0, 1      # 3 variables, 1 lag, an intercept
tt, h = 24, 4                      # 24 periods, forecast 4 ahead
iterations, burnin = 500, 250

n_x = k * p + m * (s + 1) + n      # 4 regressors per equation
nparams = k * n_x                  # 12 coefficients

rng = np.random.default_rng(1)
Y_full = np.zeros((tt + p, k))
for t in range(1, tt + p):
    Y_full[t] = 0.5 * Y_full[t - 1] + rng.standard_normal(k)
Y = Y_full[p:]                     # (tt, k), the estimation sample

# The SUR regressor matrix, (tt*k, nparams) as it reads on paper: one block of
# k rows per period.
Z = np.zeros((tt * k, nparams))
for t in range(tt):
    lag = Y_full[t]                # y_{t-1}
    Z[t * k:(t + 1) * k, :k * k * p] = np.kron(lag.reshape(1, -1), np.eye(k))
    Z[t * k:(t + 1) * k, k * k * p:] = np.eye(k)        # the intercept

# Out-of-sample regressors, (h, n_x) on paper. The block for lag j at horizon i
# is read from X only while j > i; every later one the sampler overwrites with
# the path it simulates. With one lag that is the first horizon's lag block
# alone. Deterministic and exogenous columns are yours to fill for every horizon.
X = np.zeros((h, n_x))
X[:, k * p:] = 1.0                 # the intercept, every horizon
X[0, :k * p] = Y[-1]               # the last observation

with h5py.File("var.h5", "w") as f:
    mdl = f.create_group("/model")
    mdl.attrs["algorithm"] = "VarNormalGamma"
    mdl.attrs["k"] = k
    mdl.attrs["p"] = p
    mdl.attrs["m"] = m
    mdl.attrs["s"] = s
    mdl.attrs["n"] = n
    mdl.attrs["h"] = h
    mdl.attrs["iterations"] = iterations
    mdl.attrs["burnin"] = burnin
    mdl.attrs["varsel"] = "none"
    mdl.attrs["error"] = "gamma"

    # vec(Y'): all k variables of period 1, then period 2, ... A C-order
    # flatten of a (tt, k) array is exactly that.
    f["/data/train/y"] = Y.reshape(1, -1)
    f["/data/train/z"] = Z.T           # store the transpose
    f["/data/forecast/x"] = X.T        # likewise

    f["/priors/a/mu"] = np.zeros((1, nparams))
    f["/priors/a/v_inv"] = np.eye(nparams)
    f["/priors/u_sigma/shape"] = np.full((1, k), 3.0)
    f["/priors/u_sigma/rate"] = np.full((1, k), 2.0)

    f["/initial/a"] = np.zeros((1, nparams))
    f["/initial/u_sigma_inv"] = np.eye(k)
```

Then:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bayests posterior var.h5
```

**Check the result against what you simulated.** `a` is `vec` of the `k x n_x`
coefficient matrix `[A_1 ... A_p, B_0 ... B_s, C]`, column-major, which is the
ordering `kron(lag', I_k)` implies. So with `k = 3` the first three entries are
the first *column* of `A_1`, and the last three are the intercept:

```python
with h5py.File("var.h5", "r") as f:
    a = f["/posterior/a/coeffs"][:]            # (12, 500)
A1 = a.mean(axis=1)[:k * k].reshape(k, k, order="F")
```

The diagonal of `A1` comes back near the 0.5 that generated it, with the
intercept near zero — which is what a correctly laid out `z` looks like. A `z`
built the other way round also runs, and gives you numbers that are not those.

## Adding a covariance block

Set the error spelling and add the `psi` prior and starting value, at width
`k(k-1)/2`. These lines go **inside the `with` block** of the complete VAR,
after what is already there:

```python
n_psi = k * (k - 1) // 2           # 3

f["/model"].attrs["error"] = "gamma+covar"
f["/priors/psi/mu"] = np.zeros((1, n_psi))
f["/priors/psi/v_inv"] = np.eye(n_psi)
f["/initial/psi"] = np.zeros((1, n_psi))
```

The spelling is model-specific: `gamma+covar` for the gamma models, `sv+covar`
for stochastic volatility. The wrong one switches nothing on and raises no
error.

`/posterior/psi/coeffs` comes out `(k*k, iterations)`, not `(n_psi,
iterations)`: each column is the whole lower triangular `Psi`, vectorised.

## Adding variable selection

`bvs` on the lag coefficients, leaving the intercepts in the model. **Positions
are one-based.** Again inside the `with` block of the complete VAR:

```python
f["/model"].attrs["varsel"] = "bvs"
f["/priors/a/include"] = np.arange(1, k * k * p + 1, dtype=np.int32).reshape(1, -1)
f["/priors/a/inprior"] = np.full((1, nparams), 0.5)
f["/initial/a_lambda"] = np.ones((1, nparams))
```

Note that `inprior` and `a_lambda` are one per coefficient — the full
`nparams` — while `include` names only the positions selection applies to.

**A constant-coefficient model with both a covariance block and `varsel` needs
the selection datasets for the `psi` block as well**, since the single
`/model/varsel` governs both. Leaving them out fails with exit 1 and
`Failed to read dataset '/initial/psi_lambda'`. With the covariance block
above in the file, add:

```python
f["/priors/psi/inprior"] = np.full((1, n_psi), 0.5)
f["/priors/psi/include"] = np.arange(1, n_psi + 1, dtype=np.int32).reshape(1, -1)
f["/initial/psi_lambda"] = np.ones((1, n_psi))
```

The four *time-varying* models with a covariance block read a separate
`varsel` attribute at `/model/priors/psi` for that block instead.

For `ssvs`, add `tau0` and `tau1` beside `inprior` — one per coefficient of the
block — and remember that SSVS reaches only the four constant-coefficient
models without stochastic volatility.

## Running a long chain with thin

`iterations` counts the draws **stored**, not the draws that are independent of
one another. A chain that mixes slowly — stochastic volatility near a spike, a
time-varying cointegration space — can need far more of them than a file should
hold, and every posterior dataset is sized by `iterations`. `thin` lets the chain
run longer without the file growing. Inside the `with` block of the complete
VAR:

```python
f["/model"].attrs["thin"] = 10     # keep one draw in 10 after the burn-in
```

The chain now runs `burnin + iterations * thin` draws and keeps the last of
every 10, so every result still holds `iterations` draws and the file is the
same size. The run takes about ten times as long. `bayests check` says what it
read: `chain: 500 draws kept after 250 burn-in, one in 10, so 5250 run`. The
`mcmc` attributes on each `/posterior` block record it too: `start` is 10 and
`end` is `iterations * 10`.

A chain thinned this way is exactly every tenth draw of the unthinned chain run
from the same seed. Thinning adds no information that the full chain lacks — keep
every draw instead if it fits. `thin` is what to reach for when it does not. The
sign that a chain is too short is a posterior summary that moves when only
`/model/seed` changes.

## Re-running

Every stage skips when its output is already there, so a second run does
nothing until the posterior is gone:

```python
import h5py

with h5py.File("var.h5", "a") as f:
    del f["posterior"]
```

## A time-varying model

`VarTvpGamma` differs from the complete VAR in three places. `/priors/a` becomes
a state equation, `/initial/a` becomes a whole path, and the chain needs the
state before the sample and a starting value for the precision.

In the `with` block of the complete VAR, **write these in place of** its
`/priors/a/mu`, `/priors/a/v_inv`, `/initial/a` and `/initial/u_sigma_inv`
lines, and keep the rest. h5py refuses to create a dataset that already exists,
so appending them after the originals fails; and `VarTvpGamma` starts its
precision from `/initial/u_omega_inv`, so a `u_sigma_inv` left behind is never
read — `bayests check` names it:

```python
f["/model"].attrs["algorithm"] = "VarTvpGamma"

f["/priors/a/mu"] = np.zeros((1, nparams))        # the state before the sample
f["/priors/a/v_inv"] = np.eye(nparams)
f["/priors/a/shape"] = np.full((1, nparams), 3.0) # the innovation precision
f["/priors/a/rate"] = np.full((1, nparams), 0.01)

f["/initial/a"] = np.zeros((tt, nparams))         # (tt, nparams): the path
f["/initial/a_sigma_inv"] = np.eye(nparams)
f["/initial/a_init"] = np.zeros((1, nparams))
f["/initial/u_omega_inv"] = np.eye(k)
```

The posterior widens the same way: `/posterior/a/coeffs` is
`(nparams*tt, iterations)`, period blocks of `nparams` values in order, so

```python
a_path = a.reshape(tt, nparams, iterations)       # period, coefficient, draw
```

## A VEC

`VecNormalWishart`: 3 variables with one cointegrating relation, level order 2,
and the constant restricted to the cointegration space. This is a complete file
rather than a change to the VAR, because nearly every dataset differs.

`p` and `s` are **level** orders, so a VEC of level order 2 carries one lagged
difference. `n` counts only the deterministic terms that enter *unrestricted*;
the constant here is inside the cointegration space and counted by
`n_restricted` instead.

```python
import h5py
import numpy as np

k, p, m, s, n = 3, 2, 0, 0, 0      # level order 2, nothing unrestricted
rank, n_restricted = 1, 1          # one relation, the constant inside it
tt, h = 60, 4
iterations, burnin = 500, 250

k_beta = k + m + n_restricted      # 4 rows of beta: y_{t-1}, then the constant
n_gamma = max(p - 1, 0)            # 1 lagged difference block
n_x_vec = k * n_gamma + m * s + n  # 3 regressors per equation after the loadings
nparams = k * rank + k * n_x_vec   # 12: the loadings first, then the rest

# Levels around 10 in which y_2 - y_1 is stationary while y_1 and y_3 wander.
rng = np.random.default_rng(1)
L = np.zeros((tt + p, k))
L[0] = 10.0
for t in range(1, tt + p):
    L[t, 0] = L[t - 1, 0] + rng.standard_normal()
    L[t, 1] = L[t, 0] + 0.5 * rng.standard_normal()
    L[t, 2] = L[t - 1, 2] + rng.standard_normal()

Y = L[p:] - L[p - 1:-1]                           # (tt, k): the differences
W = np.column_stack([L[p - 1:-1], np.ones(tt)])   # (tt, k_beta): y_{t-1}, constant

beta = np.zeros((k_beta, rank))    # where the chain starts
beta[0, 0] = 1.0

# (tt*k, nparams) on paper. The first k*rank columns hold beta' w_{t-1}; the
# sampler rebuilds them from every draw of beta, so filling them from the
# starting beta only makes the file readable on its own. Then the lagged
# difference -- one block here, since n_gamma is 1.
Z = np.zeros((tt * k, nparams))
for t in range(tt):
    rows = slice(t * k, (t + 1) * k)
    Z[rows, :k * rank] = np.kron((beta.T @ W[t]).reshape(1, -1), np.eye(k))
    dlag = L[p + t - 1] - L[p + t - 2]            # delta y_{t-1}
    Z[rows, k * rank:] = np.kron(dlag.reshape(1, -1), np.eye(k))

# The forecast is in LEVELS: the level VAR the draws imply, with the restricted
# constant now an ordinary column. (h, k*p + m*(s+1) + n + n_restricted) on
# paper. The block for lag j at horizon i is read from X only while j > i, so
# with p = 2 both lag blocks of the first horizon and the second lag block of
# the next one have to be filled in.
n_x_level = k * p + m * (s + 1) + n + n_restricted  # 7
X = np.zeros((h, n_x_level))
X[:, k * p:] = 1.0
for i in range(min(h, p)):
    for j in range(i + 1, p + 1):
        X[i, (j - 1) * k:j * k] = L[-(j - i)]

with h5py.File("vec.h5", "w") as f:
    mdl = f.create_group("/model")
    mdl.attrs["algorithm"] = "VecNormalWishart"
    mdl.attrs["k"] = k
    mdl.attrs["p"] = p
    mdl.attrs["m"] = m
    mdl.attrs["s"] = s
    mdl.attrs["n"] = n
    mdl.attrs["rank"] = rank
    mdl.attrs["k_beta"] = k_beta
    mdl.attrs["n_restricted"] = n_restricted
    mdl.attrs["h"] = h
    mdl.attrs["iterations"] = iterations
    mdl.attrs["burnin"] = burnin
    mdl.attrs["varsel"] = "none"
    mdl.attrs["error"] = "wishart"

    f["/data/train/y"] = Y.reshape(1, -1)
    f["/data/train/z"] = Z.T
    f["/data/train/w"] = W.T
    f["/data/forecast/x"] = X.T

    f["/priors/a/mu"] = np.zeros((1, nparams))
    f["/priors/a/v_inv"] = np.eye(nparams)
    f["/priors/u_sigma/df"] = k                # a scalar dataset
    f["/priors/u_sigma/scale"] = np.eye(k)
    f["/priors/beta/v_inv"] = 0.1              # a scalar dataset
    f["/priors/beta/p_tau_inv"] = np.eye(k_beta)

    f["/initial/a"] = np.zeros((1, nparams))
    f["/initial/u_sigma_inv"] = np.eye(k)
    f["/initial/beta"] = beta.reshape(1, -1, order="F")  # vec(beta), column by column
```

The loading columns are a function of the current draw rather than data, which
is why **variable selection may not reach them**: restrict `include` to
positions above `k*rank`.

`/posterior/beta/coeffs` is `(k_beta*rank, iterations)`, and
`/posterior/forecast` is in levels like the regressors that drive it — its first
horizon sits near `L[-1]`, not near zero. The regressor width, 7 here, is the
level layout's; `/data/train/z` carries 3 per equation after the loadings.

## A factor model

`DfmNormalGamma`: 6 observed series, 2 factors, a transition of order 1. A DFM
has no `/data/train/z` at all — its regressors are the factors, which are drawn
— and no `/data/forecast`: the horizon alone drives its forecast.

```python
import h5py
import numpy as np

k, n_factors, p = 6, 2, 1          # k counts OBSERVED series; p is the transition's order
tt, h = 60, 4
iterations, burnin = 500, 250

n_lambda = n_factors * (2 * k - n_factors - 1) // 2   # 9 free loadings
n_a = n_factors ** 2 * p                              # 4 transition coefficients

rng = np.random.default_rng(1)
F = np.zeros((tt, n_factors))
for t in range(1, tt):
    F[t] = 0.6 * F[t - 1] + rng.standard_normal(n_factors)
Lam = np.tril(np.full((k, n_factors), 0.7))           # leading block unit lower triangular
np.fill_diagonal(Lam, 1.0)
Y = F @ Lam.T + 0.3 * rng.standard_normal((tt, k))    # (tt, k)

with h5py.File("dfm.h5", "w") as f:
    mdl = f.create_group("/model")
    mdl.attrs["algorithm"] = "DfmNormalGamma"
    mdl.attrs["k"] = k
    mdl.attrs["n_factors"] = n_factors
    mdl.attrs["p"] = p
    mdl.attrs["h"] = h
    mdl.attrs["iterations"] = iterations
    mdl.attrs["burnin"] = burnin
    mdl.attrs["varsel"] = "none"
    mdl.attrs["error"] = "gamma"

    f["/data/train/y"] = Y.reshape(1, -1)            # all the data there is

    f["/priors/a/mu"] = np.zeros((1, n_a))           # the factor transition
    f["/priors/a/v_inv"] = np.eye(n_a)
    f["/priors/lambda/mu"] = np.zeros((1, n_lambda)) # the free loadings
    f["/priors/lambda/v_inv"] = np.eye(n_lambda)
    f["/priors/u_sigma/shape"] = np.full((1, k), 3.0)          # idiosyncratic errors
    f["/priors/u_sigma/rate"] = np.full((1, k), 2.0)
    f["/priors/v_sigma/shape"] = np.full((1, n_factors), 3.0)  # factor innovations
    f["/priors/v_sigma/rate"] = np.full((1, n_factors), 2.0)

    f["/initial/a"] = np.zeros((1, n_a))
    f["/initial/lambda"] = np.full((1, n_lambda), 0.5)
    f["/initial/u_sigma_inv"] = np.ones((1, k))      # (1, k), not (k, k)
    f["/initial/v_sigma_inv"] = np.ones((1, n_factors))
```

Note `/initial/u_sigma_inv`: a factor model's idiosyncratic precision is
diagonal by assumption, so it is stored flat as `(1, k)` rather than as the
`(k, k)` matrix a VAR writes — and `/posterior/u_sigma_inv/coeffs` is
`(k, iterations)` for the same reason.

Only the free loadings are stored: the leading `n_factors` square block of
`Lambda` is fixed unit lower triangular, so `n_lambda` is
`n_factors*(2k - n_factors - 1)/2` and not `k*n_factors`. The posterior is the
other way round: `/posterior/lambda/coeffs` is `(k*n_factors, iterations)`, the
whole loading matrix per draw, `vec`'d column by column with the fixed block in
place. The factor path comes back as `/posterior/factors/coeffs`,
`(n_factors*tt, iterations)`.

**A FAVAR uses a different count** — `(k - n_factors) * (n_factors +
n_obs_factors)`, because its leading block is the identity rather than a unit
triangle. The two coincide at a whole family of dimensions, so a length check
will not catch the substitution. See `algorithms.md`.

## Generating a fixture instead

The BayesTS tree ships a generator that writes a complete, admissible model file
for any of the twenty algorithms — useful as a reference file to compare yours
against:

```bash
bayests_make_model_fixture out.h5 VarTvpGamma none 1 0 4
#                          file  model       varsel covar structural h
```

Then read its shapes back with `h5py` and diff them against the file you built.
That is the fastest way to settle a question about a layout this bundle does not
cover.

## Notes on h5py types

- **String attributes**: h5py writes a Python `str` as variable-length UTF-8,
  which the reader accepts.
- **`structural`**: written as a numpy bool. Leave the attribute out entirely
  unless you need it — the default is false.
- **`include` positions** are integers, and must be written as an integer dtype
  (`np.int32`), not as floats.
- **Scalars** such as `/priors/u_sigma/df` and `/priors/beta/v_inv` are stored
  as datasets, not attributes.
