# Grouping the forecast results under `/posterior/forecast`

A design note, written before the change and kept as the argument for it. It
turns the single dataset `/posterior/forecast` into a group holding the three
things a forecast produces, and says what that costs in both repositories that
write this format.

**Implemented.** `bayests` writes `/posterior/forecast/forecasts`, and bvartools
writes and reads the group with `errors` beside it. `loglik` is a reserved name
that nothing writes yet, and the two questions at the end are still open.

## What changes

| Was | Is | Holds |
| --- | --- | --- |
| `/posterior/forecast` | `/posterior/forecast/forecasts` | The simulated paths, one per draw |
| `/posterior/forecast_errors` (bvartools only) | `/posterior/forecast/errors` | Realised minus forecast, one per draw |
| — | `/posterior/forecast/loglik` | Reserved: the log predictive density of the realised observation, one per draw per horizon |
| `/posterior/loglik` | `/posterior/loglik` | Unchanged: the in-sample pointwise log likelihood |

In HDF5 dataspace terms, one row per quantity and one column per draw:

| Dataset | Shape |
| --- | --- |
| `/posterior/forecast/forecasts` | `(h*k, iterations)`; `(h*(k + n_obs_factors), iterations)` for a FAVAR — unchanged from `/posterior/forecast` |
| `/posterior/forecast/errors` | The same |
| `/posterior/forecast/loglik` | `(h, iterations)` — the **joint** density of the `k` variables of one horizon, not one column per variable, or it will not sum to a log predictive likelihood |

An R session sees the transpose of each, draws in rows, as it does for every
other posterior dataset.

## Why a group

**The name is already trying to be one.** bvartools writes `/posterior/forecast`
and `/posterior/forecast_errors` as siblings, in a loop over
`c("loglik", "forecast", "forecast_errors")`. The underscore is a group spelled
badly, and a third artifact of the same periods is about to arrive.

**One dataset cannot answer two questions.** The eighteen model front-ends that
forecast all guard with `if (dataset_has_data(file, "/posterior/forecast")) return;`.
That asks *have I forecast?*; there is nowhere to ask *have I scored?*. With a
group each artifact has its own dataset and its own guard.

**The readers need nothing.** bvartools walks `/posterior` generically: a child
that is a group contributes a named list of draws, a child that is a dataset
contributes one matrix, and its comment already says the distinction is made
"by what they are rather than by name, so that another one needs nothing here".
`posterior$forecast$forecasts`, `posterior$forecast$errors` and
`posterior$forecast$loglik` fall out of the existing walk. What follows the
rename on the R side is the consumers — `predict()`, `add_forecast_errors()`,
`add_posterior_forecasts()` — which treat `posterior$forecast` as a matrix
today.

## Why `forecasts` and not `draws`

`draws` reads well as a lone leaf and badly as one of three, because all three
*are* draws: the errors are draws, the loglik is draws. A leaf name has to say
what the numbers are, not what they have in common. `forecasts` does, at the
price of `/posterior/forecast/forecasts` reading like a stutter. That is the
whole cost and it is worth paying; `values` avoids the stutter but is vague in a
group where the other two are also values.

The group keeps the name `forecast`, matching `/data/forecast/x` and the `h`
attribute on `/model`.

## Why `/posterior/loglik` stays where it is

One loglik nested and one at the top looks asymmetric, and
`/posterior/loglik/{train,forecast}` would look tidier. It would also be false.
The two are not one statistic over different periods:

- `/posterior/loglik` evaluates each in-sample observation under states that
  have already seen it. It is what WAIC and PSIS-LOO read.
- `/posterior/forecast/loglik` conditions only on the data before the period it
  scores, which is what makes it usable for models whose coefficients or
  variances follow a state equation, where leave-one-out importance sampling
  fails for exactly the periods a path bends towards.

Grouping by what produced a quantity rather than by what it is called keeps that
difference visible, and puts the score beside the paths it is index-aligned
with: same horizon, same `/model/forecast_states`, same `/data/forecast/x`.

## Ownership and invalidation

The `forecasts` subcommand owns the whole group; `loglik` keeps owning
`/posterior/loglik`. One subcommand fills one group, which is worth holding onto
as an invariant. The predictive density belongs to `forecast()` rather than to
`log_likelihood()` for a mechanical reason as well as a conceptual one: it needs
the states carried forward over the horizon, which is `forecast_states.h`, and
the regressors in `/data/forecast/x`, which `log_likelihood()` never reads.

`forecasts` is the input to the other two, so rewriting it invalidates them.
Write that rule down and enforce it in one place — drop the group and refill it
— rather than leaving three datasets to drift apart.

The two quantile models never create the group. `VarNormalAld::forecast()` and
`VarTvpAld::forecast()` decline with a message rather than an error, and the
predictive density is unavailable for them for the same reason: `u_sigma_inv`
holds the precision of the normal their scale mixture conditions on, period by
period, not the density of an observation, whose mixing variable would have to
be integrated out. bvartools refuses them in `add_predictive_loglik()` on the
same grounds.

## The mcpar asymmetry this exposes

The two writers of this format already disagree, and the group move is what
makes it matter.

- bvartools attaches `start`, `end` and `thin` to every posterior dataset it
  writes, `loglik`, `forecast` and `forecast_errors` included.
- BayesTS calls `write_armadillo_matrix_to_hdf5(..., add_mcpar = false)` for
  both `/posterior/forecast` and `/posterior/loglik`, and `true` for everything
  written through `write_draws`.

It is invisible today only because the R reader looks for those attributes on
datasets *inside* a group and not on the loose ones. Move `forecasts` into a
group and the reader starts asking. So `write_forecast()` has to start writing
mcpar — which is right anyway: these are draws of the same chain at the same
thinning, and the `false` looks like an oversight rather than a decision.

`/posterior/loglik` should get the same treatment even though it is not moving.

## The open question: where the realised values live

`errors` and `loglik` both need the observations that were realised over the
horizon, and the file has nowhere to put them. `/data/test/y`, beside
`/data/train/y`, keeps inputs in `/data` and results in `/posterior`, and makes
each file self-contained — which is the property that lets a caller score an
expanding window one file at a time instead of holding window *i+1* in memory to
supply the observation window *i* predicted.

That raises a second question the design should settle rather than inherit:
**`errors` is derived and the other two are not.** The loglik is a density
evaluation and cannot be recovered from the paths; the errors are a subtraction.
With the realised values in the file, `errors` is a view of `forecasts` — the
same size as the largest array in the group, and the one thing in it that can go
stale. Storing it is defensible, since it is what the forecast criteria
summarise and recomputing it on every read of a folder is not free either. It
should be a decision.

## What the change touched

All of this is done except step 5's second half -- bvartools has no `loglik` in
the group to write, because nothing computes one there yet.

**BayesTS**

1. `write_forecast()` in `src/io/hdf5/model_io_common.cpp`: `ensure_group` on
   `/posterior/forecast`, write `forecasts` under it, with mcpar.
2. The eighteen `dataset_has_data(file, "/posterior/forecast")` guards in
   `src/models/*.cpp` name `/posterior/forecast/forecasts`. Note that they fail
   *open* if missed: `exist()` is true for a group, `getDataSet()` throws, the
   catch returns `false`, and every run silently forecasts again.
3. `CHANGELOG.md`, as a breaking change to the file format.

**bvartools**

5. `write_to_hdf5.*`: `forecast` and `forecast_errors` move out of the "draws
   kept on their own" loop into a group.
6. `predict()`, `add_forecast_errors()`, `add_posterior_forecasts()` and
   `selection_criteria()` follow the in-memory rename to
   `posterior$forecast$forecasts` and `posterior$forecast$errors`.
7. The vendored core is unaffected: none of this is in `src/core/` or
   `include/bayests/`.

## Migration

A file written before the change has a dataset where a new reader expects a
group, and HDF5 tells the two apart, so a reader could support both. The writers
cannot: they pick one. The format was one release old, so the answer taken is a
rerun of `bayests forecasts`, said so in the changelog, rather than a
compatibility branch through both repositories for files that take minutes to
regenerate.

`write_forecast()` unlinks a dataset it finds at the group's path before
creating the group, which is what makes that rerun work rather than fail on the
name -- HDF5 will not put a dataset below a dataset. Unlinking frees the name
and not the space, so a file whose forecast was large is worth passing through
`h5repack`. An object in an R session is migrated the same way, by
`add_posterior_forecasts()`, and bvartools says so when it meets the old layout
instead of reading it as a model that was never forecast.
