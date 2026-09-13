# Running a model

```
bayests <command> <path_to_file.h5 | directory> [flags...]
```

Five commands, and nothing else on the command line but the path and the flags
below. The file names the sampler; the command names which results to produce.

| Command | Does |
| --- | --- |
| `check` | Reads and validates each model without running it — writes nothing |
| `posterior` | All three stages, in order |
| `coefficients` | The Gibbs sampler alone — writes `/posterior/<block>/coeffs` |
| `forecasts` | The forecast alone — writes `/posterior/forecast` |
| `loglik` | The pointwise log likelihood alone — writes `/posterior/loglik` |

## Checking a file first

```bash
bayests check model.h5
```

`check` opens the file read-only and passes each model through the reader and
`validate()` its run would use. It adds the checks the forecast makes on its
regressors, which a run otherwise meets only after the chain has finished. Then
it stops. For an accepted model it prints what the file resolved to:

```
  accepted: VarNormalGamma reads this file and would run it
  dimensions: k = 3, p = 1, m = 0, s = 0, n = 1, tt = 24
  coefficients: 12 (4 regressors per equation)
  covariance block: off (error "gamma")
  variable selection: none
  structural: no
  forecast: h = 4
  chain: 500 draws kept after 250 burn-in
```

Then one `warning:` line for each thing a run would not stop on but that
changes what it means:

| Warning | Usually means |
| --- | --- |
| `<dataset> is in the file, but <algorithm> never reads it` | A block the model does not have switched on, e.g. `/priors/psi` under the wrong `error` spelling, or a dataset written for another model |
| `/model attribute '<name>' is not one any model reads` | A misspelling, and the attribute it was meant to be is read through its default |
| `error "<x>" switches no covariance block on` | The spelling belongs to another model family |
| `/data/train/z has N columns, but ... make M` | `z` was built for different dimensions than the attributes state |

It exits 0 when every model would be accepted, warnings or not, and 1 with the
reason on stderr when a model would be refused. That includes two failures a
run meets only at the forecast stage, after the chain has run:
- forecast regressors that are missing or have too few horizons;
- a `z` whose width disagrees with the dimensions, when a forecast is asked for.

It takes `--group` and `--all-groups` like the rest.

## The order

`coefficients` first. `forecasts` and `loglik` read the posterior back and
**throw** if `/posterior/u_sigma_inv/coeffs` is missing, which the subcommand
reports and turns into exit 1.

`bayests posterior` runs them in the order `coefficients`, `loglik`,
`forecasts` — which is what you normally want:

```bash
bayests posterior model.h5
```

Its three steps can be switched off individually:

```bash
bayests posterior model.h5 --no-forecasts --no-loglik
```

Those three flags (`--no-coefficients`, `--no-forecasts`, `--no-loglik`) belong
to `posterior` only. The other three subcommands run one step each, so for them
those names are unrecognised flags — warned about and ignored, not refused.

## Re-running does nothing

Every stage skips when its output is already in the file:

| Stage | Skips when |
| --- | --- |
| `coefficients` | `/posterior/u_sigma_inv/coeffs` holds data — and says so on stdout |
| `forecasts` | `/posterior/forecast` holds data, **or** `/model` has no `h` attribute, or `h` is 0 |
| `loglik` | `/posterior/loglik` holds data |

This is deliberate — it makes a directory walk resumable — and it is the most
common reason a run appears to have "done nothing". **To re-estimate, delete
`/posterior` or write a fresh file.** From Python:

```python
import h5py
with h5py.File("model.h5", "a") as f:
    if "posterior" in f:
        del f["posterior"]
```

Note that HDF5 does not reclaim the space of an unlinked dataset, so a file
grows a little every time it is re-run. Worth a periodic `h5repack` on a file
holding many models.

Having nothing to do is **not** failing. A forecast with no horizon, and output
that is already present, both return quietly with exit 0.

## Many models at once

The path may be a file or a directory, and the directory is walked recursively
for HDF5 files. Inside each file, a model's tree hangs either at the root or
under a group:

| Flag | Meaning |
| --- | --- |
| `--group <path>` | The group each model's tree hangs under, e.g. `/models/3`. Default: the root of the file |
| `--all-groups` | Read `--group` as the root to search under and run **every** model below it. Default: the whole file |

One group applies to the whole invocation: a directory walk looks for the same
group in every file it visits, which is what a caller with a directory of files
written the same way wants.

```bash
bayests posterior models/ --all-groups        # every model in every file below models/
bayests coefficients one.h5 --group /models/3 # exactly one model
```

A model that fails is reported on stderr and the walk carries on to the next —
one bad file must not strand the rest. The exit status is what tells you
something failed.

## Exit codes

They are load-bearing; a script should branch on them.

| Code | Means |
| --- | --- |
| 0 | Everything asked for succeeded, or had already been done |
| 1 | The run started and something failed |
| 2 | It never started: the command line was unusable |

Exit 2 covers no path, a path that does not exist, an unknown command, a
`--group` with no value after it, and a group that cannot name an HDF5 group. A
path that exists but is not an HDF5 file is exit 1. A `--group` that is *well formed*
but names nothing in the file is the **first** kind, not the second: the command
line was actionable, the file just did not hold that model.

## Reproducibility

**The samplers are only reproducible single-threaded.** Set both of these when
running the binary directly:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bayests posterior model.h5
```

The draws come from Armadillo's RNG. Under an embedded host built against
RcppArmadillo that is R's own RNG, so `set.seed()` reaches the draws; from the
command line it is Armadillo's.

Fingerprints shift in the last digits with the compiler, the BLAS and the CPU,
so compare runs on the same machine and the same build or not at all.
