# AGENTS.md — working with BayesTS

Guidance for coding agents writing code that builds, runs or reads
[BayesTS](https://github.com/franzmohr/BayesTS) model files. The full bundle is
in `skills/bayests/`; this file is the part worth having in context at all
times.

## The shape of the thing

BayesTS is a C++ library of Gibbs samplers for Bayesian time series models —
VARs, VECs, dynamic factor models and a FAVAR. The unit of work is **one HDF5
file** holding the data, the priors, the starting values and a
`/model/algorithm` attribute naming the sampler:

```bash
bayests posterior model.h5
```

Nothing else is passed on the command line. **The file format is the entire
API.** A wrong field does not usually raise an error; it estimates a different
model and reports plausible numbers.

So **run `bayests check model.h5` before `posterior`**, and read what it prints.
It reads the file through the same code a run uses, without sampling or writing
anything. It prints the dimensions and switches the file resolved to, refuses
(exit 1, reason on stderr) what a run would refuse, and warns about every
dataset the model never reads and every `/model` attribute no model looks for.
A warning is usually a field meant for a different model, or a misspelled one.
Treat a warning as a bug in the file until you can say why it is there.

## Rules

1. **Orientation.** In HDF5 dataspace terms every dataset is one row per
   quantity and one column per draw or observation — `/posterior/a/coeffs` of a
   12-parameter, 80-draw model is `(12, 80)` to `h5py`. R's readers reverse the
   dimension order and see the transpose. From Python, store the transpose of
   the matrix as it reads on paper: `f["/data/train/z"] = Z.T`. Vectors are a
   single row, `(1, n)`.

2. **`/data/train/y` is stacked by period**, not by variable: `vec(Y')`, all `k`
   variables of period 1, then period 2. That is what aligns it with the SUR
   matrix `z`, whose rows come in blocks of `k` per period.

3. **Run `coefficients` before `forecasts` or `loglik`.** Both read the
   posterior back and throw without `/posterior/u_sigma_inv/coeffs`.
   `bayests posterior` does all three in the right order.

4. **Re-running does not re-estimate.** Every stage skips when its output is
   already in the file. Delete `/posterior` to redo a run.

5. **Variable-selection positions are one-based.**

6. **`error` is descriptive except for two values.** `gamma+covar` switches the
   covariance block on for the gamma models, `sv+covar` for the stochastic
   volatility models. Every other value — including the right one on the wrong
   model — does nothing at all.

7. **`varsel` is `none`, `ssvs` or `bvs`.** Any other spelling throws.

8. **Never infer a dimension from an array that happens to fit.** The DFM and
   FAVAR loading counts coincide at a whole family of dimensions, and
   substituting one for the other runs to completion on the wrong model.

9. **Exit codes are load-bearing**: 0 succeeded or had nothing to do, 1 the run
   started and failed, 2 the command line was unusable.

10. **Samplers are reproducible single-threaded only.** Set
    `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` when comparing runs.

## Where to read more

| Question | File |
| --- | --- |
| What goes in the file, and in what shape | `skills/bayests/references/model-file.md` |
| Which algorithm, and what it refuses | `skills/bayests/references/algorithms.md` |
| How to run it, and in what order | `skills/bayests/references/pipeline.md` |
| What comes out, and how to read it | `skills/bayests/references/results.md` |
| A file built from scratch, end to end | `skills/bayests/references/recipes.md` |

## Working from R

An analysis written in R does not usually touch a model file at all. The
[bvartools](https://github.com/franzmohr/bvartools) package covers VAR and VEC
models, and [dfmtools](https://github.com/franzmohr/dfmtools) dynamic factor
models and FAVARs. Both build the model as an R object and run these same
samplers compiled into the package, and each carries its own guide in
`inst/agents/`, installed at `system.file("agents", package = "bvartools")` (or
`"dfmtools"`). Read that guide rather than this one. This bundle matters to
R code only when a model moves between R and the command line through
`write_to_hdf5()` and `read_model_from_hdf5()`.

## When contributing to BayesTS itself

This bundle is about *using* BayesTS. The repository's own `CLAUDE.md` and
`CONTRIBUTING.md` govern changes to it, and two of their rules matter enough to
repeat:

- Include `"bayests/arma.h"`, never `<armadillo>`. It is the single point
  Armadillo enters the project, which is what lets an R host redirect it to
  RcppArmadillo so that `set.seed()` reaches the draws.
- `src/core/` and `include/bayests/` are **vendored by copy** into downstream R
  packages. A core change is only half done until it is propagated.
