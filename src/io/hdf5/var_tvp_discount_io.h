// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VAR_TVP_DISCOUNT_IO_H
#define BAYESTS_IO_HDF5_VAR_TVP_DISCOUNT_IO_H

#include "bayests/inputs.h"
#include "bayests/results.h"
#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::var_tvp_discount
{

/// The file half of the one model here that is not a sampler, so the three
/// functions below are not quite the ones every other model's io exposes.
///
/// What differs, and why:
///
/// - **The regressors are `/data/train/x`, not `/data/train/z`.** The filter is
///   written against one k x n_reg coefficient matrix per period rather than a
///   SUR design, and reading `z` would mean forming a matrix k^2 times the size
///   to undo it again.
/// - **The coefficient prior is `/priors/a/mean` and `/priors/a/cov`, not
///   `/priors/a/mu` and `/priors/a/v_inv`.** Deliberately different names for a
///   deliberately different object: `mean` is the n_reg x k coefficient matrix
///   rather than a stacked vector, and `cov` is the regressor side of a
///   covariance rather than a precision over the whole of it. A file that
///   brought the other pair along would be read as having no coefficient prior
///   at all, and `bayests check` says so.
/// - **What comes back is a posterior, not draws.** `/posterior/a/coeffs` is
///   absent and nothing here writes it: joining one i.i.d. draw per period
///   would look exactly like a sampled coefficient path and is not one, the
///   smoothed posterior being dependent across periods. The closed form goes to
///   `/posterior/a/mean`, `/posterior/a/scale`, `/posterior/a/cov`,
///   `/posterior/u_sigma/scale` and `/posterior/df` instead, and a host that
///   wants draws takes them from those with
///   VarTvpDiscountEstimator::draw_period().
VarTvpDiscountInput read_input(const ModelFile &file);

/// Reads the closed form back. `loglik` and `forecast_mean` come back empty:
/// neither is stored, because both fall out of one deterministic pass over the
/// sample and a number kept in two places is a number that can disagree with
/// itself. The stage that writes `/posterior/loglik` re-runs the filter.
VarTvpDiscountPosterior read_posterior(const ModelFile &file);

void write_posterior(const ModelFile &file, const VarTvpDiscountPosterior &posterior);

/// Whether the file already carries a fitted posterior, which is what the
/// front-ends skip on and `bayests check` reports. `/posterior/a/mean` rather
/// than the `/posterior/u_sigma_inv/coeffs` every sampler is probed by, this
/// model writing neither draws nor a precision.
const char *const kPosteriorProbe = "/posterior/a/mean";

} // namespace bayests::hdf5_io::var_tvp_discount

#endif // BAYESTS_IO_HDF5_VAR_TVP_DISCOUNT_IO_H
