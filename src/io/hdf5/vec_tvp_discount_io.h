// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VEC_TVP_DISCOUNT_IO_H
#define BAYESTS_IO_HDF5_VEC_TVP_DISCOUNT_IO_H

#include "bayests/inputs.h"
#include "bayests/results.h"
#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::vec_tvp_discount
{

/// The file half of the discounted VEC. Everything
/// io/hdf5/var_tvp_discount_io.h says about the discounted VAR's datasets holds
/// here; what a VEC adds is the error correction term and the cointegration
/// matrix.
///
/// - **`/data/train/w`** is the error correction term, as for every VEC, and
///   **`/data/train/x`** the short-run regressors alone in the compact layout
///   VecKlgs2010 reads -- the lagged differences, the unmodelled variables and
///   the unrestricted deterministic terms. The `rank` error correction columns
///   are built from `w` and the cointegration matrix and go in front of them.
///
/// - **`/initial/beta` is the model, not a starting value.** It is read from
///   where every VEC keeps its cointegration matrix and in the same layout,
///   vec of a k_beta x rank matrix, but nothing iterates here, so where the
///   model starts is where it stays: this is the space the run conditions on.
///   Reusing the path is what makes a grid over candidate spaces a matter of
///   copying a file and rewriting one dataset, and what lets a file written for
///   VecNormalWishart be pointed at this model by changing `/model/algorithm`
///   and adding the two discounts.
///
/// - **`/posterior/beta/coeffs`** is written back with that same matrix in it,
///   one column rather than one per iteration. It is not an estimate and is not
///   presented as one; it is there because `a` carries only the loadings, and
///   without the space beside them nothing downstream can reconstruct Pi or
///   convert the model to its level VAR.
VecTvpDiscountInput read_input(const ModelFile &file);

/// Reads the closed form back, the cointegration matrix included. `loglik` and
/// `forecast_mean` come back empty for the reason the VAR's reader gives.
VecTvpDiscountPosterior read_posterior(const ModelFile &file);

void write_posterior(const ModelFile &file, const VecTvpDiscountPosterior &posterior);

/// What the front-ends skip on; see the VAR's reader for why it is not
/// `/posterior/u_sigma_inv/coeffs`.
const char *const kPosteriorProbe = "/posterior/a/mean";

} // namespace bayests::hdf5_io::vec_tvp_discount

#endif // BAYESTS_IO_HDF5_VEC_TVP_DISCOUNT_IO_H
