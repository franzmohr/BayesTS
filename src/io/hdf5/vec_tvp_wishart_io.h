// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VEC_TVP_WISHART_IO_H
#define BAYESTS_IO_HDF5_VEC_TVP_WISHART_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::vec_tvp_wishart
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without a
/// cointegration relation, without variable selection or without forecast data
/// yields an input with those parts left empty, which is exactly how the sampler
/// decides what to run. Anything missing that the model does need is reported by
/// VecTvpWishartInput::validate(), where the message can name the field.
VecTvpWishartInput read_input(const HighFive::File &file);

/// Reads posterior draws for the likelihood: the whole coefficient and
/// cointegration paths, since every period is scored under its own, against the
/// one precision each draw carries.
VecTvpWishartDraws read_loglik_coefficients(const HighFive::File &file,
                                            const VecTvpWishartInput &input);

/// The same, with both paths cut to the last in-sample period -- what a forecast
/// carries forward. The precision does not move, so it is read whole.
VecTvpWishartDraws read_forecast_coefficients(const HighFive::File &file,
                                              const VecTvpWishartInput &input);

void write_coefficients(HighFive::File &file, const VecTvpWishartDraws &draws);

} // namespace bayests::hdf5_io::vec_tvp_wishart

#endif // BAYESTS_IO_HDF5_VEC_TVP_WISHART_IO_H
