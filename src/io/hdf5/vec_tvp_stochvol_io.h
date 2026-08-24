// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VEC_TVP_STOCHVOL_IO_H
#define BAYESTS_IO_HDF5_VEC_TVP_STOCHVOL_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::vec_tvp_stochvol
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without a
/// cointegration relation, without a covariance block, without variable
/// selection or without forecast data yields an input with those parts left
/// empty, which is exactly how the sampler decides what to run. Anything
/// missing that the model does need is reported by
/// VecTvpStochvolInput::validate(), where the message can name the field.
VecTvpStochvolInput read_input(const HighFive::File &file);

/// Reads posterior draws for the likelihood: the whole coefficient and
/// cointegration paths, since every period is scored under its own, and the
/// last period's precision, which is the one this model scores every
/// observation under.
VecTvpStochvolDraws read_loglik_coefficients(const HighFive::File &file,
                                             const VecTvpStochvolInput &input);

/// Reads posterior draws for the forecast: the last in-sample period of the
/// coefficient path, of the cointegration path and of the precision, which is
/// what a forecast carries forward.
VecTvpStochvolDraws read_forecast_coefficients(const HighFive::File &file,
                                               const VecTvpStochvolInput &input);

void write_coefficients(HighFive::File &file, const VecTvpStochvolDraws &draws);

} // namespace bayests::hdf5_io::vec_tvp_stochvol

#endif // BAYESTS_IO_HDF5_VEC_TVP_STOCHVOL_IO_H
