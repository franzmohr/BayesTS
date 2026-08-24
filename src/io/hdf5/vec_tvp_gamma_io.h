// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VEC_TVP_GAMMA_IO_H
#define BAYESTS_IO_HDF5_VEC_TVP_GAMMA_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::vec_tvp_gamma
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without a
/// cointegration relation, without a covariance block, without variable
/// selection or without forecast data yields an input with those parts left
/// empty, which is exactly how the sampler decides what to run. Anything
/// missing that the model does need is reported by
/// VecTvpGammaInput::validate(), where the message can name the field.
VecTvpGammaInput read_input(const HighFive::File &file);

/// Reads posterior draws for the likelihood: the whole coefficient and
/// cointegration paths, since every period is scored under its own, plus the
/// precision in whichever of its two shapes the covariance block implies.
VecTvpGammaDraws read_loglik_coefficients(const HighFive::File &file,
                                          const VecTvpGammaInput &input);

/// The same, with both paths cut to the last in-sample period -- and the
/// precision too, when a covariance block makes it move.
VecTvpGammaDraws read_forecast_coefficients(const HighFive::File &file,
                                            const VecTvpGammaInput &input);

void write_coefficients(HighFive::File &file, const VecTvpGammaDraws &draws);

} // namespace bayests::hdf5_io::vec_tvp_gamma

#endif // BAYESTS_IO_HDF5_VEC_TVP_GAMMA_IO_H
