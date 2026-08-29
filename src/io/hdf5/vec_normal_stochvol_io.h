// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VEC_NORMAL_STOCHVOL_IO_H
#define BAYESTS_IO_HDF5_VEC_NORMAL_STOCHVOL_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::vec_normal_stochvol
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without a
/// cointegration relation, without a covariance block, without variable
/// selection or without forecast data yields an input with those parts left
/// empty, which is exactly how the sampler decides what to run. Anything
/// missing that the model does need is reported by
/// VecNormalStochvolInput::validate(), where the message can name the field.
VecNormalStochvolInput read_input(const ModelFile &file);

/// Everything in the posterior, the volatility path included. What the
/// likelihood wants: every period is scored under its own precision.
VecNormalStochvolDraws read_coefficients(const ModelFile &file);

/// The same, with the precision cut to the last in-sample period -- the one a
/// forecast carries forward. The coefficients do not move, so they are read
/// whole.
VecNormalStochvolDraws read_forecast_coefficients(const ModelFile &file,
                                                  const VecNormalStochvolInput &input);

void write_coefficients(const ModelFile &file, const VecNormalStochvolDraws &draws);

} // namespace bayests::hdf5_io::vec_normal_stochvol

#endif // BAYESTS_IO_HDF5_VEC_NORMAL_STOCHVOL_IO_H
