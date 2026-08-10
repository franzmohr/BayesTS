// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VAR_NORMAL_GAMMA_IO_H
#define BAYESTS_IO_HDF5_VAR_NORMAL_GAMMA_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::var_normal_gamma
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without
/// regressors, without a covariance block, without variable selection or
/// without forecast data yields an input with those parts left empty, which is
/// exactly how the sampler decides what to run. Anything missing that the
/// model does need is reported by VarNormalGammaInput::validate(), where the
/// message can name the field.
VarNormalGammaInput read_input(const HighFive::File &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout. The error precision of this model does
/// not move with time, so the same draws serve the forecast and the likelihood.
VarNormalGammaDraws read_coefficients(const HighFive::File &file);

void write_coefficients(HighFive::File &file, const VarNormalGammaDraws &draws);

} // namespace bayests::hdf5_io::var_normal_gamma

#endif // BAYESTS_IO_HDF5_VAR_NORMAL_GAMMA_IO_H
