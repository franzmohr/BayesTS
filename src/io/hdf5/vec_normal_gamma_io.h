// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VEC_NORMAL_GAMMA_IO_H
#define BAYESTS_IO_HDF5_VEC_NORMAL_GAMMA_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::vec_normal_gamma
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without a
/// cointegration relation, without a covariance block, without variable
/// selection or without forecast data yields an input with those parts left
/// empty, which is exactly how the sampler decides what to run. Anything
/// missing that the model does need is reported by
/// VecNormalGammaInput::validate(), where the message can name the field.
VecNormalGammaInput read_input(const ModelFile &file);

/// Everything in the posterior. Nothing in this model moves with time, so one
/// reader serves the forecast and the likelihood alike.
VecNormalGammaDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const VecNormalGammaDraws &draws);

} // namespace bayests::hdf5_io::vec_normal_gamma

#endif // BAYESTS_IO_HDF5_VEC_NORMAL_GAMMA_IO_H
