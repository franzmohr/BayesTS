// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VAR_NORMAL_WISHART_IO_H
#define BAYESTS_IO_HDF5_VAR_NORMAL_WISHART_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::var_normal_wishart
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without
/// regressors, without variable selection or without forecast data yields an
/// input with those parts left empty, which is exactly how the sampler decides
/// what to run. Anything missing that the model does need is reported by
/// VarNormalWishartInput::validate(), where the message can name the field.
VarNormalWishartInput read_input(const HighFive::File &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
VarNormalWishartDraws read_coefficients(const HighFive::File &file);

void write_coefficients(HighFive::File &file, const VarNormalWishartDraws &draws);

} // namespace bayests::hdf5_io::var_normal_wishart

#endif // BAYESTS_IO_HDF5_VAR_NORMAL_WISHART_IO_H
