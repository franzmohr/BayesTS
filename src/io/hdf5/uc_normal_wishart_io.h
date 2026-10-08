// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_UC_NORMAL_WISHART_IO_H
#define BAYESTS_IO_HDF5_UC_NORMAL_WISHART_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::uc_normal_wishart
{

/// Builds a sampler input from an HDF5 model file: UcNormalGamma's tree, with
/// `/priors/u_sigma` the Wishart prior's `df` and `2k x 2k` `scale` and
/// `/initial/u_sigma_inv` a `2k x 2k` matrix, and no covariance block.
UcNormalWishartInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
UcNormalWishartDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const UcNormalWishartDraws &draws);

} // namespace bayests::hdf5_io::uc_normal_wishart

#endif // BAYESTS_IO_HDF5_UC_NORMAL_WISHART_IO_H
