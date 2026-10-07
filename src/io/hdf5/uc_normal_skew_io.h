// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_UC_NORMAL_SKEW_IO_H
#define BAYESTS_IO_HDF5_UC_NORMAL_SKEW_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::uc_normal_skew
{

/// Builds a sampler input from an HDF5 model file: UcNormalGamma's tree, with
/// `/priors/u_sigma` a gamma prior on each of the k normal shocks only, the
/// skew index at `/data/train/skew_x` (tt x n_skew) and `/data/forecast/skew_x`
/// (h x n_skew), both optional, and the scale coefficients' prior and start at
/// `/priors/skew` and `/initial/skew`.
UcNormalSkewInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
UcNormalSkewDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const UcNormalSkewDraws &draws);

} // namespace bayests::hdf5_io::uc_normal_skew

#endif // BAYESTS_IO_HDF5_UC_NORMAL_SKEW_IO_H
