// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_UC_NORMAL_STOCHVOL_IO_H
#define BAYESTS_IO_HDF5_UC_NORMAL_STOCHVOL_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::uc_normal_stochvol
{

/// Builds a sampler input from an HDF5 model file: UcNormalGamma's tree, with
/// `/priors/u_sigma` the stochastic volatility group every stochvol model reads
/// -- `offset`, `shape` and `rate` or `omega_v`, `mu`, `v_inv`, and `sigma`, the
/// starting innovation variance -- at the width of the 2k shocks, and
/// `/initial/h` (tt x 2k) and `/initial/h_init`. "sv+covar" turns the covariance
/// block on.
UcNormalStochvolInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
UcNormalStochvolDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const UcNormalStochvolDraws &draws);

} // namespace bayests::hdf5_io::uc_normal_stochvol

#endif // BAYESTS_IO_HDF5_UC_NORMAL_STOCHVOL_IO_H
