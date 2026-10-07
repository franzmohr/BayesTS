// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_UC_NORMAL_GAMMA_IO_H
#define BAYESTS_IO_HDF5_UC_NORMAL_GAMMA_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::uc_normal_gamma
{

/// Builds a sampler input from an HDF5 model file.
///
/// `/data/train/y` holds the series and is the whole of the sample: the model
/// has no regressors, so there is no `/data/train/z`. `/data/train/x` holds the
/// deterministic terms of the trend's drift where `/model/n` counts any, and
/// `/data/forecast/x` their continuation over the horizon. Four normal blocks
/// -- `/priors/a` over the free cycle coefficients, `/priors/c` over the drift,
/// `/priors/tau0` over the trend before the sample and, with "gamma+covar",
/// `/priors/psi` -- and `/priors/u_sigma`, a gamma prior on each of the 2k shock
/// precisions. `/model/trend` and `/model/cycle` choose the model's shape.
UcNormalGammaInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
UcNormalGammaDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const UcNormalGammaDraws &draws);

} // namespace bayests::hdf5_io::uc_normal_gamma

#endif // BAYESTS_IO_HDF5_UC_NORMAL_GAMMA_IO_H
