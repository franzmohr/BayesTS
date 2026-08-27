// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_DFM_NORMAL_GAMMA_IO_H
#define BAYESTS_IO_HDF5_DFM_NORMAL_GAMMA_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::dfm_normal_gamma
{

/// Builds a sampler input from an HDF5 model file.
///
/// The groups differ from every other model's, because a dynamic factor model
/// is not a regression: there are two coefficient blocks with unrelated shapes
/// -- `/priors/lambda` over the free loadings and `/priors/a` over the factor
/// transition -- and two error blocks, `/priors/u_sigma` for the idiosyncratic
/// errors and `/priors/v_sigma` for the factor innovations. There is no
/// `/data/train/z` at all; `/data/train/y` holds the observed series and is the
/// only data the model takes.
///
/// `/model/n_factors` is what read_spec() picks the factor count out of, and a
/// file that omits it describes a model with none -- which
/// DfmNormalGammaInput::validate() rejects by name rather than running.
DfmNormalGammaInput read_input(const HighFive::File &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
DfmNormalGammaDraws read_coefficients(const HighFive::File &file);

void write_coefficients(HighFive::File &file, const DfmNormalGammaDraws &draws);

} // namespace bayests::hdf5_io::dfm_normal_gamma

#endif // BAYESTS_IO_HDF5_DFM_NORMAL_GAMMA_IO_H
