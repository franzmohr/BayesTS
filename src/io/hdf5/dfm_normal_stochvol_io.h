// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_DFM_NORMAL_STOCHVOL_IO_H
#define BAYESTS_IO_HDF5_DFM_NORMAL_STOCHVOL_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::dfm_normal_stochvol
{

/// Builds a sampler input from an HDF5 model file.
///
/// The DfmNormalGamma tree with the two error groups changed from gamma priors to
/// stochastic volatility ones, and nothing else: `/data/train/y` is still the only
/// data, `/priors/lambda` and `/priors/a` are still the two coefficient blocks,
/// and there is still no `/data/train/z`.
///
/// Each error group carries what VarNormalStochvol's `/priors/u_sigma` carries --
/// `offset`, the `shape` and `rate` of the log-volatility variance, the `mu` and
/// `v_inv` of the log-volatility before the sample, and `sigma`, which is the
/// *starting value* of that variance rather than a prior and sits in the prior
/// group only because that is where VarNormalStochvol's files already put it.
///
/// The two log-volatility paths are `/initial/u_h` and `/initial/v_h`, tt x k and
/// tt x n_factors, with `/initial/u_h_init` and `/initial/v_h_init` beside them.
/// Named apart rather than sharing VarNormalStochvol's `/initial/h`, because this
/// model has two of them and neither is the obvious default.
DfmNormalStochvolInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
DfmNormalStochvolDraws read_coefficients(const ModelFile &file);

/// The same, with both volatility paths cut down to the terminal period.
///
/// The forecast holds the volatility at its last in-sample value, so that is the
/// only period it reads -- which for a path that is k or n_factors wide per
/// period and tt periods long is the difference between one block and all of
/// them. `read_coefficients` is what the log likelihood needs, and it needs the
/// whole path.
DfmNormalStochvolDraws read_forecast_coefficients(const ModelFile &file,
                                                  const DfmNormalStochvolInput &input);

void write_coefficients(const ModelFile &file, const DfmNormalStochvolDraws &draws);

} // namespace bayests::hdf5_io::dfm_normal_stochvol

#endif // BAYESTS_IO_HDF5_DFM_NORMAL_STOCHVOL_IO_H
