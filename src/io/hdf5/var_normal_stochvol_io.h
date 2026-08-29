// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VAR_NORMAL_STOCHVOL_IO_H
#define BAYESTS_IO_HDF5_VAR_NORMAL_STOCHVOL_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::var_normal_stochvol
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded: a file without
/// regressors, without a covariance block, without variable selection or
/// without forecast data yields an input with those parts left empty, which is
/// exactly how the sampler decides what to run. Anything missing that the
/// model does need is reported by VarNormalStochvolInput::validate(), where
/// the message can name the field.
///
/// The starting variance of the log-volatility innovations lives under
/// /priors/u_sigma/sigma on disk but is a state, not a prior, so it is read
/// into `initial` where the sampler expects to find it.
VarNormalStochvolInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, with the whole volatility
/// path -- one k x k precision per period -- which is what the likelihood
/// scores each observation under.
VarNormalStochvolDraws read_coefficients(const ModelFile &file);

/// The same, with the precision restricted to the last in-sample period. That
/// is the value a forecast path carries forward, and reading only it keeps the
/// sampler from having to know where in the stored path to look.
VarNormalStochvolDraws read_forecast_coefficients(const ModelFile &file,
                                                  const VarNormalStochvolInput &input);

void write_coefficients(const ModelFile &file, const VarNormalStochvolDraws &draws);

} // namespace bayests::hdf5_io::var_normal_stochvol

#endif // BAYESTS_IO_HDF5_VAR_NORMAL_STOCHVOL_IO_H
