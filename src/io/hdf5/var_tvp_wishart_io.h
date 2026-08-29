// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VAR_TVP_WISHART_IO_H
#define BAYESTS_IO_HDF5_VAR_TVP_WISHART_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::var_tvp_wishart
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded. Anything missing that the
/// model does need is reported by VarTvpWishartInput::validate(), where the
/// message can name the field.
VarTvpWishartInput read_input(const ModelFile &file);

/// Reads posterior draws for the likelihood: the whole coefficient path, since
/// every period is scored under its own coefficients, and the last period's
/// precision, which is the one this model scores every observation under.
VarTvpWishartDraws read_loglik_coefficients(const ModelFile &file,
                                            const VarTvpWishartInput &input);

/// Reads posterior draws for the forecast: the last in-sample period of both
/// the coefficient path and the precision, which is what a forecast carries
/// forward.
VarTvpWishartDraws read_forecast_coefficients(const ModelFile &file,
                                              const VarTvpWishartInput &input);

void write_coefficients(const ModelFile &file, const VarTvpWishartDraws &draws);

} // namespace bayests::hdf5_io::var_tvp_wishart

#endif // BAYESTS_IO_HDF5_VAR_TVP_WISHART_IO_H
