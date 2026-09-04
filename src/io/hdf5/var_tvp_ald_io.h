// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VAR_TVP_ALD_IO_H
#define BAYESTS_IO_HDF5_VAR_TVP_ALD_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::var_tvp_ald
{

/// Builds a sampler input from an HDF5 model file.
///
/// VarNormalAld's reader with the coefficient block widened to a path, and the
/// same two absences: no covariance block and no forecast data, both of which
/// VarTvpAldInput::validate() refuses rather than half-reads.
///
/// The quantile is a `/model` attribute, defaulting to the median.
VarTvpAldInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, with the whole coefficient
/// path -- one vector per period -- which is what the likelihood scores each
/// observation under.
VarTvpAldDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const VarTvpAldDraws &draws);

} // namespace bayests::hdf5_io::var_tvp_ald

#endif // BAYESTS_IO_HDF5_VAR_TVP_ALD_IO_H
