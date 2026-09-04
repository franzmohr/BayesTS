// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_VAR_NORMAL_ALD_IO_H
#define BAYESTS_IO_HDF5_VAR_NORMAL_ALD_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::var_normal_ald
{

/// Builds a sampler input from an HDF5 model file.
///
/// Optional groups are skipped rather than demanded, exactly as the other
/// readers do: a file without regressors or without variable selection yields
/// an input with those parts left empty, and anything missing that the model
/// does need is reported by VarNormalAldInput::validate(), where the message can
/// name the field.
///
/// The quantile is a `/model` attribute rather than a dataset, because it is
/// part of the shape of the model rather than of its priors -- the same reading
/// under which `p` and `k` are attributes. It defaults to the median, so a file
/// that omits it describes the one case in which the asymmetric Laplace is
/// symmetric.
///
/// There is no covariance block to read and no forecast data to read: this model
/// rejects both, so a file carrying either is refused by validate() rather than
/// silently half-read here.
VarNormalAldInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run.
///
/// One reader rather than the pair the other models have. A forecast would need
/// the terminal period alone, and this model does not forecast.
VarNormalAldDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const VarNormalAldDraws &draws);

} // namespace bayests::hdf5_io::var_normal_ald

#endif // BAYESTS_IO_HDF5_VAR_NORMAL_ALD_IO_H
