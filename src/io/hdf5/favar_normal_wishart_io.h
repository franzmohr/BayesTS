// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_FAVAR_NORMAL_WISHART_IO_H
#define BAYESTS_IO_HDF5_FAVAR_NORMAL_WISHART_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::favar_normal_wishart
{

/// Builds a sampler input from an HDF5 model file.
///
/// The tree is a dynamic factor model's with two changes, and both come from the
/// same place -- half of a factor augmented VAR's state is data.
///
///   - `/data/train/f_obs` holds the observed factors, tt x `n_obs_factors`, one
///     period per row, beside the panel in `/data/train/y`. There is no
///     `/data/train/z`: a FAVAR has no regressors of the kind a VAR does.
///   - `/priors/v_sigma` carries `df`/`scale` rather than `shape`/`rate`. The
///     state innovation precision is a matrix here, not a diagonal, so its prior
///     is Wishart and not a stack of independent gammas. `/priors/u_sigma` is
///     still `shape`/`rate` -- the idiosyncratic precision of a factor model is
///     diagonal by assumption in every member of the family.
///
/// `/model/n_obs_factors` is what read_spec() picks the observed factor count
/// out of, and a file that omits it describes a dynamic factor model -- which
/// FavarNormalWishartInput::validate() rejects by name rather than running as
/// one.
FavarNormalWishartInput read_input(const ModelFile &file);

/// Reads posterior draws written by a previous run, transposed back into the
/// sampler's draw-per-column layout.
FavarNormalWishartDraws read_coefficients(const ModelFile &file);

void write_coefficients(const ModelFile &file, const FavarNormalWishartDraws &draws);

} // namespace bayests::hdf5_io::favar_normal_wishart

#endif // BAYESTS_IO_HDF5_FAVAR_NORMAL_WISHART_IO_H
