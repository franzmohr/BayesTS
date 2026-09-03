// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_DFM_TVP_GAMMA_IO_H
#define BAYESTS_IO_HDF5_DFM_TVP_GAMMA_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::dfm_tvp_gamma
{

/// Builds a sampler input from an HDF5 model file.
///
/// The tree is DfmNormalGamma's -- `/priors/lambda` over the free loadings,
/// `/priors/a` over the factor transition, `/priors/u_sigma` and
/// `/priors/v_sigma` over the two error precisions, and `/data/train/y` as the
/// only data -- with the two coefficient groups carrying what every
/// time-varying block here carries instead of a normal prior on a point:
///
///   - `/priors/lambda/{shape,rate}` and `/priors/a/{shape,rate}`, the inverse
///     gamma on the variance of the state innovations, and
///     `/priors/lambda/{mu,v_inv}` and `/priors/a/{mu,v_inv}`, the normal on the
///     state of the period before the sample. The same pair of datasets in one
///     group that `/priors/a` holds for a VAR whose coefficients drift.
///   - `/initial/lambda` and `/initial/a` as paths -- one long row, cut into
///     periods on the way in -- beside `/initial/lambda_sigma_inv`,
///     `/initial/lambda_init`, `/initial/a_sigma_inv` and `/initial/a_init`.
///
/// A model with one observed series has no free loading at all, its whole
/// loading matrix being the identifying block, and then nothing in the lambda
/// group is read or wanted.
DfmTvpGammaInput read_input(const ModelFile &file);

/// Posterior draws as the pointwise log likelihood wants them: the whole
/// loading path, since every period is scored under its own Lambda_t.
DfmTvpGammaDraws read_loglik_coefficients(const ModelFile &file);

/// Posterior draws as a forecast wants them: the loadings and the transition cut
/// to their last in-sample period, which is what the horizon is run under. The
/// factor path is read whole -- the forecast starts from the last p factors of
/// it, and cutting it to one period would leave a transition of order two
/// without its second lag.
DfmTvpGammaDraws read_forecast_coefficients(const ModelFile &file, const DfmTvpGammaInput &input);

void write_coefficients(const ModelFile &file, const DfmTvpGammaDraws &draws);

} // namespace bayests::hdf5_io::dfm_tvp_gamma

#endif // BAYESTS_IO_HDF5_DFM_TVP_GAMMA_IO_H
