// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_DFM_TVP_STOCHVOL_IO_H
#define BAYESTS_IO_HDF5_DFM_TVP_STOCHVOL_IO_H

#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::dfm_tvp_stochvol
{

/// Builds a sampler input from an HDF5 model file.
///
/// The tree is the union of the two models this one sits between, with nothing
/// new in it. From DfmTvpGamma come the two coefficient groups as state
/// equations -- `/priors/lambda/{shape,rate}` and `/priors/a/{shape,rate}` on the
/// variance of the state innovations beside `/priors/{lambda,a}/{mu,v_inv}` on
/// the state before the sample, and `/initial/{lambda,a}` as paths beside
/// `/initial/{lambda,a}_sigma_inv` and `/initial/{lambda,a}_init`. From
/// DfmNormalStochvol come the two error groups,
/// `/priors/{u_sigma,v_sigma}/{offset,shape,rate,mu,v_inv,sigma}` and
/// `/initial/{u_h,u_h_init,v_h,v_h_init}`.
///
/// Two things about that union are worth stating because they are silent if got
/// wrong. `shape` and `rate` appear in all four prior groups and mean the same
/// thing in each -- the inverse gamma on a random walk's innovation variance --
/// but at three different widths: n_lambda, n_factor_a, k and n_factors. And
/// `/priors/{u_sigma,v_sigma}/sigma` is a *starting value* rather than a prior,
/// the same convention every stochastic volatility model here follows.
DfmTvpStochvolInput read_input(const ModelFile &file);

/// Posterior draws as the pointwise log likelihood wants them: the loading path
/// and the idiosyncratic precision path whole, since every period is scored under
/// its own Lambda_t and its own U_t.
DfmTvpStochvolDraws read_loglik_coefficients(const ModelFile &file);

/// Posterior draws as a forecast wants them: the loadings and the transition cut
/// to their last in-sample period, which is what the horizon is run under. The
/// two precisions are read whole and cut inside the sampler, which takes the last
/// block of whatever it is handed; the factor path is read whole because the
/// forecast starts from the last p factors of it.
DfmTvpStochvolDraws read_forecast_coefficients(const ModelFile &file,
                                               const DfmTvpStochvolInput &input);

void write_coefficients(const ModelFile &file, const DfmTvpStochvolDraws &draws);

} // namespace bayests::hdf5_io::dfm_tvp_stochvol

#endif // BAYESTS_IO_HDF5_DFM_TVP_STOCHVOL_IO_H
