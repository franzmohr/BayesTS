// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_MODEL_IO_COMMON_H
#define BAYESTS_IO_HDF5_MODEL_IO_COMMON_H

#include "bayests/inputs.h"
#include "bayests/results.h"

#include <highfive/H5File.hpp>

#include <string>

namespace bayests::hdf5_io
{

/// Attributes that older files predate. Reading them through a default keeps a
/// fixture from failing on a field the model does not use.
int optional_attribute_int(const HighFive::File &file, const std::string &group,
                           const std::string &name, int fallback);
std::string optional_attribute_string(const HighFive::File &file, const std::string &group,
                                      const std::string &name, const std::string &fallback);
bool optional_attribute_bool(const HighFive::File &file, const std::string &group,
                             const std::string &name, bool fallback);

arma::vec read_vec(const HighFive::File &file, const std::string &dataset);
arma::mat read_mat(const HighFive::File &file, const std::string &dataset);

/// Reads `dataset` when it is there and leaves `out` alone when it is not.
/// Returns whether anything was read, for the callers that branch on it.
bool read_vec_if_present(const HighFive::File &file, const std::string &dataset, arma::vec &out);
bool read_mat_if_present(const HighFive::File &file, const std::string &dataset, arma::mat &out);

/// A time-varying starting value. A state path is stored as one long row, so
/// reading it back means saying how wide a period is; the sampler is handed
/// the rectangle, `rows` by `periods`.
arma::mat read_path(const HighFive::File &file, const std::string &dataset, arma::uword rows,
                    arma::uword periods);

/// Selection positions are stored one-based, the way R and the file format
/// count. The samplers index from zero.
arma::uvec read_positions(const HighFive::File &file, const std::string &dataset);

void ensure_group(HighFive::File &file, const std::string &group);

/// The /model attributes that describe the shape of the model, whichever model
/// it is.
///
/// `covar_error` is the spelling of the error specification that turns
/// spec.covar on -- "gamma+covar" for the gamma models, "sv+covar" for
/// stochastic volatility -- or nullptr for a model with no covariance block to
/// look for. Only k, iterations and burnin are demanded; everything else is
/// read through a default, so a file written for a simpler model still yields
/// a usable spec.
VarSpec read_spec(const HighFive::File &file, const char *covar_error);

/// The (mu, v_inv) pair every normal prior is stored as.
NormalPrior read_normal_prior(const HighFive::File &file, const std::string &group);

/// The (v_inv, p_tau_inv) pair every constant cointegration-space prior is stored as.
ConstantCointSpacePrior read_coint_space_prior_constant(const HighFive::File &file, const std::string &group);

/// The time-varying counterpart: the (mu, v_inv) of the state before the sample
/// plus the state equation's autoregression. The same group name as the constant
/// prior above, holding different datasets -- a cointegration space that moves is
/// a state equation rather than a shrinkage towards a central location, so there
/// is nothing shared to read. `rho` is optional; TvpCointSpacePrior says what it
/// defaults to and why.
TvpCointSpacePrior read_coint_space_prior_tvp(const HighFive::File &file, const std::string &group);

/// The (shape, rate) pair every gamma prior is stored as.
GammaPrior read_gamma_prior(const HighFive::File &file, const std::string &group);

/// The prior inclusion probabilities, the positions selection applies to and,
/// for SSVS, the two mixture components. `group` is the prior group of the
/// block being selected -- "/priors/a" or "/priors/psi".
VarSelPrior read_varsel_prior(const HighFive::File &file, const std::string &group,
                              VarSelection scheme);

/// Posterior draws read back in the samplers' draw-per-column layout. On disk
/// draws run along the rows, which is what R and coda expect of an mcmc
/// object; the samplers work the other way round.
arma::mat read_draws(const HighFive::File &file, const std::string &dataset);

/// The same, restricted to one period out of a path that stores `width`
/// numbers per period. Used where a forecast starts from the last in-sample
/// value of a time-varying quantity.
arma::mat read_draws_at_period(const HighFive::File &file, const std::string &dataset,
                               arma::uword period, arma::uword width);

/// The posterior precision as a forecast or a likelihood wants it. When the
/// precision moves with time the stored path is cut to its last in-sample
/// period, `k` by `k` wide; when it does not, every draw is one matrix and the
/// whole dataset is read. Yields an empty matrix if the file holds no
/// posterior yet.
/// Takes the spec rather than a bare `k` so that the two dimensions cannot be
/// passed the wrong way round: `tt` is then the only loose number.
arma::mat read_precision(const HighFive::File &file, const VarSpec &spec, arma::uword tt,
                         bool time_varying);

void write_draws(HighFive::File &file, const std::string &dataset, const arma::mat &draws);

/// Neither of these depends on which model produced the numbers.
void write_forecast(HighFive::File &file, const ForecastDraws &forecast);
void write_log_likelihood(HighFive::File &file, const arma::mat &loglik);

} // namespace bayests::hdf5_io

#endif // BAYESTS_IO_HDF5_MODEL_IO_COMMON_H
