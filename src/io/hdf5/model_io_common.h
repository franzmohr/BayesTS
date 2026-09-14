// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_IO_HDF5_MODEL_IO_COMMON_H
#define BAYESTS_IO_HDF5_MODEL_IO_COMMON_H

#include "bayests/inputs.h"
#include "bayests/results.h"
#include "io/hdf5/hdf5_and_armadillo.h"

#include <highfive/H5File.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace bayests::hdf5_io
{

/// Attributes that older files predate. Reading them through a default keeps a
/// fixture from failing on a field the model does not use.
int optional_attribute_int(const ModelFile &file, const std::string &group,
                           const std::string &name, int fallback);
std::string optional_attribute_string(const ModelFile &file, const std::string &group,
                                      const std::string &name, const std::string &fallback);
bool optional_attribute_bool(const ModelFile &file, const std::string &group,
                             const std::string &name, bool fallback);
double optional_attribute_double(const ModelFile &file, const std::string &group,
                                 const std::string &name, double fallback);

arma::vec read_vec(const ModelFile &file, const std::string &dataset);
arma::mat read_mat(const ModelFile &file, const std::string &dataset);

/// Reads `dataset` when it is there and leaves `out` alone when it is not.
/// Returns whether anything was read, for the callers that branch on it.
bool read_vec_if_present(const ModelFile &file, const std::string &dataset, arma::vec &out);
bool read_mat_if_present(const ModelFile &file, const std::string &dataset, arma::mat &out);

/// The out-of-sample regressors, in the compact layout ForecastData::x is
/// written in: one row per horizon, one column per regressor.
///
/// Reads `/data/forecast/x` when it is there. A file written before that layout
/// carries `/data/forecast/z` instead -- the same regressors kroneckered up with
/// I_k, at k times the rows and k times the columns -- and is compacted back on
/// the way in, so a model file recorded by an earlier version still forecasts.
/// `k` is what decides which of the two a `z` is, so it has to come from the
/// spec rather than from the dataset's own shape.
///
/// Leaves `out` alone when neither is there, which is what a model file with no
/// forecast requested looks like; require_forecast_regressors() is what turns
/// that into an error for a model that needed them.
bool read_forecast_regressors(const ModelFile &file, int k, arma::mat &out);

/// A time-varying starting value. A state path is stored as one long row, so
/// reading it back means saying how wide a period is; the sampler is handed
/// the rectangle, `rows` by `periods`.
///
/// The dataset has to hold exactly `rows * periods` values, and one that does not
/// is refused. The reshape this used to be alone pads a short path with zeros and
/// cuts a long one, after which validate() sees a matrix of exactly the shape it
/// asks for whatever the file held. An empty matrix comes back when there is no
/// sample to measure against (`periods` of zero); validate() refuses that run.
arma::mat read_path(const ModelFile &file, const std::string &dataset, arma::uword rows,
                    arma::uword periods);

/// The index of the last in-sample period, where a forecast of a time-varying
/// quantity starts. Throws for a file with no training sample, where
/// `periods - 1` would wrap round to the largest index there is.
arma::uword last_sample_period(arma::uword periods);

/// Selection positions are stored one-based, the way R and the file format
/// count. The samplers index from zero.
arma::uvec read_positions(const ModelFile &file, const std::string &dataset);

void ensure_group(const ModelFile &file, const std::string &group);

/// The /model attributes that describe the shape of the model, whichever model
/// it is.
///
/// `covar_error` is the spelling of the error specification that turns
/// spec.covar on -- "gamma+covar" for the gamma models, "sv+covar" for
/// stochastic volatility -- or nullptr for a model with no covariance block to
/// look for. Only k, iterations and burnin are demanded; everything else is
/// read through a default, so a file written for a simpler model still yields
/// a usable spec.
VarSpec read_spec(const ModelFile &file, const char *covar_error);

/// Whether some reader looks for a /model attribute of this name -- the ones
/// read_spec() reads, `algorithm` and `seed`. `bayests check` warns about any
/// other, since an attribute nothing reads is usually a misspelling of one that
/// is then read through its default.
bool is_model_attribute(const std::string &name);

/// /model/seed: what the command line seeds each stage of this model's run
/// from, or nothing when the file names no seed. Read here, where every other
/// attribute is read, and applied by src/model_seed.cpp -- seeding the
/// generator is the host's business, not the samplers'. A whole number stored
/// as a float is accepted, as R writes one unless told `L`; anything that is
/// not a non-negative whole number throws rather than seeding from a guess.
std::optional<std::uint64_t> read_model_seed(const ModelFile &file);

/// The (mu, v_inv) pair every normal prior is stored as.
NormalPrior read_normal_prior(const ModelFile &file, const std::string &group);

/// The (v_inv, p_tau_inv) pair every constant cointegration-space prior is stored as.
ConstantCointSpacePrior read_coint_space_prior_constant(const ModelFile &file, const std::string &group);

/// The time-varying counterpart: the (mu, v_inv) of the state before the sample
/// plus the state equation's autoregression. The same group name as the constant
/// prior above, holding different datasets -- a cointegration space that moves is
/// a state equation rather than a shrinkage towards a central location, so there
/// is nothing shared to read. `rho` is optional; TvpCointSpacePrior says what it
/// defaults to and why. `rho_min` and `rho_max` are optional as a pair and turn
/// rho from a fixed hyperparameter into a drawn one, with `rho` then the value
/// the chain starts at. `p_tau` is optional and centres the marginal prior of the
/// space on a given one; absent, the transition is rho alone.
TvpCointSpacePrior read_coint_space_prior_tvp(const ModelFile &file, const std::string &group);

/// The (shape, rate) pair every gamma prior is stored as.
GammaPrior read_gamma_prior(const ModelFile &file, const std::string &group);

/// The prior inclusion probabilities, the positions selection applies to and,
/// for SSVS, the two mixture components. `group` is the prior group of the
/// block being selected -- "/priors/a" or "/priors/psi".
VarSelPrior read_varsel_prior(const ModelFile &file, const std::string &group,
                              VarSelection scheme);

/// Posterior draws read back in the samplers' draw-per-column layout. On disk
/// draws run along the rows, which is what R and coda expect of an mcmc
/// object; the samplers work the other way round.
arma::mat read_draws(const ModelFile &file, const std::string &dataset);

/// The same, restricted to one period out of a path that stores `width`
/// numbers per period. Used where a forecast starts from the last in-sample
/// value of a time-varying quantity.
arma::mat read_draws_at_period(const ModelFile &file, const std::string &dataset,
                               arma::uword period, arma::uword width);

/// read_draws() where the dataset may not be there, leaving `out` alone if it is
/// not.
bool read_draws_if_present(const ModelFile &file, const std::string &dataset, arma::mat &out);

/// Throws, naming the file, when a stochastic volatility model is asked to
/// simulate its volatility forward from a posterior that does not hold the
/// variance of the log-volatility innovations -- one drawn before `dataset` was
/// written. The sampler would refuse the same draws, but could not say which
/// file or what to do about it. A factor model asks twice, once for each of its
/// two volatility groups.
void require_log_volatility_variances(const ModelFile &file, const VarSpec &spec,
                                      const std::string &dataset = "/posterior/u_sigma_inv/sigma");

/// The posterior precision as a forecast or a likelihood wants it. When the
/// precision moves with time the stored path is cut to its last in-sample
/// period, `k` by `k` wide; when it does not, every draw is one matrix and the
/// whole dataset is read. Yields an empty matrix if the file holds no
/// posterior yet.
/// Takes the spec rather than a bare `k` so that the two dimensions cannot be
/// passed the wrong way round: `tt` is then the only loose number.
arma::mat read_precision(const ModelFile &file, const VarSpec &spec, arma::uword tt,
                         bool time_varying);

void write_draws(const ModelFile &file, const std::string &dataset, const arma::mat &draws);

/// Neither of these depends on which model produced the numbers.
void write_forecast(const ModelFile &file, const ForecastDraws &forecast);
void write_log_likelihood(const ModelFile &file, const arma::mat &loglik);

} // namespace bayests::hdf5_io

#endif // BAYESTS_IO_HDF5_MODEL_IO_COMMON_H
