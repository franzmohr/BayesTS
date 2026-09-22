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
/// Reads `/data/test/y`, the observations the forecast horizon realised, into
/// `out`. False and `out` untouched where the file has none, which is every
/// file that is a forecast rather than a forecast being scored.
///
/// One row per period and one column per variable, the layout and the variable
/// order of `/data/train/y`, and refused otherwise -- the stacked spelling that
/// `/data/train/y` also accepts is not taken here, because a `k` column matrix
/// and a stacked vector of the same length cannot be told apart by anything but
/// convention once `h` is unknown, and guessing would score a forecast against
/// a reshuffle of the right numbers.
///
/// Fewer rows than `h` is not an error: the last windows of an expanding window
/// exercise realise fewer periods than they forecast, and the ones that are
/// there are the ones that can be scored. More than `h` is, since the file then
/// describes a horizon other than the one `/model` asks for. Neither is checked
/// against `h` when there is no horizon at all -- a file can carry what it will
/// be scored on before it is told how far to forecast, and refusing that here
/// would fail the sampler over data no sampler reads. `bayests check` reports
/// it instead.
bool read_test_observations(const ModelFile &file, const VarSpec &spec, arma::mat &out);

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

/// The three datasets a block drawn under the non-centred parameterisation adds
/// beside its `sigma` -- `omega`, `omega_log_zero` and `omega_log_zero_joint`
/// under `group`. Writes nothing for a centred block.
void write_noncentred(const ModelFile &file, const std::string &group,
                      const NoncentredStateDraws &draws);

/// A per-period posterior quantity, for the discounted models, whose output is
/// a posterior rather than a chain.
///
/// `path` is one row per quantity and one column per period, and the dataset is
/// written in the orientation write_draws() gives a chain: in HDF5 dataspace
/// terms one row per quantity and one column per period, which R's readers
/// reverse into periods-in-rows the way they reverse a posterior's draws. What
/// it does not write is `start`, `end` and `thin` -- there is no chain behind
/// these numbers, so a coda object built from them would be claiming a sweep
/// that never ran.
void write_posterior_path(const ModelFile &file, const std::string &dataset,
                          const arma::mat &path);

/// The inverse, giving back the quantity-by-period matrix.
arma::mat read_posterior_path(const ModelFile &file, const std::string &dataset);
bool read_posterior_path_if_present(const ModelFile &file, const std::string &dataset,
                                    arma::mat &out);

/// Neither of these depends on which model produced the numbers, and both write
/// the `start`/`end`/`thin` attributes every other posterior dataset carries.
///
/// The paths go to `/posterior/forecast/forecasts`. The group is the place for
/// everything the forecast periods produce: a host that scores the paths
/// against what was realised writes `errors` and `loglik` beside them, which is
/// why the leaf is named after what it holds rather than called `draws` -- all
/// three are draws. `/posterior/loglik`, the in-sample pointwise likelihood,
/// stays outside it: it conditions on states that have seen the observation it
/// evaluates, which the forecast one does not.
void write_forecast(const ModelFile &file, const ForecastDraws &forecast);
void write_log_likelihood(const ModelFile &file, const arma::mat &loglik);

/// The score of a forecast, draws x scored periods, to
/// `/posterior/forecast/loglik`. Beside the paths it scores rather than beside
/// `/posterior/loglik`, which is the in-sample pointwise likelihood and a
/// different statistic: that one evaluates each observation under states that
/// have already seen it, which is what a forecast score must not do.
///
/// Expects the group to be there, which it is: the paths are written first, and
/// a score without them would be a score of nothing.
void write_forecast_loglik(const ModelFile &file, const arma::mat &loglik);

/// What a previous `coefficients` stage left in a model's /posterior.
///
/// Every stage skips when its output is already there, and for `coefficients`
/// "there" used to mean one dataset -- in half the models not the last one the
/// stage writes, and in the two discounted ones the first. A run stopped
/// between two writes left a file whose rerun skipped and whose later stages
/// failed, for good. The stage now marks /posterior before its first write and
/// again after its last, and `interrupted` is a file still carrying the first.
enum class CoefficientsState
{
    absent,      ///< Nothing written, or not the dataset the model's skip reads.
    interrupted, ///< A run started writing and did not finish.
    complete,    ///< Written in full -- or by a version before the marker.
};

/// The attribute on /posterior the marker is kept in, and its two values.
inline constexpr const char *kCoefficientsAttribute = "coefficients";
inline constexpr const char *kCoefficientsWriting = "writing";
inline constexpr const char *kCoefficientsComplete = "complete";

/// Reads the marker, and `probe` -- the dataset the model's skip always read.
///
/// A file without the marker is one written before it existed. Those were
/// written by runs that finished, or they would be reported as broken already,
/// so the probe decides for them exactly as it did before: present is complete.
CoefficientsState coefficients_state(const ModelFile &file, const std::string &probe);

/// Marks /posterior as being written, and flushes, so the mark is on disk
/// before any draw is. Call before the first write of the stage.
void mark_coefficients_writing(const ModelFile &file);

/// Marks /posterior as written in full, and flushes. Call after the last write.
void mark_coefficients_complete(const ModelFile &file);

/// Throws if a run of `coefficients` started writing this model's draws and did
/// not finish. Called by every reader of draws, so `loglik` and `forecasts` do
/// not run on half a posterior.
void require_coefficients_finished(const ModelFile &file);

} // namespace bayests::hdf5_io

#endif // BAYESTS_IO_HDF5_MODEL_IO_COMMON_H
