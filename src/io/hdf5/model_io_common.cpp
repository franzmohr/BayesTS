// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/model_io_common.h"

#include "io/hdf5/hdf5_and_armadillo.h"

#include <cmath>
#include <set>
#include <stdexcept>

namespace bayests::hdf5_io
{

int optional_attribute_int(const ModelFile &file, const std::string &group,
                           const std::string &name, int fallback)
{
    return attribute_exists(file, group, name) ? get_attribute_int(file, group, name) : fallback;
}

std::string optional_attribute_string(const ModelFile &file, const std::string &group,
                                      const std::string &name, const std::string &fallback)
{
    return attribute_exists(file, group, name) ? get_attribute_string(file, group, name) : fallback;
}

double optional_attribute_double(const ModelFile &file, const std::string &group,
                                 const std::string &name, double fallback)
{
    return attribute_exists(file, group, name) ? get_attribute_double(file, group, name) : fallback;
}

bool optional_attribute_bool(const ModelFile &file, const std::string &group,
                             const std::string &name, bool fallback)
{
    return attribute_exists(file, group, name) ? get_attribute_bool(file, group, name) : fallback;
}

double read_double(const ModelFile &file, const std::string &dataset, const std::string &attr_name)
{
    return get_attribute_double(file, dataset, attr_name);
}

arma::vec read_vec(const ModelFile &file, const std::string &dataset)
{
    return arma::vectorise(hdf5_dataset_to_armadillo_matrix_double(file, dataset));
}

arma::mat read_mat(const ModelFile &file, const std::string &dataset)
{
    return hdf5_dataset_to_armadillo_matrix_double(file, dataset);
}

bool read_vec_if_present(const ModelFile &file, const std::string &dataset, arma::vec &out)
{
    if (!file.exist(dataset))
    {
        return false;
    }
    out = read_vec(file, dataset);
    return true;
}

bool read_mat_if_present(const ModelFile &file, const std::string &dataset, arma::mat &out)
{
    if (!file.exist(dataset))
    {
        return false;
    }
    out = read_mat(file, dataset);
    return true;
}

bool read_forecast_regressors(const ModelFile &file, int k, arma::mat &out)
{
    if (read_mat_if_present(file, "/data/forecast/x", out))
    {
        return true;
    }

    arma::mat sur;
    if (!read_mat_if_present(file, "/data/forecast/z", sur))
    {
        return false;
    }

    if (k <= 0)
    {
        throw std::invalid_argument(
            "/data/forecast/z is the SUR layout and can only be read against a positive k, got " +
            std::to_string(k));
    }
    const arma::uword width = static_cast<arma::uword>(k);
    if (sur.n_rows % width != 0 || sur.n_cols % width != 0)
    {
        throw std::invalid_argument(
            "/data/forecast/z is the SUR layout, so both of its dimensions have to be multiples "
            "of k = " + std::to_string(k) + ", got " + std::to_string(sur.n_rows) + " by " +
            std::to_string(sur.n_cols));
    }

    // z is kron(x, I_k), so the block at row i and column j is x(i, j) * I_k and
    // x(i, j) is the one element of it that every block has in the same place.
    // Taking the corner rather than, say, a block mean is deliberate: a file
    // whose z is not a kron of anything is a file this cannot rescue, and
    // averaging would turn it into plausible numbers instead of wrong ones.
    out.set_size(sur.n_rows / width, sur.n_cols / width);
    for (arma::uword i = 0; i < out.n_rows; i++)
    {
        for (arma::uword j = 0; j < out.n_cols; j++)
        {
            out(i, j) = sur(i * width, j * width);
        }
    }
    return true;
}

arma::mat read_path(const ModelFile &file, const std::string &dataset, arma::uword rows,
                    arma::uword periods)
{
    const arma::vec values = read_vec(file, dataset);

    if (periods == 0)
    {
        return {};
    }

    // Counted before the reshape: see the header. A path 24 values short used to
    // pass `bayests check` and a run with exit code 0, its missing starting
    // values zeros.
    if (values.n_elem != rows * periods)
    {
        throw std::invalid_argument(
            "'" + file.resolve(dataset) + "' holds " + std::to_string(values.n_elem) +
            " values, but a path of " + std::to_string(rows) + " per period over " +
            std::to_string(periods) + " periods needs " + std::to_string(rows * periods));
    }

    return arma::reshape(values, rows, periods);
}

arma::uword last_sample_period(const arma::uword periods)
{
    if (periods == 0)
    {
        throw std::invalid_argument(
            "a forecast of a time-varying quantity starts from its last in-sample period, and "
            "without /data/train/y there is no sample to find that period in");
    }
    return periods - 1;
}

arma::uvec read_positions(const ModelFile &file, const std::string &dataset)
{
    const arma::vec one_based = arma::vectorise(
        hdf5_dataset_to_armadillo_matrix_integer(file, dataset));

    // Checked before the subtraction: converting a zero to an unsigned index
    // wraps to a value no bounds check further down would recognise.
    if (one_based.n_elem > 0 && one_based.min() < 1.0)
    {
        throw std::invalid_argument("'" + dataset + "' holds a position below 1; "
                                    "coefficient positions are counted from one");
    }

    return arma::conv_to<arma::uvec>::from(one_based - 1);
}

void ensure_group(const ModelFile &file, const std::string &group)
{
    if (!file.exist(group))
    {
        file.createGroup(group);
    }
}

VarSpec read_spec(const ModelFile &file, const char *covar_error)
{
    VarSpec spec;

    spec.k = get_attribute_int(file, "/model", "k");
    spec.iterations = get_attribute_int(file, "/model", "iterations");
    spec.burnin = get_attribute_int(file, "/model", "burnin");
    spec.thin = optional_attribute_int(file, "/model", "thin", 1);
    spec.p = optional_attribute_int(file, "/model", "p", 0);
    spec.m = optional_attribute_int(file, "/model", "m", 0);
    spec.s = optional_attribute_int(file, "/model", "s", 0);
    spec.h = optional_attribute_int(file, "/model", "h", 0);

    // Absent from every file but a quantile regression model's, and left at the
    // median there, which is the value at which the asymmetric Laplace is
    // symmetric and the model is the one every other sampler here already is.
    spec.quantile = optional_attribute_double(file, "/model", "quantile", 0.5);

    // Deterministic terms entering outside the cointegration space, under the one
    // name every model uses for them. A VEC's terms restricted to that space are
    // counted separately, below.
    spec.n = optional_attribute_int(file, "/model", "n", 0);

    // Absent from every VAR file, and left at zero there, which is what says
    // "no cointegration relation" rather than "one of rank zero".
    spec.rank = optional_attribute_int(file, "/model", "rank", 0);
    spec.k_beta = optional_attribute_int(file, "/model", "k_beta", 0);
    spec.n_restricted = optional_attribute_int(file, "/model", "n_restricted", 0);

    // Absent from every file that is not a dynamic factor model's, and left at
    // zero there, which is what says "the variables in this model are all
    // observed" rather than "a factor model with no factors".
    spec.n_factors = optional_attribute_int(file, "/model", "n_factors", 0);

    // A factor augmented VAR's observed factors, and absent from every other
    // file including a dynamic factor model's -- a factor model with none of
    // them is exactly what a DFM is.
    spec.n_obs_factors = optional_attribute_int(file, "/model", "n_obs_factors", 0);

    spec.varsel = var_selection_from_string(
        optional_attribute_string(file, "/model", "varsel", "none"));
    spec.structural = optional_attribute_bool(file, "/model", "structural", false);

    // Absent from every file written before the choice existed. Those files
    // forecast as `hold` then and read as `simulate` now: the predictive
    // distribution of the model the file estimates is what a forecast is for,
    // and `hold` is there to be asked for. See ForecastStates.
    spec.forecast_states = forecast_states_from_string(
        optional_attribute_string(file, "/model", "forecast_states", "simulate"));

    if (covar_error != nullptr)
    {
        spec.covar = optional_attribute_string(file, "/model", "error", "") == covar_error;
    }

    return spec;
}

NormalPrior read_normal_prior(const ModelFile &file, const std::string &group)
{
    NormalPrior prior;
    prior.mu = read_vec(file, group + "/mu");
    prior.v_inv = read_mat(file, group + "/v_inv");
    return prior;
}

ConstantCointSpacePrior read_coint_space_prior_constant(const ModelFile &file, const std::string &group)
{
    ConstantCointSpacePrior prior;

    // A dataset, as every other prior in these files is -- the group holds
    // /v_inv and /p_tau_inv next to the /type the reader dispatches on -- and not
    // an attribute of the group.
    prior.v_inv = get_dataset_double(file, group + "/v_inv");

    // k_ect x k_ect, the shape of the cointegration space, not n_beta square.
    prior.p_tau_inv = read_mat(file, group + "/p_tau_inv");
    return prior;
}

TvpCointSpacePrior read_coint_space_prior_tvp(const ModelFile &file, const std::string &group)
{
    TvpCointSpacePrior prior;

    prior.initial_state = read_normal_prior(file, group);

    if (file.exist(group + "/rho"))
    {
        prior.rho = get_dataset_double(file, group + "/rho");
    }

    // The support of the uniform prior on rho, and with it the switch that
    // turns rho's draw on. Both ends or neither: one alone would leave the
    // sampler to invent the other, and which end is missing changes the model
    // rather than a detail of it.
    const bool has_min = file.exist(group + "/rho_min");
    const bool has_max = file.exist(group + "/rho_max");

    if (has_min != has_max)
    {
        throw std::invalid_argument(
            "the prior support of rho needs both ends: " + group + "/rho" +
            (has_min ? "_max" : "_min") + " is missing. Leave both out to hold rho fixed at " +
            group + "/rho");
    }

    if (has_min)
    {
        prior.rho_prior.draw = true;
        prior.rho_prior.min = get_dataset_double(file, group + "/rho_min");
        prior.rho_prior.max = get_dataset_double(file, group + "/rho_max");
    }

    // Koop, Leon-Gonzalez and Strachan's informative marginal prior: the
    // transition of the state equation with rho taken out, k_beta square. Absent
    // is the identity, which is what every file written before it means.
    read_mat_if_present(file, group + "/p_tau", prior.p_tau);

    return prior;
}

GammaPrior read_gamma_prior(const ModelFile &file, const std::string &group)
{
    GammaPrior prior;
    read_vec_if_present(file, group + "/shape", prior.shape);
    read_vec_if_present(file, group + "/rate", prior.rate);
    return prior;
}

VarSelPrior read_varsel_prior(const ModelFile &file, const std::string &group,
                              VarSelection scheme)
{
    VarSelPrior prior;
    prior.inprior = read_vec(file, group + "/inprior");
    prior.include = read_positions(file, group + "/include");

    if (scheme == VarSelection::ssvs)
    {
        prior.ssvs.tau0 = read_vec(file, group + "/tau0");
        prior.ssvs.tau1 = read_vec(file, group + "/tau1");
    }

    return prior;
}

arma::mat read_draws(const ModelFile &file, const std::string &dataset)
{
    return arma::trans(read_mat(file, dataset));
}

arma::mat read_draws_at_period(const ModelFile &file, const std::string &dataset,
                               arma::uword period, arma::uword width)
{
    const arma::mat stored = read_mat(file, dataset);
    return arma::trans(stored.cols(period * width, (period + 1) * width - 1));
}

bool read_draws_if_present(const ModelFile &file, const std::string &dataset, arma::mat &out)
{
    if (!dataset_has_data(file, dataset))
    {
        return false;
    }
    out = read_draws(file, dataset);
    return true;
}

void require_log_volatility_variances(const ModelFile &file, const VarSpec &spec,
                                      const std::string &dataset)
{
    if (spec.forecast_states != ForecastStates::simulate || dataset_has_data(file, dataset))
    {
        return;
    }
    throw std::runtime_error(
        "'" + file.resolve(dataset) +
        "' is missing: simulating the volatility forward over the forecast horizon needs the "
        "variance of the log-volatility innovations, and these coefficients were drawn by a "
        "BayesTS that did not store it. Delete /posterior and run coefficients again, or set "
        "/model/forecast_states to \"hold\" to forecast from the last in-sample volatility");
}

arma::mat read_precision(const ModelFile &file, const VarSpec &spec, arma::uword tt,
                         bool time_varying)
{
    const std::string dataset = "/posterior/u_sigma_inv/coeffs";
    if (!dataset_has_data(file, dataset))
    {
        return {};
    }
    if (!time_varying)
    {
        return read_draws(file, dataset);
    }
    const arma::uword k = static_cast<arma::uword>(spec.k);
    return read_draws_at_period(file, dataset, last_sample_period(tt), k * k);
}

void write_draws(const ModelFile &file, const std::string &dataset, const arma::mat &draws)
{
    write_armadillo_matrix_to_hdf5(file, dataset, arma::trans(draws), true);
}

void write_forecast(const ModelFile &file, const ForecastDraws &forecast)
{
    ensure_group(file, "/posterior");
    write_armadillo_matrix_to_hdf5(file, "/posterior/forecast", arma::trans(forecast.values), false);
}

void write_log_likelihood(const ModelFile &file, const arma::mat &loglik)
{
    ensure_group(file, "/posterior");
    write_armadillo_matrix_to_hdf5(file, "/posterior/loglik", loglik, false);
}

} // namespace bayests::hdf5_io

namespace bayests::hdf5_io
{

bool is_model_attribute(const std::string &name)
{
    // read_spec() above reads all of these but `algorithm`, which picks the
    // reader before there is one, and `seed`, which read_model_seed() below reads
    // for the command line. Keep them in the same file, and add a name here in
    // the edit that teaches a reader to read it: `bayests check` warns about
    // every /model attribute this does not list.
    static const std::set<std::string> names = {
        "algorithm", "k",         "iterations", "burnin",       "thin",       "p",
        "m",         "s",         "h",          "quantile",     "n",
        "rank",      "k_beta",    "n_restricted", "n_factors",  "n_obs_factors",
        "varsel",    "structural", "error",     "seed",       "forecast_states",
    };
    return names.count(name) > 0;
}

std::optional<std::uint64_t> read_model_seed(const ModelFile &file)
{
    if (!attribute_exists(file, "/model", "seed"))
    {
        return std::nullopt;
    }

    const HighFive::Attribute attr = file.getGroup("/model").getAttribute("seed");
    const HighFive::DataType type = attr.getDataType();
    const auto refuse = [](const std::string &why) {
        return std::invalid_argument("/model/seed " + why +
                                     "; a seed is a non-negative whole number, or left out");
    };

    if (attr.getSpace().getElementCount() != 1)
    {
        throw refuse("holds more than one value");
    }

    if (type.getClass() == HighFive::DataTypeClass::Integer)
    {
        if (H5Tget_sign(type.getId()) == H5T_SGN_NONE)
        {
            std::uint64_t value = 0;
            attr.read(value);
            return value;
        }

        std::int64_t value = 0;
        attr.read(value);
        if (value < 0)
        {
            throw refuse("is " + std::to_string(value));
        }
        return static_cast<std::uint64_t>(value);
    }

    if (type.getClass() == HighFive::DataTypeClass::Float)
    {
        // As get_attribute_int() accepts for a dimension, and for the same reason:
        // `seed = 20260901` in R is a double. Above 2^53 a double no longer holds
        // every whole number, so the seed it holds may not be the one typed.
        double value = 0.0;
        attr.read(value);
        if (!(value >= 0.0) || value != std::floor(value) || value > 9007199254740992.0)
        {
            throw refuse("is " + std::to_string(value));
        }
        return static_cast<std::uint64_t>(value);
    }

    throw refuse("is not a number");
}

} // namespace bayests::hdf5_io
