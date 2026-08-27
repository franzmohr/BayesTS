// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/model_io_common.h"

#include "io/hdf5/hdf5_and_armadillo.h"

#include <stdexcept>

namespace bayests::hdf5_io
{

int optional_attribute_int(const HighFive::File &file, const std::string &group,
                           const std::string &name, int fallback)
{
    return attribute_exists(file, group, name) ? get_attribute_int(file, group, name) : fallback;
}

std::string optional_attribute_string(const HighFive::File &file, const std::string &group,
                                      const std::string &name, const std::string &fallback)
{
    return attribute_exists(file, group, name) ? get_attribute_string(file, group, name) : fallback;
}

bool optional_attribute_bool(const HighFive::File &file, const std::string &group,
                             const std::string &name, bool fallback)
{
    return attribute_exists(file, group, name) ? get_attribute_bool(file, group, name) : fallback;
}

double read_double(const HighFive::File &file, const std::string &dataset, const std::string &attr_name)
{
    return get_attribute_double(file, dataset, attr_name);
}

arma::vec read_vec(const HighFive::File &file, const std::string &dataset)
{
    return arma::vectorise(hdf5_dataset_to_armadillo_matrix_double(file, dataset));
}

arma::mat read_mat(const HighFive::File &file, const std::string &dataset)
{
    return hdf5_dataset_to_armadillo_matrix_double(file, dataset);
}

bool read_vec_if_present(const HighFive::File &file, const std::string &dataset, arma::vec &out)
{
    if (!file.exist(dataset))
    {
        return false;
    }
    out = read_vec(file, dataset);
    return true;
}

bool read_mat_if_present(const HighFive::File &file, const std::string &dataset, arma::mat &out)
{
    if (!file.exist(dataset))
    {
        return false;
    }
    out = read_mat(file, dataset);
    return true;
}

arma::mat read_path(const HighFive::File &file, const std::string &dataset, arma::uword rows,
                    arma::uword periods)
{
    return arma::reshape(read_vec(file, dataset), rows, periods);
}

arma::uvec read_positions(const HighFive::File &file, const std::string &dataset)
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

void ensure_group(HighFive::File &file, const std::string &group)
{
    if (!file.exist(group))
    {
        file.createGroup(group);
    }
}

VarSpec read_spec(const HighFive::File &file, const char *covar_error)
{
    VarSpec spec;

    spec.k = get_attribute_int(file, "/model", "k");
    spec.iterations = get_attribute_int(file, "/model", "iterations");
    spec.burnin = get_attribute_int(file, "/model", "burnin");
    spec.p = optional_attribute_int(file, "/model", "p", 0);
    spec.m = optional_attribute_int(file, "/model", "m", 0);
    spec.s = optional_attribute_int(file, "/model", "s", 0);
    spec.h = optional_attribute_int(file, "/model", "h", 0);

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
    spec.varsel = var_selection_from_string(
        optional_attribute_string(file, "/model", "varsel", "none"));
    spec.structural = optional_attribute_bool(file, "/model", "structural", false);

    if (covar_error != nullptr)
    {
        spec.covar = optional_attribute_string(file, "/model", "error", "") == covar_error;
    }

    return spec;
}

NormalPrior read_normal_prior(const HighFive::File &file, const std::string &group)
{
    NormalPrior prior;
    prior.mu = read_vec(file, group + "/mu");
    prior.v_inv = read_mat(file, group + "/v_inv");
    return prior;
}

ConstantCointSpacePrior read_coint_space_prior_constant(const HighFive::File &file, const std::string &group)
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

TvpCointSpacePrior read_coint_space_prior_tvp(const HighFive::File &file, const std::string &group)
{
    TvpCointSpacePrior prior;

    prior.initial_state = read_normal_prior(file, group);

    if (file.exist(group + "/rho"))
    {
        prior.rho = get_dataset_double(file, group + "/rho");
    }

    return prior;
}

GammaPrior read_gamma_prior(const HighFive::File &file, const std::string &group)
{
    GammaPrior prior;
    read_vec_if_present(file, group + "/shape", prior.shape);
    read_vec_if_present(file, group + "/rate", prior.rate);
    return prior;
}

VarSelPrior read_varsel_prior(const HighFive::File &file, const std::string &group,
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

arma::mat read_draws(const HighFive::File &file, const std::string &dataset)
{
    return arma::trans(read_mat(file, dataset));
}

arma::mat read_draws_at_period(const HighFive::File &file, const std::string &dataset,
                               arma::uword period, arma::uword width)
{
    const arma::mat stored = read_mat(file, dataset);
    return arma::trans(stored.cols(period * width, (period + 1) * width - 1));
}

arma::mat read_precision(const HighFive::File &file, const VarSpec &spec, arma::uword tt,
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
    return read_draws_at_period(file, dataset, tt - 1, k * k);
}

void write_draws(HighFive::File &file, const std::string &dataset, const arma::mat &draws)
{
    write_armadillo_matrix_to_hdf5(file, dataset, arma::trans(draws), true);
}

void write_forecast(HighFive::File &file, const ForecastDraws &forecast)
{
    ensure_group(file, "/posterior");
    write_armadillo_matrix_to_hdf5(file, "/posterior/forecast", arma::trans(forecast.values), false);
}

void write_log_likelihood(HighFive::File &file, const arma::mat &loglik)
{
    ensure_group(file, "/posterior");
    write_armadillo_matrix_to_hdf5(file, "/posterior/loglik", loglik, false);
}

} // namespace bayests::hdf5_io
