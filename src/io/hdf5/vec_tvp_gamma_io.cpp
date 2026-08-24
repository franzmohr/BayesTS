// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/vec_tvp_gamma_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::vec_tvp_gamma
{

namespace
{

/// The coefficient and cointegration paths, cut to one period or read whole.
/// `period_width` of zero means the whole path.
void read_paths(const HighFive::File &file, const VecTvpGammaInput &input,
                VecTvpGammaDraws &draws, bool last_period_only)
{
    const arma::uword tt = input.train.periods(input.spec.k);
    const arma::uword n_a = static_cast<arma::uword>(input.spec.nparams_per_period_vec());
    const arma::uword n_beta = static_cast<arma::uword>(input.spec.n_beta());

    if (n_a > 0 && dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = last_period_only
                      ? read_draws_at_period(file, "/posterior/a/coeffs", tt - 1, n_a)
                      : read_draws(file, "/posterior/a/coeffs");
    }
    if (n_beta > 0 && dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        draws.beta = last_period_only
                         ? read_draws_at_period(file, "/posterior/beta/coeffs", tt - 1, n_beta)
                         : read_draws(file, "/posterior/beta/coeffs");
    }
}

} // namespace

VecTvpGammaInput read_input(const HighFive::File &file)
{
    VecTvpGammaInput input;

    input.spec = read_spec(file, "gamma+covar");

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/w", input.train.w);
    read_mat_if_present(file, "/data/train/z", input.train.z);
    read_mat_if_present(file, "/data/forecast/z", input.forecast.z);

    const arma::uword tt = input.train.y.n_elem > 0 && input.spec.k > 0
                               ? input.train.periods(input.spec.k)
                               : 0;

    if (input.use_a())
    {
        const arma::uword n_a = input.train.nparams();

        input.initial.a = read_path(file, "/initial/a", n_a, tt);
        input.initial.a_sigma_inv = read_mat(file, "/initial/a_sigma_inv");
        input.initial.a_init = read_vec(file, "/initial/a_init");

        // One group carries both halves of the state equation: how far the
        // coefficients may drift, and where they start.
        input.a_prior.sigma = read_gamma_prior(file, "/priors/a");
        input.a_prior.initial_state = read_normal_prior(file, "/priors/a");

        // BVS is the only scheme this model implements. An SSVS file is left
        // unread here and rejected by validate(), which can say why.
        if (input.spec.varsel == VarSelection::bvs)
        {
            input.initial.a_lambda = read_vec(file, "/initial/a_lambda");
            input.a_varsel_prior = read_varsel_prior(file, "/priors/a", input.spec.varsel);
        }
    }

    if (input.use_beta())
    {
        const arma::uword n_beta = static_cast<arma::uword>(input.spec.n_beta());

        input.initial.beta = read_path(file, "/initial/beta", n_beta, tt);
        input.initial.beta_init = read_vec(file, "/initial/beta_init");
        input.beta_prior = read_coint_space_prior_tvp(file, "/priors/beta");
    }

    if (input.use_psi())
    {
        const arma::uword n_psi = static_cast<arma::uword>(input.spec.n_psi());

        input.initial.psi = read_path(file, "/initial/psi", n_psi, tt);
        input.initial.psi_sigma_inv = read_mat(file, "/initial/psi_sigma_inv");
        input.initial.psi_init = read_vec(file, "/initial/psi_init");

        input.psi_prior.sigma = read_gamma_prior(file, "/priors/psi");
        input.psi_prior.initial_state = read_normal_prior(file, "/priors/psi");

        // Selection for the covariance block is declared in its own group, so
        // it can differ from the model's.
        input.psi_varsel = var_selection_from_string(
            optional_attribute_string(file, "/model/priors/psi", "varsel", "none"));

        if (input.psi_varsel == VarSelection::bvs)
        {
            input.initial.psi_lambda = read_vec(file, "/initial/psi_lambda");
            input.psi_varsel_prior = read_varsel_prior(file, "/priors/psi", input.psi_varsel);
        }
    }

    // Only draw_coefficients needs these. A file that holds nothing but a
    // fitted posterior can still be forecast from, so a missing prior is left
    // for validate() to complain about if it turns out to matter.
    input.u_sigma_prior = read_gamma_prior(file, "/priors/u_sigma");
    read_mat_if_present(file, "/initial/u_omega_inv", input.initial.u_omega_inv);

    return input;
}

VecTvpGammaDraws read_loglik_coefficients(const HighFive::File &file,
                                          const VecTvpGammaInput &input)
{
    VecTvpGammaDraws draws;

    read_paths(file, input, draws, false);
    draws.u_sigma_inv =
        read_precision(file, input.spec, input.train.periods(input.spec.k), input.use_psi());

    return draws;
}

VecTvpGammaDraws read_forecast_coefficients(const HighFive::File &file,
                                            const VecTvpGammaInput &input)
{
    VecTvpGammaDraws draws;

    read_paths(file, input, draws, true);
    draws.u_sigma_inv =
        read_precision(file, input.spec, input.train.periods(input.spec.k), input.use_psi());

    return draws;
}

void write_coefficients(HighFive::File &file, const VecTvpGammaDraws &draws)
{
    ensure_group(file, "/posterior");

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
        write_draws(file, "/posterior/a/sigma", draws.a_sigma);
        if (draws.a_lambda.n_elem > 0)
        {
            write_draws(file, "/posterior/a/lambda", draws.a_lambda);
        }
    }

    // The cointegration path. Without it `a` carries only the loadings, so
    // nothing downstream could reconstruct Pi -- and neither the forecast nor
    // the log likelihood, both of which rebuild the loadings' regressors from
    // it, could be computed at all.
    if (draws.has_beta())
    {
        write_draws(file, "/posterior/beta/coeffs", draws.beta);
    }

    if (draws.has_psi())
    {
        write_draws(file, "/posterior/psi/coeffs", draws.psi);
        write_draws(file, "/posterior/psi/sigma", draws.psi_sigma);
        if (draws.psi_lambda.n_elem > 0)
        {
            write_draws(file, "/posterior/psi/lambda", draws.psi_lambda);
        }
    }

    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::vec_tvp_gamma
