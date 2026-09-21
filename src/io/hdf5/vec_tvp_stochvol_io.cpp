// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/vec_tvp_stochvol_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::vec_tvp_stochvol
{

VecTvpStochvolInput read_input(const ModelFile &file)
{
    VecTvpStochvolInput input;

    input.spec = read_spec(file, "sv+covar");

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/w", input.train.w);
    read_mat_if_present(file, "/data/train/z", input.train.z);
    read_forecast_regressors(file, input.spec.k, input.forecast.x);
    read_test_observations(file, input.spec, input.test.y);

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

        // The non-centred parameterisation, in place of shape and rate. Which
        // one the file chose is decided by validate(), which refuses both.
        read_vec_if_present(file, "/priors/a/omega_v", input.a_prior.omega_v);

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
        read_vec_if_present(file, "/priors/psi/omega_v", input.psi_prior.omega_v);

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
    read_vec_if_present(file, "/priors/u_sigma/offset", input.u_sigma_prior.offset);
    read_vec_if_present(file, "/priors/u_sigma/shape", input.u_sigma_prior.state.sigma.shape);
    read_vec_if_present(file, "/priors/u_sigma/rate", input.u_sigma_prior.state.sigma.rate);
    read_vec_if_present(file, "/priors/u_sigma/omega_v", input.u_sigma_prior.state.omega_v);
    read_vec_if_present(file, "/priors/u_sigma/mu", input.u_sigma_prior.state.initial_state.mu);
    read_mat_if_present(file, "/priors/u_sigma/v_inv",
                        input.u_sigma_prior.state.initial_state.v_inv);

    // A state the sampler redraws every iteration, even though the file keeps
    // it next to the prior it is drawn under.
    read_vec_if_present(file, "/priors/u_sigma/sigma", input.initial.h_sigma);
    read_mat_if_present(file, "/initial/h", input.initial.h);
    read_vec_if_present(file, "/initial/h_init", input.initial.h_init);

    return input;
}

VecTvpStochvolDraws read_loglik_coefficients(const ModelFile &file,
                                             const VecTvpStochvolInput &input)
{
    VecTvpStochvolDraws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        draws.beta = read_draws(file, "/posterior/beta/coeffs");
    }
    // The whole precision path, not the last period a forecast starts from:
    // the log likelihood scores every period under its own.
    draws.u_sigma_inv = read_precision(file, input.spec, input.train.periods(input.spec.k), false);

    return draws;
}

VecTvpStochvolDraws read_forecast_coefficients(const ModelFile &file,
                                               const VecTvpStochvolInput &input)
{
    VecTvpStochvolDraws draws;

    const arma::uword tt = input.train.periods(input.spec.k);
    const arma::uword n_a = static_cast<arma::uword>(input.spec.nparams_per_period_vec());
    const arma::uword n_beta = static_cast<arma::uword>(input.spec.n_beta());

    if (n_a > 0 && dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws_at_period(file, "/posterior/a/coeffs", last_sample_period(tt), n_a);
    }
    if (n_beta > 0 && dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        draws.beta = read_draws_at_period(file, "/posterior/beta/coeffs", last_sample_period(tt), n_beta);
    }
    draws.u_sigma_inv = read_precision(file, input.spec, tt, true);

    // Simulating the states forward reads how far each random walk moves per
    // period, which coefficients selection left out, the cointegration state
    // equation's rho where the chain drew it, and the last period of Psi and of
    // the log-volatilities the precision is rebuilt from.
    if (input.spec.forecast_states == ForecastStates::simulate)
    {
        const arma::uword k = static_cast<arma::uword>(input.spec.k);
        if (n_a > 0)
        {
            read_draws_if_present(file, "/posterior/a/sigma", draws.a_sigma);
            read_draws_if_present(file, "/posterior/a/lambda", draws.a_lambda);
        }
        if (n_beta > 0)
        {
            read_draws_if_present(file, "/posterior/beta/rho", draws.rho);
        }
        if (input.use_psi() && dataset_has_data(file, "/posterior/psi/coeffs"))
        {
            draws.psi = read_draws_at_period(file, "/posterior/psi/coeffs", last_sample_period(tt),
                                             k * k);
            read_draws_if_present(file, "/posterior/psi/sigma", draws.psi_sigma);
            read_draws_if_present(file, "/posterior/psi/lambda", draws.psi_lambda);
        }
        if (dataset_has_data(file, "/posterior/u_omega_inv/coeffs"))
        {
            draws.u_omega_inv = read_draws_at_period(file, "/posterior/u_omega_inv/coeffs",
                                                     last_sample_period(tt), k);
        }
        read_draws_if_present(file, "/posterior/u_sigma_inv/sigma", draws.h_sigma);
    }

    return draws;
}

void write_coefficients(const ModelFile &file, const VecTvpStochvolDraws &draws)
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
        write_noncentred(file, "/posterior/a", draws.a_noncentred);
    }

    // The cointegration path. Without it `a` carries only the loadings, so
    // nothing downstream could reconstruct Pi -- and neither the forecast nor
    // the log likelihood, both of which rebuild the loadings' regressors from
    // it, could be computed at all.
    if (draws.has_beta())
    {
        write_draws(file, "/posterior/beta/coeffs", draws.beta);

        // Only where the file put a prior on it. Held fixed, it is a
        // hyperparameter the file already carries at /priors/beta/rho, and
        // writing a row of the same number back would read as a posterior.
        if (draws.has_rho())
        {
            write_draws(file, "/posterior/beta/rho", draws.rho);
        }
    }

    if (draws.has_psi())
    {
        write_draws(file, "/posterior/psi/coeffs", draws.psi);
        write_draws(file, "/posterior/psi/sigma", draws.psi_sigma);
        if (draws.psi_lambda.n_elem > 0)
        {
            write_draws(file, "/posterior/psi/lambda", draws.psi_lambda);
        }
        write_noncentred(file, "/posterior/psi", draws.psi_noncentred);
    }

    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    write_draws(file, "/posterior/u_sigma_inv/sigma", draws.h_sigma);
    write_noncentred(file, "/posterior/u_sigma_inv", draws.h_noncentred);
}

} // namespace bayests::hdf5_io::vec_tvp_stochvol
