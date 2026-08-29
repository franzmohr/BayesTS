// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/var_tvp_gamma_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::var_tvp_gamma
{

VarTvpGammaInput read_input(const ModelFile &file)
{
    VarTvpGammaInput input;

    input.spec = read_spec(file, "gamma+covar");

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/z", input.train.z);
    read_mat_if_present(file, "/data/forecast/z", input.forecast.z);

    const arma::uword tt = input.train.y.n_elem > 0 && input.spec.k > 0
                               ? input.train.periods(input.spec.k)
                               : 0;

    if (input.use_a())
    {
        const arma::uword nparams = input.train.nparams();

        input.initial.a = read_path(file, "/initial/a", nparams, tt);
        input.initial.a_sigma_inv = read_mat(file, "/initial/a_sigma_inv");
        input.initial.a_init = read_vec(file, "/initial/a_init");

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

VarTvpGammaDraws read_loglik_coefficients(const ModelFile &file,
                                          const VarTvpGammaInput &input)
{
    VarTvpGammaDraws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    draws.u_sigma_inv = read_precision(file, input.spec, input.train.periods(input.spec.k), input.use_psi());

    return draws;
}

VarTvpGammaDraws read_forecast_coefficients(const ModelFile &file,
                                            const VarTvpGammaInput &input)
{
    VarTvpGammaDraws draws;

    const arma::uword nparams = static_cast<arma::uword>(input.spec.nparams_per_period());

    if (nparams > 0 && dataset_has_data(file, "/posterior/a/coeffs"))
    {
        const arma::uword tt = input.train.periods(input.spec.k);
        draws.a = read_draws_at_period(file, "/posterior/a/coeffs", tt - 1, nparams);
    }
    draws.u_sigma_inv = read_precision(file, input.spec, input.train.periods(input.spec.k), input.use_psi());

    return draws;
}

void write_coefficients(const ModelFile &file, const VarTvpGammaDraws &draws)
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

} // namespace bayests::hdf5_io::var_tvp_gamma
