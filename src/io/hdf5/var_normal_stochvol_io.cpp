// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/var_normal_stochvol_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::var_normal_stochvol
{

namespace
{

/// The parts of the posterior that do not depend on how much of the volatility
/// path the caller wants.
void read_common_draws(const HighFive::File &file, VarNormalStochvolDraws &draws)
{
    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/a/lambda"))
    {
        draws.a_lambda = read_draws(file, "/posterior/a/lambda");
    }
    if (dataset_has_data(file, "/posterior/psi/coeffs"))
    {
        draws.psi = read_draws(file, "/posterior/psi/coeffs");
    }
    if (dataset_has_data(file, "/posterior/psi/lambda"))
    {
        draws.psi_lambda = read_draws(file, "/posterior/psi/lambda");
    }
    if (dataset_has_data(file, "/posterior/u_omega_inv/coeffs"))
    {
        draws.u_omega_inv = read_draws(file, "/posterior/u_omega_inv/coeffs");
    }
}

} // namespace

VarNormalStochvolInput read_input(const HighFive::File &file)
{
    VarNormalStochvolInput input;

    input.spec = read_spec(file, "sv+covar");

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/z", input.train.z);
    read_mat_if_present(file, "/data/forecast/z", input.forecast.z);

    if (input.use_a())
    {
        input.a_prior = read_normal_prior(file, "/priors/a");
        input.initial.a = read_vec(file, "/initial/a");

        // BVS is the only scheme this model implements. An SSVS file is left
        // unread here and rejected by validate(), which can say why -- reading
        // it would fail first, on mixture components the file does not carry.
        if (input.spec.varsel == VarSelection::bvs)
        {
            input.initial.a_lambda = read_vec(file, "/initial/a_lambda");
            input.a_varsel_prior = read_varsel_prior(file, "/priors/a", input.spec.varsel);
        }
    }

    if (input.use_psi())
    {
        input.psi_prior = read_normal_prior(file, "/priors/psi");
        input.initial.psi = read_vec(file, "/initial/psi");

        // BVS is the only scheme this model implements. An SSVS file is left
        // unread here and rejected by validate(), which can say why -- reading
        // it would fail first, on mixture components the file does not carry.
        if (input.spec.varsel == VarSelection::bvs)
        {
            input.initial.psi_lambda = read_vec(file, "/initial/psi_lambda");
            input.psi_varsel_prior = read_varsel_prior(file, "/priors/psi", input.spec.varsel);
        }
    }

    // Only draw_coefficients needs the rest. A file that holds nothing but a
    // fitted posterior can still be forecast from, so a missing prior is left
    // for validate() to complain about if it turns out to matter.
    read_vec_if_present(file, "/priors/u_sigma/offset", input.u_sigma_prior.offset);
    read_vec_if_present(file, "/priors/u_sigma/shape", input.u_sigma_prior.state.sigma.shape);
    read_vec_if_present(file, "/priors/u_sigma/rate", input.u_sigma_prior.state.sigma.rate);
    read_vec_if_present(file, "/priors/u_sigma/mu", input.u_sigma_prior.state.initial_state.mu);
    read_mat_if_present(file, "/priors/u_sigma/v_inv", input.u_sigma_prior.state.initial_state.v_inv);

    read_vec_if_present(file, "/priors/u_sigma/sigma", input.initial.h_sigma);
    read_mat_if_present(file, "/initial/h", input.initial.h);
    read_vec_if_present(file, "/initial/h_init", input.initial.h_init);

    return input;
}

VarNormalStochvolDraws read_coefficients(const HighFive::File &file)
{
    VarNormalStochvolDraws draws;
    read_common_draws(file, draws);

    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

VarNormalStochvolDraws read_forecast_coefficients(const HighFive::File &file,
                                                  const VarNormalStochvolInput &input)
{
    VarNormalStochvolDraws draws;
    read_common_draws(file, draws);

    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        const arma::uword k = static_cast<arma::uword>(input.spec.k);
        const arma::uword tt = input.train.periods(input.spec.k);
        draws.u_sigma_inv = read_draws_at_period(file, "/posterior/u_sigma_inv/coeffs",
                                                 tt - 1, k * k);
    }

    return draws;
}

void write_coefficients(HighFive::File &file, const VarNormalStochvolDraws &draws)
{
    ensure_group(file, "/posterior");

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
        if (draws.a_lambda.n_elem > 0)
        {
            write_draws(file, "/posterior/a/lambda", draws.a_lambda);
        }
    }
    if (draws.has_psi())
    {
        write_draws(file, "/posterior/psi/coeffs", draws.psi);
        if (draws.psi_lambda.n_elem > 0)
        {
            write_draws(file, "/posterior/psi/lambda", draws.psi_lambda);
        }
    }
    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::var_normal_stochvol
