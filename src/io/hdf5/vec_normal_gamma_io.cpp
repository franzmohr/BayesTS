// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/vec_normal_gamma_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::vec_normal_gamma
{

VecNormalGammaInput read_input(const ModelFile &file)
{
    VecNormalGammaInput input;

    input.spec = read_spec(file, "gamma+covar");

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/w", input.train.w);
    read_mat_if_present(file, "/data/train/z", input.train.z);
    read_mat_if_present(file, "/data/forecast/z", input.forecast.z);

    if (input.use_a())
    {
        input.a_prior = read_normal_prior(file, "/priors/a");
        input.initial.a = read_vec(file, "/initial/a");

        if (input.spec.uses_varsel())
        {
            input.initial.a_lambda = read_vec(file, "/initial/a_lambda");
            input.varsel_prior = read_varsel_prior(file, "/priors/a", input.spec.varsel);
        }
    }

    if (input.use_beta())
    {
        input.beta_prior = read_coint_space_prior_constant(file, "/priors/beta");
        input.initial.beta = read_vec(file, "/initial/beta");
    }

    if (input.use_psi())
    {
        input.psi_prior = read_normal_prior(file, "/priors/psi");
        input.initial.psi = read_vec(file, "/initial/psi");

        if (input.spec.uses_varsel())
        {
            input.initial.psi_lambda = read_vec(file, "/initial/psi_lambda");
            input.psi_varsel_prior = read_varsel_prior(file, "/priors/psi", input.spec.varsel);
        }
    }

    // Only draw_coefficients needs these. A file that holds nothing but a
    // fitted posterior can still be forecast from, so a missing prior is left
    // for validate() to complain about if it turns out to matter.
    input.u_sigma_prior = read_gamma_prior(file, "/priors/u_sigma");
    read_mat_if_present(file, "/initial/u_sigma_inv", input.initial.u_sigma_inv);

    return input;
}

VecNormalGammaDraws read_coefficients(const ModelFile &file)
{
    VecNormalGammaDraws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/a/lambda"))
    {
        draws.a_lambda = read_draws(file, "/posterior/a/lambda");
    }
    if (dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        draws.beta = read_draws(file, "/posterior/beta/coeffs");
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
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

void write_coefficients(const ModelFile &file, const VecNormalGammaDraws &draws)
{
    ensure_group(file, "/posterior");

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
        if (draws.has_lambda())
        {
            write_draws(file, "/posterior/a/lambda", draws.a_lambda);
        }
    }

    // The cointegration matrix. Without it `a` carries only the loadings, so
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
        if (draws.psi_lambda.n_elem > 0)
        {
            write_draws(file, "/posterior/psi/lambda", draws.psi_lambda);
        }
    }

    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::vec_normal_gamma
