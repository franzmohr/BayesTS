// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/var_tvp_ald_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::var_tvp_ald
{

VarTvpAldInput read_input(const ModelFile &file)
{
    VarTvpAldInput input;

    // nullptr rather than a "+covar" spelling: this model has no covariance
    // block, so the error attribute is never consulted and spec.covar stays
    // false. A file that asks for one is caught by validate().
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/z", input.train.z);

    const arma::uword tt = input.train.periods(input.spec.k);

    if (input.use_a())
    {
        const arma::uword nparams = input.train.nparams();

        input.initial.a = read_path(file, "/initial/a", nparams, tt);
        input.initial.a_sigma_inv = read_mat(file, "/initial/a_sigma_inv");
        input.initial.a_init = read_vec(file, "/initial/a_init");

        // The coefficient group carries both halves of the state equation: an
        // inverse gamma on the variance of the innovations, and a normal on the
        // state of the period before the sample.
        input.a_prior.sigma = read_gamma_prior(file, "/priors/a");
        input.a_prior.initial_state = read_normal_prior(file, "/priors/a");

        if (input.spec.varsel == VarSelection::bvs)
        {
            input.initial.a_lambda = read_vec(file, "/initial/a_lambda");
            input.a_varsel_prior = read_varsel_prior(file, "/priors/a", input.spec.varsel);
        }
    }

    input.u_scale_prior = read_gamma_prior(file, "/priors/u_scale");

    read_mat_if_present(file, "/initial/w", input.initial.w);
    read_vec_if_present(file, "/initial/u_scale", input.initial.u_scale);

    return input;
}

VarTvpAldDraws read_coefficients(const ModelFile &file)
{
    VarTvpAldDraws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/a/sigma"))
    {
        draws.a_sigma = read_draws(file, "/posterior/a/sigma");
    }
    if (dataset_has_data(file, "/posterior/a/lambda"))
    {
        draws.a_lambda = read_draws(file, "/posterior/a/lambda");
    }
    if (dataset_has_data(file, "/posterior/u_scale/coeffs"))
    {
        draws.u_scale = read_draws(file, "/posterior/u_scale/coeffs");
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

void write_coefficients(const ModelFile &file, const VarTvpAldDraws &draws)
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
    write_draws(file, "/posterior/u_scale/coeffs", draws.u_scale);
    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::var_tvp_ald
