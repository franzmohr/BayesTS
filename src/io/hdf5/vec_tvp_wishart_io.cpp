// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/vec_tvp_wishart_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::vec_tvp_wishart
{

VecTvpWishartInput read_input(const ModelFile &file)
{
    VecTvpWishartInput input;

    // No covariance block in this model, so the error specification says
    // nothing it needs to know. "wishart" is the spelling these files carry, and
    // passing it here would set spec.covar on a model that has no psi block --
    // inert today, since neither use_psi() nor n_psi() is reached from here, but
    // it is not what the flag means.
    input.spec = read_spec(file, nullptr);

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

    // Only draw_coefficients needs these. A file that holds nothing but a
    // fitted posterior can still be forecast from, so a missing prior is left
    // for validate() to complain about if it turns out to matter.
    if (file.exist("/priors/u_sigma/df"))
    {
        input.u_sigma_prior.df = get_dataset_int(file, "/priors/u_sigma/df");
    }
    read_mat_if_present(file, "/priors/u_sigma/scale", input.u_sigma_prior.scale);
    read_mat_if_present(file, "/initial/u_sigma_inv", input.initial.u_sigma_inv);

    return input;
}

// `input` is unused here and kept for the uniform signature the front-ends call
// every model's readers through. Unlike VecTvpGamma, whose precision may be
// stored per period, this model's u_sigma_inv is one k x k matrix per draw, so
// the dataset is read as it stands.
VecTvpWishartDraws read_loglik_coefficients(const ModelFile &file,
                                            [[maybe_unused]] const VecTvpWishartInput &input)
{
    VecTvpWishartDraws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        draws.beta = read_draws(file, "/posterior/beta/coeffs");
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

VecTvpWishartDraws read_forecast_coefficients(const ModelFile &file,
                                              const VecTvpWishartInput &input)
{
    VecTvpWishartDraws draws;

    const arma::uword tt = input.train.periods(input.spec.k);
    const arma::uword n_a = static_cast<arma::uword>(input.spec.nparams_per_period_vec());
    const arma::uword n_beta = static_cast<arma::uword>(input.spec.n_beta());

    if (n_a > 0 && dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws_at_period(file, "/posterior/a/coeffs", tt - 1, n_a);
    }
    if (n_beta > 0 && dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        draws.beta = read_draws_at_period(file, "/posterior/beta/coeffs", tt - 1, n_beta);
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

void write_coefficients(const ModelFile &file, const VecTvpWishartDraws &draws)
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

    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::vec_tvp_wishart
