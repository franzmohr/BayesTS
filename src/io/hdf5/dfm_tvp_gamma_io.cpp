// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/dfm_tvp_gamma_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::dfm_tvp_gamma
{

namespace
{

/// Periods in the sample, or zero for a file that carries no data to count them
/// off. Every path below is cut into periods, so this is the one number they all
/// depend on.
arma::uword sample_periods(const DfmTvpGammaInput &input)
{
    return input.train.y.n_elem > 0 && input.spec.k > 0 ? input.train.periods(input.spec.k) : 0;
}

} // namespace

DfmTvpGammaInput read_input(const ModelFile &file)
{
    DfmTvpGammaInput input;

    // No covariance block in this model, so the error specification says
    // nothing it needs to know.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);

    const arma::uword tt = sample_periods(input);
    const arma::uword n_lambda = static_cast<arma::uword>(input.spec.n_lambda());

    // Only draw_coefficients needs the priors and the starting values. A file
    // that holds nothing but a fitted posterior can still be forecast from, so a
    // missing one is left for validate() to complain about if it turns out to
    // matter.
    if (n_lambda > 0)
    {
        if (file.exist("/initial/lambda"))
        {
            input.initial.lambda = read_path(file, "/initial/lambda", n_lambda, tt);
            input.initial.lambda_sigma_inv = read_mat(file, "/initial/lambda_sigma_inv");
            input.initial.lambda_init = read_vec(file, "/initial/lambda_init");
        }
        if (file.exist("/priors/lambda/shape"))
        {
            input.lambda_prior.sigma = read_gamma_prior(file, "/priors/lambda");
            input.lambda_prior.initial_state = read_normal_prior(file, "/priors/lambda");
        }
    }

    if (input.use_a())
    {
        const arma::uword n_a = static_cast<arma::uword>(input.spec.n_factor_a());

        if (file.exist("/initial/a"))
        {
            input.initial.a = read_path(file, "/initial/a", n_a, tt);
            input.initial.a_sigma_inv = read_mat(file, "/initial/a_sigma_inv");
            input.initial.a_init = read_vec(file, "/initial/a_init");
        }
        if (file.exist("/priors/a/shape"))
        {
            input.a_prior.sigma = read_gamma_prior(file, "/priors/a");
            input.a_prior.initial_state = read_normal_prior(file, "/priors/a");
        }
    }

    if (file.exist("/priors/u_sigma/shape"))
    {
        input.u_sigma_prior = read_gamma_prior(file, "/priors/u_sigma");
    }
    if (file.exist("/priors/v_sigma/shape"))
    {
        input.v_sigma_prior = read_gamma_prior(file, "/priors/v_sigma");
    }

    // Both precisions are diagonal and are stored as the diagonal, which is
    // where the dynamic factor models part company with every other model here:
    // the constant variance VARs write /initial/u_sigma_inv as a k x k matrix.
    read_vec_if_present(file, "/initial/u_sigma_inv", input.initial.u_sigma_inv);
    read_vec_if_present(file, "/initial/v_sigma_inv", input.initial.v_sigma_inv);

    return input;
}

namespace
{

/// What both readers below take unchanged: the factor path and the two
/// precisions, none of which is cut to a period.
void read_common(const ModelFile &file, DfmTvpGammaDraws &draws)
{
    if (dataset_has_data(file, "/posterior/factors/coeffs"))
    {
        draws.factors = read_draws(file, "/posterior/factors/coeffs");
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }
    if (dataset_has_data(file, "/posterior/v_sigma_inv/coeffs"))
    {
        draws.v_sigma_inv = read_draws(file, "/posterior/v_sigma_inv/coeffs");
    }
}

} // namespace

DfmTvpGammaDraws read_loglik_coefficients(const ModelFile &file)
{
    DfmTvpGammaDraws draws;

    if (dataset_has_data(file, "/posterior/lambda/coeffs"))
    {
        draws.lambda = read_draws(file, "/posterior/lambda/coeffs");
    }
    read_common(file, draws);

    return draws;
}

DfmTvpGammaDraws read_forecast_coefficients(const ModelFile &file, const DfmTvpGammaInput &input)
{
    DfmTvpGammaDraws draws;

    const arma::uword tt = sample_periods(input);

    if (tt > 0 && dataset_has_data(file, "/posterior/lambda/coeffs"))
    {
        const arma::uword width =
            static_cast<arma::uword>(input.spec.k) * input.spec.n_factors;
        draws.lambda = read_draws_at_period(file, "/posterior/lambda/coeffs", tt - 1, width);
    }

    if (tt > 0 && input.use_a() && dataset_has_data(file, "/posterior/a/coeffs"))
    {
        const arma::uword n_a = static_cast<arma::uword>(input.spec.n_factor_a());
        draws.a = read_draws_at_period(file, "/posterior/a/coeffs", tt - 1, n_a);
    }

    read_common(file, draws);

    return draws;
}

void write_coefficients(const ModelFile &file, const DfmTvpGammaDraws &draws)
{
    ensure_group(file, "/posterior");

    write_draws(file, "/posterior/lambda/coeffs", draws.lambda);
    if (draws.lambda_sigma.n_elem > 0)
    {
        write_draws(file, "/posterior/lambda/sigma", draws.lambda_sigma);
    }

    // The factor path, which is part of the posterior rather than a by-product:
    // the factors are unobserved, so nothing downstream -- not the forecast, not
    // the log likelihood -- can be recomputed from the parameters alone.
    if (draws.has_factors())
    {
        write_draws(file, "/posterior/factors/coeffs", draws.factors);
    }

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
        write_draws(file, "/posterior/a/sigma", draws.a_sigma);
    }

    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    write_draws(file, "/posterior/v_sigma_inv/coeffs", draws.v_sigma_inv);
}

} // namespace bayests::hdf5_io::dfm_tvp_gamma
