// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/dfm_tvp_stochvol_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::dfm_tvp_stochvol
{

namespace
{

/// Periods in the sample, or zero for a file that carries no data to count them
/// off. Every path below is cut into periods, so this is the one number they all
/// depend on.
arma::uword sample_periods(const DfmTvpStochvolInput &input)
{
    return input.train.y.n_elem > 0 && input.spec.k > 0 ? input.train.periods(input.spec.k) : 0;
}

/// One of the two stochastic volatility prior groups. They differ in their name
/// and their width and in nothing else, so reading both through one function is
/// what keeps the two from drifting apart. `sigma` among them is the starting
/// value of the log-volatility innovation variance rather than a prior; it sits
/// in the prior group because that is where the files put it.
void read_stochvol_group(const ModelFile &file, const std::string &group, StochvolPrior &prior,
                         arma::vec &initial_h_sigma)
{
    read_vec_if_present(file, group + "/offset", prior.offset);
    read_vec_if_present(file, group + "/shape", prior.state.sigma.shape);
    read_vec_if_present(file, group + "/rate", prior.state.sigma.rate);
    read_vec_if_present(file, group + "/mu", prior.state.initial_state.mu);
    read_mat_if_present(file, group + "/v_inv", prior.state.initial_state.v_inv);
    read_vec_if_present(file, group + "/sigma", initial_h_sigma);
}

/// One of the two coefficient blocks: a state equation and a path to start it
/// from. `width` is n_lambda for the loadings and n_factor_a for the transition.
void read_random_walk_block(const ModelFile &file, const std::string &prior_group,
                            const std::string &name, arma::uword width, arma::uword tt,
                            RandomWalkPrior &prior, arma::mat &path, arma::mat &sigma_inv,
                            arma::vec &init)
{
    if (file.exist("/initial/" + name))
    {
        path = read_path(file, "/initial/" + name, width, tt);
        sigma_inv = read_mat(file, "/initial/" + name + "_sigma_inv");
        init = read_vec(file, "/initial/" + name + "_init");
    }
    if (file.exist(prior_group + "/shape"))
    {
        prior.sigma = read_gamma_prior(file, prior_group);
        prior.initial_state = read_normal_prior(file, prior_group);
    }
}

} // namespace

DfmTvpStochvolInput read_input(const ModelFile &file)
{
    DfmTvpStochvolInput input;

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
        read_random_walk_block(file, "/priors/lambda", "lambda", n_lambda, tt,
                               input.lambda_prior, input.initial.lambda,
                               input.initial.lambda_sigma_inv, input.initial.lambda_init);
    }

    if (input.use_a())
    {
        read_random_walk_block(file, "/priors/a", "a",
                               static_cast<arma::uword>(input.spec.n_factor_a()), tt,
                               input.a_prior, input.initial.a, input.initial.a_sigma_inv,
                               input.initial.a_init);
    }

    read_stochvol_group(file, "/priors/u_sigma", input.u_sigma_prior, input.initial.u_h_sigma);
    read_stochvol_group(file, "/priors/v_sigma", input.v_sigma_prior, input.initial.v_h_sigma);

    read_mat_if_present(file, "/initial/u_h", input.initial.u_h);
    read_vec_if_present(file, "/initial/u_h_init", input.initial.u_h_init);
    read_mat_if_present(file, "/initial/v_h", input.initial.v_h);
    read_vec_if_present(file, "/initial/v_h_init", input.initial.v_h_init);

    return input;
}

namespace
{

/// What both readers below take unchanged: the factor path and the two
/// precision paths. Neither precision is cut here -- the sampler takes the last
/// block of whatever it is handed, so a forecast needs no slice of them.
void read_common(const ModelFile &file, DfmTvpStochvolDraws &draws)
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

DfmTvpStochvolDraws read_loglik_coefficients(const ModelFile &file)
{
    DfmTvpStochvolDraws draws;

    if (dataset_has_data(file, "/posterior/lambda/coeffs"))
    {
        draws.lambda = read_draws(file, "/posterior/lambda/coeffs");
    }
    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    read_common(file, draws);

    return draws;
}

DfmTvpStochvolDraws read_forecast_coefficients(const ModelFile &file,
                                               const DfmTvpStochvolInput &input)
{
    DfmTvpStochvolDraws draws;

    const arma::uword tt = sample_periods(input);

    if (tt > 0 && dataset_has_data(file, "/posterior/lambda/coeffs"))
    {
        const arma::uword width = static_cast<arma::uword>(input.spec.k) * input.spec.n_factors;
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

void write_coefficients(const ModelFile &file, const DfmTvpStochvolDraws &draws)
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

} // namespace bayests::hdf5_io::dfm_tvp_stochvol
