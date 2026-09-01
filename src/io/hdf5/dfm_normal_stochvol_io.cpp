// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/dfm_normal_stochvol_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::dfm_normal_stochvol
{

namespace
{

/// One of the two stochastic volatility groups, which differ in nothing but
/// their name and their width.
///
/// `sigma` is read into the *initial* values rather than into the prior: the
/// variance of the log-volatility innovations is a state the sampler redraws
/// every iteration, and it sits under /priors only because that is where
/// VarNormalStochvol's files put it. Read here so both groups cannot disagree
/// about it.
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

/// The parts of the posterior that do not depend on how much of either
/// volatility path the caller wants.
void read_common_draws(const ModelFile &file, DfmNormalStochvolDraws &draws)
{
    if (dataset_has_data(file, "/posterior/lambda/coeffs"))
    {
        draws.lambda = read_draws(file, "/posterior/lambda/coeffs");
    }
    if (dataset_has_data(file, "/posterior/factors/coeffs"))
    {
        draws.factors = read_draws(file, "/posterior/factors/coeffs");
    }
    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
}

} // namespace

DfmNormalStochvolInput read_input(const ModelFile &file)
{
    DfmNormalStochvolInput input;

    // No covariance block in this model, so the error specification says nothing
    // it needs to know -- the volatility is not optional here, it is the model.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);

    // Only draw_coefficients needs the priors and the starting values. A file
    // that holds nothing but a fitted posterior can still be forecast from, so a
    // missing one is left for validate() to complain about if it turns out to
    // matter.
    if (file.exist("/priors/lambda/mu"))
    {
        input.lambda_prior = read_normal_prior(file, "/priors/lambda");
    }
    read_vec_if_present(file, "/initial/lambda", input.initial.lambda);

    if (input.use_a())
    {
        if (file.exist("/priors/a/mu"))
        {
            input.a_prior = read_normal_prior(file, "/priors/a");
        }
        read_vec_if_present(file, "/initial/a", input.initial.a);
    }

    read_stochvol_group(file, "/priors/u_sigma", input.u_sigma_prior, input.initial.u_h_sigma);
    read_stochvol_group(file, "/priors/v_sigma", input.v_sigma_prior, input.initial.v_h_sigma);

    read_mat_if_present(file, "/initial/u_h", input.initial.u_h);
    read_vec_if_present(file, "/initial/u_h_init", input.initial.u_h_init);
    read_mat_if_present(file, "/initial/v_h", input.initial.v_h);
    read_vec_if_present(file, "/initial/v_h_init", input.initial.v_h_init);

    return input;
}

DfmNormalStochvolDraws read_coefficients(const ModelFile &file)
{
    DfmNormalStochvolDraws draws;
    read_common_draws(file, draws);

    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }
    if (dataset_has_data(file, "/posterior/v_sigma_inv/coeffs"))
    {
        draws.v_sigma_inv = read_draws(file, "/posterior/v_sigma_inv/coeffs");
    }

    return draws;
}

DfmNormalStochvolDraws read_forecast_coefficients(const ModelFile &file,
                                                  const DfmNormalStochvolInput &input)
{
    DfmNormalStochvolDraws draws;
    read_common_draws(file, draws);

    const arma::uword tt = input.train.periods(input.spec.k);

    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws_at_period(file, "/posterior/u_sigma_inv/coeffs", tt - 1,
                                                 static_cast<arma::uword>(input.spec.k));
    }
    if (dataset_has_data(file, "/posterior/v_sigma_inv/coeffs"))
    {
        draws.v_sigma_inv = read_draws_at_period(file, "/posterior/v_sigma_inv/coeffs", tt - 1,
                                                 static_cast<arma::uword>(input.spec.n_factors));
    }

    return draws;
}

void write_coefficients(const ModelFile &file, const DfmNormalStochvolDraws &draws)
{
    ensure_group(file, "/posterior");

    write_draws(file, "/posterior/lambda/coeffs", draws.lambda);

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
    }

    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    write_draws(file, "/posterior/v_sigma_inv/coeffs", draws.v_sigma_inv);
}

} // namespace bayests::hdf5_io::dfm_normal_stochvol
