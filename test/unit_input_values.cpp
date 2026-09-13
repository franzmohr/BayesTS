// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_input_values.cpp
/// @brief Checks that validate() refuses prior values no model can mean, and
///        that a VAR forecast refuses regressors without a row per horizon.
///
/// Every refusal here used to run. A negative gamma rate or an inclusion
/// probability of 1.5 went through `bayests check` and `bayests coefficients`
/// with exit code 0: the first produced plausible numbers from a prior that is
/// not one, the second kept every coefficient at zero for the whole chain,
/// because log(1 - 1.5) is a NaN and a NaN excludes. A forecast handed fewer
/// regressor rows than horizons read and wrote past the end of the matrix.
///
/// Each check starts from an input validate() accepts and changes one thing, so a
/// refusal is attributable to that one thing rather than to the builder.

#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_stochvol.h"

#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

const int kPeriods = 30;

/// Two equations and an intercept each, errors drawn independently.
bayests::VarNormalGammaInput gamma_input()
{
    bayests::VarNormalGammaInput input;
    input.spec.k = 2;
    input.spec.n = 1;
    input.spec.iterations = 20;
    input.spec.burnin = 10;

    input.train.y = arma::randn<arma::mat>(kPeriods, 2);
    input.train.z = arma::repmat(arma::eye<arma::mat>(2, 2), kPeriods, 1);

    input.a_prior.mu = arma::zeros<arma::vec>(2);
    input.a_prior.v_inv = arma::eye<arma::mat>(2, 2);
    input.u_sigma_prior.shape = arma::vec(2, arma::fill::value(3.0));
    input.u_sigma_prior.rate = arma::vec(2, arma::fill::value(2.0));

    input.initial.a = arma::zeros<arma::vec>(2);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(2, 2);
    return input;
}

/// The same model with SSVS on both coefficients.
bayests::VarNormalGammaInput ssvs_input()
{
    bayests::VarNormalGammaInput input = gamma_input();
    input.spec.varsel = bayests::VarSelection::ssvs;
    input.a_varsel_prior.inprior = arma::vec(2, arma::fill::value(0.5));
    input.a_varsel_prior.include = arma::uvec{0, 1};
    input.a_varsel_prior.ssvs.tau0 = arma::vec(2, arma::fill::value(0.1));
    input.a_varsel_prior.ssvs.tau1 = arma::vec(2, arma::fill::value(10.0));
    input.initial.a_lambda = arma::ones<arma::vec>(2);
    return input;
}

bayests::VarNormalStochvolInput stochvol_input()
{
    bayests::VarNormalStochvolInput input;
    input.spec.k = 2;
    input.spec.n = 1;
    input.spec.iterations = 20;
    input.spec.burnin = 10;

    input.train.y = arma::randn<arma::mat>(kPeriods, 2);
    input.train.z = arma::repmat(arma::eye<arma::mat>(2, 2), kPeriods, 1);

    input.a_prior.mu = arma::zeros<arma::vec>(2);
    input.a_prior.v_inv = arma::eye<arma::mat>(2, 2);
    input.u_sigma_prior.offset = arma::vec(2, arma::fill::value(1e-4));
    input.u_sigma_prior.state.sigma.shape = arma::vec(2, arma::fill::value(3.0));
    input.u_sigma_prior.state.sigma.rate = arma::vec(2, arma::fill::value(0.2));
    input.u_sigma_prior.state.initial_state.mu = arma::zeros<arma::vec>(2);
    input.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(2, 2);

    input.initial.a = arma::zeros<arma::vec>(2);
    input.initial.h = arma::zeros<arma::mat>(kPeriods, 2);
    input.initial.h_init = arma::zeros<arma::vec>(2);
    input.initial.h_sigma = arma::vec(2, arma::fill::value(0.05));
    return input;
}

template <class Input>
bool refused(const Input &input)
{
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

template <class Input>
void expect_refused(const std::string &what, Input input, const std::function<void(Input &)> &change)
{
    change(input);
    check(what + " is refused", refused(input));
}

void out_of_range_priors_are_refused()
{
    std::printf("prior values\n");

    const double nan = std::numeric_limits<double>::quiet_NaN();

    check("the gamma input as built is accepted", !refused(gamma_input()));
    check("the SSVS input as built is accepted", !refused(ssvs_input()));
    check("the stochastic volatility input as built is accepted", !refused(stochvol_input()));

    using Gamma = bayests::VarNormalGammaInput;
    expect_refused<Gamma>("a negative gamma rate", gamma_input(),
                          [](Gamma &in) { in.u_sigma_prior.rate(1) = -1.0; });
    expect_refused<Gamma>("a negative gamma shape", gamma_input(),
                          [](Gamma &in) { in.u_sigma_prior.shape(0) = -0.5; });
    expect_refused<Gamma>("a NaN gamma shape", gamma_input(),
                          [nan](Gamma &in) { in.u_sigma_prior.shape(0) = nan; });
    expect_refused<Gamma>("an asymmetric prior precision", gamma_input(),
                          [](Gamma &in) { in.a_prior.v_inv(0, 1) = 0.5; });
    expect_refused<Gamma>("a non-diagonal initial error precision", gamma_input(),
                          [](Gamma &in) { in.initial.u_sigma_inv(1, 0) = in.initial.u_sigma_inv(0, 1) = 0.3; });
    expect_refused<Gamma>("an inclusion probability of 1.5", ssvs_input(),
                          [](Gamma &in) { in.a_varsel_prior.inprior(0) = 1.5; });
    expect_refused<Gamma>("a negative inclusion probability", ssvs_input(),
                          [](Gamma &in) { in.a_varsel_prior.inprior(1) = -0.1; });
    expect_refused<Gamma>("a zero SSVS spike", ssvs_input(),
                          [](Gamma &in) { in.a_varsel_prior.ssvs.tau0(0) = 0.0; });

    // Zero is a boundary rather than a mistake: an improper gamma prior whose
    // posterior is proper once the sample is added.
    Gamma zero_rate = gamma_input();
    zero_rate.u_sigma_prior.rate.zeros();
    check("a zero gamma rate is accepted", !refused(zero_rate));

    using Stochvol = bayests::VarNormalStochvolInput;
    expect_refused<Stochvol>("a zero log-volatility offset", stochvol_input(),
                             [](Stochvol &in) { in.u_sigma_prior.offset(0) = 0.0; });
    expect_refused<Stochvol>("a negative log-volatility innovation rate", stochvol_input(),
                             [](Stochvol &in) { in.u_sigma_prior.state.sigma.rate(1) = -0.2; });
    expect_refused<Stochvol>("a zero initial log-volatility variance", stochvol_input(),
                             [](Stochvol &in) { in.initial.h_sigma(0) = 0.0; });
}

bool forecast_refused(const bayests::VarNormalGammaInput &input,
                      const bayests::VarNormalGammaDraws &draws)
{
    bayests::NullReporter reporter;
    try
    {
        bayests::VarNormalGammaSampler{}.forecast(input, draws, reporter);
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

void forecast_regressors_need_a_row_per_horizon()
{
    std::printf("forecast regressors\n");

    bayests::VarNormalGammaInput input = gamma_input();
    input.spec.h = 3;

    bayests::NullReporter reporter;
    const bayests::VarNormalGammaDraws draws =
        bayests::VarNormalGammaSampler{}.draw_coefficients(input, reporter);

    input.forecast.x = arma::ones<arma::mat>(3, 1);
    check("one row per horizon is accepted", !forecast_refused(input, draws));

    input.forecast.x = arma::ones<arma::mat>(2, 1);
    check("a row short is refused", forecast_refused(input, draws));

    input.forecast.x = arma::ones<arma::mat>(4, 1);
    check("a row over is refused", forecast_refused(input, draws));
}

} // namespace

int main()
{
    std::printf("unit_input_values\n");
    arma::arma_rng::set_seed(20260913);

    out_of_range_priors_are_refused();
    forecast_regressors_need_a_row_per_horizon();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
