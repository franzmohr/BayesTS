// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_forecast_states.cpp
/// @brief Checks that a time-varying VAR's forecast carries its random walks
///        over the horizon under ForecastStates::simulate and holds them under
///        ForecastStates::hold.
///
/// Every case starts all draws from the same end-of-sample state and makes the
/// quantity under test the only source of spread, so what the model implies at
/// horizon i -- counted from one -- has a closed form:
///
///   - coefficients: an intercept that is a random walk with innovation variance
///     s, under an error of negligible variance, gives Var(y_{T+i}) = i s;
///   - selection: a lag coefficient BVS excluded stays at zero, so the 100 its
///     regressor holds in the last observation adds nothing to that variance;
///   - covariance block: Psi's one free element a random walk from zero with
///     variance s, under Omega = I, gives Var(u_2) = 1 + i s, because
///     Psi u ~ N(0, Omega) makes u_2 = e_2 - psi u_1;
///   - log-volatility: h_T = 0 with innovation variance s and no regressors
///     gives E[y_{T+i}^2] = E[exp(h_{T+i})] = exp(i s / 2);
///   - a factor model's volatilities: the same, through the factor with an
///     idiosyncratic error of negligible variance, and through the idiosyncratic
///     error with a factor of negligible variance;
///   - a factor model's loadings: a free loading that is a random walk from zero
///     with variance s, on a unit factor, gives Var(x_2) = i s, while the fixed
///     loading of the first series stays at one;
///   - a factor model's transition: a coefficient that is a random walk from
///     zero with variance s, from a zero last factor, gives Var(f_2) = 1 + 2 s;
///   - a VEC's cointegration vector: one series with a unit loading, so the
///     level VAR is y_t = (1 + alpha beta_t) y_{t-1}, from a last level of 10 and
///     beta_{T+1} = rho beta_T + eta, gives mean 10 (1 + rho beta_T) and variance
///     100 at the first horizon;
///   - a VEC's loading: alpha a random walk from zero with variance s against
///     beta_T = 1 gives Var(y_{T+1}) = 100 s (1 + 1);
///   - a VEC's log-volatility: no cointegration and no lags, so y_t = y_{t-1} +
///     u_t from zero, gives E[y_{T+i}^2] = sum_{j <= i} exp(j s / 2).
///
/// And one case that is not statistical: log-volatilities that start far apart
/// and drift further, under a covariance block, must still forecast finite
/// numbers. Drawing the error through the inverse of Psi' diag(exp(-h)) Psi put
/// a NaN into every variable from the first horizon where that inverse had an
/// eigenvalue a rounding error below zero. A held forecast from that same
/// precision, in any of the six VARs, must be finite for the same reason.
///
/// These are statistical statements. Each is checked to 5 percent with enough
/// draws that the Monte Carlo standard error is about 1 percent or less, and
/// under `hold` every one of them lands far outside that band.

#include "bayests/dfm_normal_stochvol.h"
#include "bayests/dfm_tvp_gamma.h"
#include "bayests/dfm_tvp_stochvol.h"
#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_stochvol.h"
#include "bayests/var_normal_wishart.h"
#include "bayests/var_tvp_gamma.h"
#include "bayests/var_tvp_stochvol.h"
#include "bayests/var_tvp_wishart.h"
#include "bayests/vec_normal_stochvol.h"
#include "bayests/vec_tvp_stochvol.h"
#include "bayests/vec_tvp_wishart.h"
#include "core/models/forecast_states.h"

#include <cmath>
#include <cstdio>
#include <exception>
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

/// Within `tolerance` of `expected`, relatively; prints both either way.
void check_close(const std::string &what, const double value, const double expected,
                 const double tolerance = 0.05)
{
    std::printf("    %s: %.4f against %.4f\n", what.c_str(), value, expected);
    check(what, std::abs(value / expected - 1.0) < tolerance);
}

using bayests::ForecastStates;

void coefficients_drift()
{
    std::printf("an intercept that is a random walk\n");

    constexpr int h = 6;
    constexpr arma::uword draws = 40000;
    constexpr double s = 0.5;

    bayests::VarTvpWishartInput input;
    input.spec.k = 1;
    input.spec.n = 1;
    input.spec.h = h;
    input.forecast.x = arma::ones<arma::mat>(h, 1);

    bayests::VarTvpWishartDraws posterior;
    posterior.a = arma::zeros<arma::mat>(1, draws);
    posterior.a_sigma = arma::mat(1, draws, arma::fill::value(s));
    posterior.u_sigma_inv = arma::mat(1, draws, arma::fill::value(1e8));

    bayests::NullReporter reporter;
    const arma::mat simulated =
        bayests::VarTvpWishartSampler{}.forecast(input, posterior, reporter).values;
    for (int i = 0; i < h; i++)
    {
        check_close("Var(y) at horizon " + std::to_string(i + 1) + " is i s",
                    arma::var(simulated.row(i)), (i + 1) * s);
    }

    input.spec.forecast_states = ForecastStates::hold;
    const arma::mat held =
        bayests::VarTvpWishartSampler{}.forecast(input, posterior, reporter).values;
    check("held, Var(y) at the last horizon is the error's alone",
          arma::var(held.row(h - 1)) < 1e-6);

    // The draws a simulated forecast reads on top of a held one.
    posterior.a_sigma.reset();
    input.spec.forecast_states = ForecastStates::simulate;
    bool refused = false;
    try
    {
        bayests::VarTvpWishartSampler{}.forecast(input, posterior, reporter);
    }
    catch (const std::invalid_argument &)
    {
        refused = true;
    }
    check("simulating without the innovation variances is refused", refused);
}

void excluded_coefficients_stay_excluded()
{
    std::printf("a lag coefficient BVS left out\n");

    constexpr arma::uword draws = 40000;

    bayests::VarTvpGammaInput input;
    input.spec.k = 1;
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.h = 1;

    // The lag block first, then the intercept, and a last observation of 100.
    input.forecast.x = {{100.0, 1.0}};

    bayests::VarTvpGammaDraws posterior;
    posterior.a = arma::zeros<arma::mat>(2, draws);
    posterior.a_sigma = arma::repmat(arma::vec{1.0, 0.5}, 1, draws);
    posterior.a_lambda = arma::repmat(arma::vec{0.0, 1.0}, 1, draws);
    posterior.u_omega_inv = arma::mat(1, draws, arma::fill::value(1e8));
    posterior.u_sigma_inv = posterior.u_omega_inv;

    bayests::NullReporter reporter;
    const arma::mat masked = bayests::VarTvpGammaSampler{}.forecast(input, posterior, reporter).values;
    check_close("Var(y) is the intercept's step alone", arma::var(masked.row(0)), 0.5);

    // What the same forecast spreads to when the mask is not applied, so the
    // check above could not pass by accident.
    posterior.a_lambda.reset();
    const arma::mat unmasked =
        bayests::VarTvpGammaSampler{}.forecast(input, posterior, reporter).values;
    check_close("without the mask, the lag's step times 100 squared joins it",
                arma::var(unmasked.row(0)), 10000.5);
}

void covariance_block_drifts()
{
    std::printf("a covariance block that is a random walk\n");

    constexpr int h = 6;
    constexpr arma::uword draws = 100000;
    constexpr double s = 0.5;

    bayests::VarTvpGammaInput input;
    input.spec.k = 2;
    input.spec.h = h;
    input.spec.covar = true;

    bayests::VarTvpGammaDraws posterior;
    posterior.psi = arma::repmat(arma::vectorise(arma::eye<arma::mat>(2, 2)), 1, draws);
    posterior.psi_sigma = arma::mat(1, draws, arma::fill::value(s));
    posterior.u_omega_inv = arma::ones<arma::mat>(2, draws);
    posterior.u_sigma_inv = posterior.psi;

    bayests::NullReporter reporter;
    const arma::mat simulated =
        bayests::VarTvpGammaSampler{}.forecast(input, posterior, reporter).values;
    for (int i = 0; i < h; i++)
    {
        check_close("Var(u_2) at horizon " + std::to_string(i + 1) + " is 1 + i s",
                    arma::var(simulated.row(2 * i + 1)), 1.0 + (i + 1) * s);
    }
    check_close("Var(u_1) at the last horizon stays 1", arma::var(simulated.row(2 * h - 2)), 1.0);

    input.spec.forecast_states = ForecastStates::hold;
    const arma::mat held = bayests::VarTvpGammaSampler{}.forecast(input, posterior, reporter).values;
    check_close("held, Var(u_2) at the last horizon stays 1", arma::var(held.row(2 * h - 1)), 1.0);
}

/// Both stochastic volatility VARs, which differ here only in the type.
template <typename Sampler, typename Input, typename Draws>
void volatility_drifts(const char *name)
{
    std::printf("%s: a log-volatility that is a random walk\n", name);

    constexpr int h = 4;
    constexpr arma::uword draws = 100000;
    constexpr double s = 0.3;

    Input input;
    input.spec.k = 1;
    input.spec.h = h;

    Draws posterior;
    posterior.u_omega_inv = arma::ones<arma::mat>(1, draws);
    posterior.u_sigma_inv = posterior.u_omega_inv;
    posterior.h_sigma = arma::mat(1, draws, arma::fill::value(s));

    bayests::NullReporter reporter;
    const arma::mat simulated = Sampler{}.forecast(input, posterior, reporter).values;
    for (int i = 0; i < h; i++)
    {
        check_close("E[y^2] at horizon " + std::to_string(i + 1) + " is exp(i s / 2)",
                    arma::mean(arma::square(simulated.row(i))), std::exp((i + 1) * s / 2));
    }

    input.spec.forecast_states = ForecastStates::hold;
    const arma::mat held = Sampler{}.forecast(input, posterior, reporter).values;
    check_close("held, E[y^2] at the last horizon stays 1",
                arma::mean(arma::square(held.row(h - 1))), 1.0);

    // A posterior drawn before h_sigma was stored: refused when it is needed,
    // and not when it is not.
    posterior.h_sigma.reset();
    bool held_runs = false;
    try
    {
        Sampler{}.forecast(input, posterior, reporter);
        held_runs = true;
    }
    catch (const std::exception &e)
    {
        std::printf("    threw: %s\n", e.what());
    }
    check("held, a posterior without h_sigma still forecasts", held_runs);

    input.spec.forecast_states = ForecastStates::simulate;
    bool refused = false;
    try
    {
        Sampler{}.forecast(input, posterior, reporter);
    }
    catch (const std::invalid_argument &)
    {
        refused = true;
    }
    check("simulated, a posterior without h_sigma is refused", refused);
}

/// Both stochastic volatility factor models, one series on one factor with no
/// transition, so x = f + u and the two volatilities can be switched on one at a
/// time by giving the other a negligible variance and no drift.
template <typename Sampler, typename Input, typename Draws>
void factor_volatilities_drift(const char *name)
{
    std::printf("%s: two log-volatilities that are random walks\n", name);

    constexpr int h = 4;
    constexpr arma::uword draws = 100000;
    constexpr double s = 0.3;

    Input input;
    input.spec.k = 1;
    input.spec.n_factors = 1;
    input.spec.h = h;
    input.train.y = arma::zeros<arma::mat>(2, 1);

    Draws posterior;
    posterior.lambda = arma::ones<arma::mat>(1, draws);
    posterior.factors = arma::zeros<arma::mat>(2, draws);

    const auto run = [&](const double u_precision, const double u_step,
                         const double v_precision, const double v_step) {
        posterior.u_sigma_inv = arma::mat(1, draws, arma::fill::value(u_precision));
        posterior.v_sigma_inv = arma::mat(1, draws, arma::fill::value(v_precision));
        posterior.u_h_sigma = arma::mat(1, draws, arma::fill::value(u_step));
        posterior.v_h_sigma = arma::mat(1, draws, arma::fill::value(v_step));
        bayests::NullReporter reporter;
        return Sampler{}.forecast(input, posterior, reporter).values;
    };

    const arma::mat factor_moves = run(1e8, 0.0, 1.0, s);
    const arma::mat series_moves = run(1.0, s, 1e8, 0.0);
    for (int i = 0; i < h; i++)
    {
        const double expected = std::exp((i + 1) * s / 2);
        check_close("through the factor, E[x^2] at horizon " + std::to_string(i + 1),
                    arma::mean(arma::square(factor_moves.row(i))), expected);
        check_close("through the series, E[x^2] at horizon " + std::to_string(i + 1),
                    arma::mean(arma::square(series_moves.row(i))), expected);
    }

    input.spec.forecast_states = ForecastStates::hold;
    check_close("held, E[x^2] at the last horizon stays 1",
                arma::mean(arma::square(run(1e8, 0.0, 1.0, s).row(h - 1))), 1.0);

    posterior.v_h_sigma.reset();
    input.spec.forecast_states = ForecastStates::simulate;
    bool refused = false;
    try
    {
        bayests::NullReporter reporter;
        Sampler{}.forecast(input, posterior, reporter);
    }
    catch (const std::invalid_argument &)
    {
        refused = true;
    }
    check("simulated, a posterior without v_h_sigma is refused", refused);
}

void factor_coefficients_drift()
{
    std::printf("DfmTvpGamma: loadings and a transition that are random walks\n");

    constexpr int h = 3;
    constexpr arma::uword draws = 40000;
    constexpr double s = 0.5;

    // Two series on one factor of order one. The first series' loading is the
    // identifying one; the second's is the model's single free loading.
    bayests::DfmTvpGammaInput input;
    input.spec.k = 2;
    input.spec.n_factors = 1;
    input.spec.p = 1;
    input.spec.h = h;
    input.train.y = arma::zeros<arma::mat>(2, 2);

    bayests::DfmTvpGammaDraws posterior;
    posterior.lambda = arma::repmat(arma::vec{1.0, 0.0}, 1, draws);
    posterior.factors = arma::zeros<arma::mat>(2, draws);
    posterior.a = arma::zeros<arma::mat>(1, draws);
    posterior.u_sigma_inv = arma::mat(2, draws, arma::fill::value(1e8));
    posterior.v_sigma_inv = arma::ones<arma::mat>(1, draws);

    bayests::NullReporter reporter;

    posterior.lambda_sigma = arma::mat(1, draws, arma::fill::value(s));
    posterior.a_sigma = arma::zeros<arma::mat>(1, draws);
    const arma::mat loadings_move =
        bayests::DfmTvpGammaSampler{}.forecast(input, posterior, reporter).values;
    for (int i = 0; i < h; i++)
    {
        check_close("Var(x_2) at horizon " + std::to_string(i + 1) + " is i s",
                    arma::var(loadings_move.row(2 * i + 1)), (i + 1) * s);
    }
    check_close("the identifying loading does not move: Var(x_1) stays 1",
                arma::var(loadings_move.row(2 * h - 2)), 1.0);

    posterior.lambda_sigma = arma::zeros<arma::mat>(1, draws);
    posterior.a_sigma = arma::mat(1, draws, arma::fill::value(s));
    const arma::mat transition_moves =
        bayests::DfmTvpGammaSampler{}.forecast(input, posterior, reporter).values;
    check_close("Var(x_1) at horizon 1 is the innovation's alone",
                arma::var(transition_moves.row(0)), 1.0);
    check_close("Var(x_1) at horizon 2 is 1 + 2 s", arma::var(transition_moves.row(2)),
                1.0 + 2 * s);

    input.spec.forecast_states = ForecastStates::hold;
    const arma::mat held = bayests::DfmTvpGammaSampler{}.forecast(input, posterior, reporter).values;
    check_close("held, Var(x_1) at horizon 2 stays 1", arma::var(held.row(2)), 1.0);
}

void vec_coefficients_drift()
{
    std::printf("VecTvpWishart: a cointegration vector and a loading that drift\n");

    constexpr arma::uword draws = 40000;

    // One series of rank one and no lagged differences: the level VAR is the
    // single block A_1 = 1 + alpha beta, and the last level is 10.
    bayests::VecTvpWishartInput input;
    input.spec.k = 1;
    input.spec.p = 1;
    input.spec.rank = 1;
    input.spec.k_beta = 1;
    input.spec.h = 1;
    input.forecast.x = arma::mat(1, 1, arma::fill::value(10.0));
    input.beta_prior.rho = 1.0;

    bayests::VecTvpWishartDraws posterior;
    posterior.u_sigma_inv = arma::mat(1, draws, arma::fill::value(1e8));

    bayests::NullReporter reporter;
    const auto run = [&]() {
        return bayests::VecTvpWishartSampler{}.forecast(input, posterior, reporter).values;
    };

    // The cointegration vector alone, from zero, under a unit loading.
    posterior.a = arma::ones<arma::mat>(1, draws);
    posterior.a_sigma = arma::zeros<arma::mat>(1, draws);
    posterior.beta = arma::zeros<arma::mat>(1, draws);
    const arma::mat beta_moves = run();
    check_close("beta: E[y] at horizon 1 is 10", arma::mean(beta_moves.row(0)), 10.0);
    check_close("beta: Var(y) at horizon 1 is 100", arma::var(beta_moves.row(0)), 100.0);

    // rho reaches the step: from beta_T = 2 at rho = 0.5 the mean is 10 (1 + 1).
    input.beta_prior.rho = 0.5;
    posterior.beta = arma::mat(1, draws, arma::fill::value(2.0));
    check_close("beta: rho 0.5 from 2 puts E[y] at 20", arma::mean(run().row(0)), 20.0);

    // A rho the chain drew overrides the prior's.
    posterior.rho = arma::mat(1, draws, arma::fill::value(1.0));
    check_close("beta: a drawn rho of 1 from 2 puts E[y] at 30", arma::mean(run().row(0)), 30.0);
    posterior.rho.reset();
    input.beta_prior.rho = 1.0;

    // The loading, from zero, against beta_T = 1 that also takes its unit step.
    posterior.a = arma::zeros<arma::mat>(1, draws);
    posterior.a_sigma = arma::mat(1, draws, arma::fill::value(0.5));
    posterior.beta = arma::ones<arma::mat>(1, draws);
    check_close("alpha: Var(y) at horizon 1 is 100 s (1 + 1)", arma::var(run().row(0)), 100.0);

    input.spec.forecast_states = ForecastStates::hold;
    check("held, Var(y) at horizon 1 is the error's alone", arma::var(run().row(0)) < 1e-3);
}

/// Both stochastic volatility VECs: one series, no cointegration and no lags,
/// so the level VAR is a random walk and the forecast accumulates the volatility.
template <typename Sampler, typename Input, typename Draws>
void vec_volatility_drifts(const char *name)
{
    std::printf("%s: a log-volatility that is a random walk\n", name);

    constexpr int h = 2;
    constexpr arma::uword draws = 100000;
    constexpr double s = 0.3;

    Input input;
    input.spec.k = 1;
    input.spec.h = h;
    input.forecast.x = arma::zeros<arma::mat>(h, 1);

    Draws posterior;
    posterior.u_omega_inv = arma::ones<arma::mat>(1, draws);
    posterior.u_sigma_inv = posterior.u_omega_inv;
    posterior.h_sigma = arma::mat(1, draws, arma::fill::value(s));

    bayests::NullReporter reporter;
    const arma::mat simulated = Sampler{}.forecast(input, posterior, reporter).values;
    double expected = 0.0;
    for (int i = 0; i < h; i++)
    {
        expected += std::exp((i + 1) * s / 2);
        check_close("E[y^2] at horizon " + std::to_string(i + 1),
                    arma::mean(arma::square(simulated.row(i))), expected);
    }

    input.spec.forecast_states = ForecastStates::hold;
    check_close("held, E[y^2] at the last horizon is h",
                arma::mean(arma::square(Sampler{}.forecast(input, posterior, reporter).values.row(h - 1))),
                static_cast<double>(h));

    posterior.h_sigma.reset();
    input.spec.forecast_states = ForecastStates::simulate;
    bool refused = false;
    try
    {
        Sampler{}.forecast(input, posterior, reporter);
    }
    catch (const std::invalid_argument &)
    {
        refused = true;
    }
    check("simulated, a posterior without h_sigma is refused", refused);
}

/// A unit lower triangular Psi with a free element in every position.
arma::mat far_drift_psi()
{
    return {{1.0, 0.0, 0.0, 0.0},
            {0.9, 1.0, 0.0, 0.0},
            {-1.3, 2.1, 1.0, 0.0},
            {0.4, -0.7, 1.6, 1.0}};
}

/// Log-volatilities 36 apart: a condition number around 1e16 before any drift.
arma::vec far_drift_h()
{
    return {-18.0, 18.0, -9.0, 9.0};
}

void covariance_root_far_apart()
{
    std::printf("the error's root under log-volatilities far apart\n");

    const arma::mat psi = far_drift_psi();
    const arma::vec variances = arma::exp(far_drift_h());
    const arma::mat root = bayests::core::covariance_root(psi, variances);

    // What the route through the precision gets from the same matrix, for the
    // record rather than as a check: whether it lands below zero depends on
    // the BLAS.
    arma::vec eigval;
    arma::mat eigvec;
    const arma::mat precision = arma::trans(psi) * arma::diagmat(1.0 / variances) * psi;
    arma::eig_sym(eigval, eigvec,
                  arma::symmatu(arma::solve(precision, arma::eye<arma::mat>(4, 4))));
    std::printf("    through the precision, the smallest eigenvalue is %.3e\n", eigval.min());

    const arma::mat factor =
        arma::solve(arma::trimatl(psi), arma::mat(arma::diagmat(arma::sqrt(variances))));
    const arma::mat covariance = factor * arma::trans(factor);
    check("the root is finite", root.is_finite());
    check("the root is symmetric", arma::norm(root - arma::trans(root), "fro") <=
                                       1e-12 * arma::norm(root, "fro"));
    check("the root squares to the covariance",
          arma::norm(root * root - covariance, "fro") <= 1e-10 * arma::norm(covariance, "fro"));

    // With the covariance block switched off the root is the standard deviations.
    const arma::mat diagonal =
        bayests::core::covariance_root(arma::eye<arma::mat>(4, 4), variances);
    check("without Psi, the root is diag(exp(h / 2))",
          arma::approx_equal(diagonal, arma::mat(arma::diagmat(arma::exp(far_drift_h() / 2))),
                             "reldiff", 1e-12));
}

/// Both stochastic volatility VARs and both VECs under a covariance block, from
/// log-volatilities far apart that then take steps of variance 4 for twelve
/// horizons: every forecast must be a number.
template <typename Sampler, typename Input, typename Draws>
void far_drift_stays_finite(const char *name, const bool vec)
{
    std::printf("%s: log-volatilities that drift far apart\n", name);

    constexpr int k = 4;
    constexpr int h = 12;
    constexpr arma::uword draws = 2000;

    Input input;
    input.spec.k = k;
    input.spec.h = h;
    input.spec.covar = true;
    if (vec)
    {
        // No cointegration and no lags: the level VAR is a random walk.
        input.forecast.x = arma::zeros<arma::mat>(h, k);
    }

    const arma::mat psi = far_drift_psi();
    Draws posterior;
    posterior.psi = arma::repmat(arma::vectorise(psi), 1, draws);
    posterior.u_omega_inv = arma::repmat(arma::exp(-far_drift_h()), 1, draws);
    posterior.u_sigma_inv = arma::repmat(
        arma::vectorise(arma::trans(psi) * arma::diagmat(arma::exp(-far_drift_h())) * psi), 1,
        draws);
    posterior.h_sigma = arma::mat(k, draws, arma::fill::value(4.0));
    if constexpr (requires { posterior.psi_sigma; })
    {
        posterior.psi_sigma = arma::mat(k * (k - 1) / 2, draws, arma::fill::value(0.01));
    }

    bayests::NullReporter reporter;
    const arma::mat simulated = Sampler{}.forecast(input, posterior, reporter).values;
    unsigned long long non_finite = 0;
    simulated.for_each([&](const double value) { non_finite += std::isfinite(value) ? 0 : 1; });
    std::printf("    %llu of %llu forecasts not finite\n", non_finite,
                static_cast<unsigned long long>(simulated.n_elem));
    check("every simulated forecast is finite", simulated.is_finite());
}

/// Every VAR's forecast from a precision that stays where the sample ends, set
/// to the badly conditioned one above: its inverse has an eigenvalue a rounding
/// error below zero, and every forecast must still be a number.
template <typename Sampler, typename Input, typename Draws>
void held_far_apart_stays_finite(const char *name)
{
    std::printf("%s: held, from log-volatilities far apart\n", name);

    constexpr int k = 4;
    constexpr int h = 4;
    constexpr arma::uword draws = 500;

    Input input;
    input.spec.k = k;
    input.spec.h = h;
    input.spec.forecast_states = ForecastStates::hold;

    const arma::mat psi = far_drift_psi();
    Draws posterior;
    posterior.u_sigma_inv = arma::repmat(
        arma::vectorise(arma::trans(psi) * arma::diagmat(arma::exp(-far_drift_h())) * psi), 1,
        draws);

    bayests::NullReporter reporter;
    check("every held forecast is finite",
          Sampler{}.forecast(input, posterior, reporter).values.is_finite());
}

} // namespace

int main()
{
    arma::arma_rng::set_seed(20260914);

    coefficients_drift();
    excluded_coefficients_stay_excluded();
    covariance_block_drifts();
    volatility_drifts<bayests::VarTvpStochvolSampler, bayests::VarTvpStochvolInput,
                      bayests::VarTvpStochvolDraws>("VarTvpStochvol");
    volatility_drifts<bayests::VarNormalStochvolSampler, bayests::VarNormalStochvolInput,
                      bayests::VarNormalStochvolDraws>("VarNormalStochvol");
    factor_volatilities_drift<bayests::DfmNormalStochvolSampler, bayests::DfmNormalStochvolInput,
                              bayests::DfmNormalStochvolDraws>("DfmNormalStochvol");
    factor_volatilities_drift<bayests::DfmTvpStochvolSampler, bayests::DfmTvpStochvolInput,
                              bayests::DfmTvpStochvolDraws>("DfmTvpStochvol");
    factor_coefficients_drift();
    vec_coefficients_drift();
    vec_volatility_drifts<bayests::VecNormalStochvolSampler, bayests::VecNormalStochvolInput,
                          bayests::VecNormalStochvolDraws>("VecNormalStochvol");
    vec_volatility_drifts<bayests::VecTvpStochvolSampler, bayests::VecTvpStochvolInput,
                          bayests::VecTvpStochvolDraws>("VecTvpStochvol");
    covariance_root_far_apart();
    far_drift_stays_finite<bayests::VarTvpStochvolSampler, bayests::VarTvpStochvolInput,
                           bayests::VarTvpStochvolDraws>("VarTvpStochvol", false);
    far_drift_stays_finite<bayests::VarNormalStochvolSampler, bayests::VarNormalStochvolInput,
                           bayests::VarNormalStochvolDraws>("VarNormalStochvol", false);
    far_drift_stays_finite<bayests::VecTvpStochvolSampler, bayests::VecTvpStochvolInput,
                           bayests::VecTvpStochvolDraws>("VecTvpStochvol", true);
    far_drift_stays_finite<bayests::VecNormalStochvolSampler, bayests::VecNormalStochvolInput,
                           bayests::VecNormalStochvolDraws>("VecNormalStochvol", true);
    held_far_apart_stays_finite<bayests::VarNormalWishartSampler, bayests::VarNormalWishartInput,
                                bayests::VarNormalWishartDraws>("VarNormalWishart");
    held_far_apart_stays_finite<bayests::VarNormalGammaSampler, bayests::VarNormalGammaInput,
                                bayests::VarNormalGammaDraws>("VarNormalGamma");
    held_far_apart_stays_finite<bayests::VarNormalStochvolSampler, bayests::VarNormalStochvolInput,
                                bayests::VarNormalStochvolDraws>("VarNormalStochvol");
    held_far_apart_stays_finite<bayests::VarTvpWishartSampler, bayests::VarTvpWishartInput,
                                bayests::VarTvpWishartDraws>("VarTvpWishart");
    held_far_apart_stays_finite<bayests::VarTvpGammaSampler, bayests::VarTvpGammaInput,
                                bayests::VarTvpGammaDraws>("VarTvpGamma");
    held_far_apart_stays_finite<bayests::VarTvpStochvolSampler, bayests::VarTvpStochvolInput,
                                bayests::VarTvpStochvolDraws>("VarTvpStochvol");

    if (failures > 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
