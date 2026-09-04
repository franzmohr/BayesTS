// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_var_ald.cpp
/// @brief Checks that the quantile VARs estimate a quantile.
///
/// The first unit test of a VAR sampler here -- the others are covered by the
/// golden fixtures alone, which run them and record fingerprints. That is enough
/// for a model whose failure mode is throwing or writing nothing. It is not
/// enough for these two, because their failure mode is silent: a quantile model
/// that has lost its skew term somewhere estimates the *median* instead, writes
/// a full set of well-conditioned draws, and passes every smoke test there is.
///
/// So the test is built around the property that defines what the model claims
/// to be. If the coefficients really locate the q-th conditional quantile, then
/// the fitted residuals fall below zero a fraction q of the time -- that is not
/// a consequence of the estimator, it is the definition of the thing being
/// estimated, and it is what the check loss is minimised to achieve. A model
/// fitting the mean or the median lands at one half whatever q was asked for,
/// which is a gap no tolerance hides.
///
/// Two supporting checks around it. The shape constants at q = 0.5 are exact
/// integers, which pins the arithmetic that the skew is built from. And the
/// coefficients at three quantiles have to be ordered as the quantiles are, on
/// data whose conditional distribution is known -- which catches a model that
/// responds to q in the wrong direction, something a single-quantile residual
/// check cannot see.

#include "bayests/var_normal_ald.h"
#include "bayests/var_tvp_ald.h"
#include "core/models/ald_support.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok, const std::string &detail = "")
{
    std::printf("  %-56s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!detail.empty())
    {
        std::printf("      %s\n", detail.c_str());
    }
    if (!ok)
    {
        failures++;
    }
}

void check_close(const std::string &what, const double got, const double want, const double tol)
{
    const bool ok = std::abs(got - want) < tol;
    std::printf("  %-56s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    std::printf("      got %.4f, want %.4f, tolerance %.4f\n", got, want, tol);
    if (!ok)
    {
        failures++;
    }
}

/// One equation, an intercept and one regressor, normal errors.
///
/// The conditional distribution is known throughout, so the true q-th quantile
/// of y given x is `intercept + z_q + slope * x` with z_q the standard normal
/// quantile -- the slope is the same at every quantile and only the intercept
/// moves. That separation is what the ordering check below reads.
struct Sample
{
    arma::mat y;
    arma::mat z;
    int tt = 0;
};

Sample make_sample(const int tt, const double intercept, const double slope)
{
    Sample out;
    out.tt = tt;
    out.z = arma::mat(tt, 2);
    out.y = arma::mat(tt, 1);

    for (int t = 0; t < tt; t++)
    {
        const double x = std::sin(0.7 * t) + 0.3 * std::cos(2.1 * t);
        out.z(t, 0) = 1.0;
        out.z(t, 1) = x;
        out.y(t, 0) = intercept + slope * x + arma::randn<double>();
    }
    return out;
}

bayests::VarNormalAldInput make_input(const Sample &sample, const double quantile,
                                      const int iterations, const int burnin)
{
    bayests::VarNormalAldInput input;

    input.spec.k = 1;
    input.spec.p = 0;
    input.spec.n = 2; // the intercept and the regressor
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.spec.quantile = quantile;

    input.train.y = sample.y;
    input.train.z = sample.z;

    input.a_prior.mu = arma::vec(2, arma::fill::zeros);
    input.a_prior.v_inv = arma::eye<arma::mat>(2, 2) * 0.01;

    input.u_scale_prior.shape = arma::vec(1, arma::fill::value(3.0));
    input.u_scale_prior.rate = arma::vec(1, arma::fill::value(0.2));

    input.initial.a = arma::vec(2, arma::fill::zeros);
    input.initial.w = arma::mat(sample.tt, 1, arma::fill::ones);
    input.initial.u_scale = arma::vec(1, arma::fill::ones);

    return input;
}

/// The posterior mean of the coefficients.
arma::vec posterior_mean(const arma::mat &a)
{
    return arma::mean(a, 1);
}

/// The share of fitted residuals that fall below zero.
double share_below_zero(const Sample &sample, const arma::vec &a)
{
    const arma::vec fitted = sample.z * a;
    const arma::vec residual = sample.y.col(0) - fitted;
    return static_cast<double>(arma::accu(residual < 0.0)) / static_cast<double>(sample.tt);
}

void the_shape_constants_are_exact_at_the_median()
{
    std::printf("the shape constants\n");

    const bayests::core::AldShape median = bayests::core::ald_shape(0.5);
    check("the skew at the median is exactly zero", median.theta == 0.0);
    check("the variance multiplier at the median is exactly eight", median.tau2 == 8.0);

    // Away from the median the skew has the sign of (1 - 2q): positive below,
    // negative above. Getting this backwards is the error that would fit
    // 1 - q instead of q.
    check("the skew is positive below the median", bayests::core::ald_shape(0.25).theta > 0.0);
    check("the skew is negative above the median", bayests::core::ald_shape(0.75).theta < 0.0);

    bool threw = false;
    try
    {
        bayests::core::ald_shape(0.0);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    check("a quantile of zero is refused", threw);
}

void it_estimates_the_quantile_it_was_asked_for()
{
    std::printf("the fitted quantile\n");

    const double quantiles[3] = {0.25, 0.5, 0.8};

    for (int i = 0; i < 3; i++)
    {
        arma::arma_rng::set_seed(4321);
        const Sample sample = make_sample(400, 1.0, 0.5);
        const auto input = make_input(sample, quantiles[i], 600, 300);

        bayests::NullReporter reporter;
        const auto draws = bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter);

        const arma::vec a = posterior_mean(draws.a);
        const double share = share_below_zero(sample, a);

        // The defining property: a fitted q-th quantile has a fraction q of the
        // sample beneath it. A model that had lost its skew would sit at one
        // half here whatever was asked for.
        check_close("the share of residuals below the fit is q, q = " +
                        std::to_string(quantiles[i]).substr(0, 4),
                    share, quantiles[i], 0.06);
    }
}

void the_coefficients_are_ordered_as_the_quantiles_are()
{
    std::printf("the ordering\n");

    const double intercept = 1.0;
    const double slope = 0.5;

    arma::vec fitted_intercept(3);
    arma::vec fitted_slope(3);
    const double quantiles[3] = {0.1, 0.5, 0.9};

    for (int i = 0; i < 3; i++)
    {
        arma::arma_rng::set_seed(99);
        const Sample sample = make_sample(500, intercept, slope);
        const auto input = make_input(sample, quantiles[i], 600, 300);

        bayests::NullReporter reporter;
        const auto draws = bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter);

        const arma::vec a = posterior_mean(draws.a);
        fitted_intercept(i) = a(0);
        fitted_slope(i) = a(1);
    }

    check("the intercept rises with the quantile",
          fitted_intercept(0) < fitted_intercept(1) && fitted_intercept(1) < fitted_intercept(2),
          "10%: " + std::to_string(fitted_intercept(0)) + "  50%: " +
              std::to_string(fitted_intercept(1)) + "  90%: " + std::to_string(fitted_intercept(2)));

    // The errors do not depend on x, so every quantile of this data has the same
    // slope, and a model that mixed the skew into the regressors rather than
    // into the intercept would fan these out.
    //
    // The bound is loose on purpose, and comes from the asymptotic standard
    // error rather than from what looks tight. A quantile regression slope is
    // estimated with precision sqrt(q(1-q)) / f(F^-1(q)), and for normal errors
    // that is about 1.7 at the deciles against 1.25 at the median -- so the tail
    // slopes here carry a standard error near 0.10 and the median one near 0.08.
    // A tighter bound would fail on the sampling noise of a correct model, which
    // is the failure mode that teaches nothing.
    for (int i = 0; i < 3; i++)
    {
        check_close("the slope at q = " + std::to_string(quantiles[i]).substr(0, 3) +
                        " is the true one",
                    fitted_slope(i), slope, 0.3);
    }

    // And the middle one is the truth, since the median of a normal is its mean.
    check_close("the median intercept is the true one", fitted_intercept(1), intercept, 0.2);
    check_close("the median slope is the true one", fitted_slope(1), slope, 0.15);
}

void the_time_varying_model_estimates_the_same_quantile()
{
    std::printf("the time-varying model\n");

    const double quantile = 0.3;

    arma::arma_rng::set_seed(2024);
    const Sample sample = make_sample(300, 1.0, 0.5);

    bayests::VarTvpAldInput input;
    input.spec.k = 1;
    input.spec.p = 0;
    input.spec.n = 2;
    input.spec.iterations = 400;
    input.spec.burnin = 200;
    input.spec.quantile = quantile;

    input.train.y = sample.y;
    input.train.z = sample.z;

    // A tight state equation, so the path stays close to constant and the
    // fitted quantile is comparable with the constant-coefficient model's.
    input.a_prior.sigma.shape = arma::vec(2, arma::fill::value(10.0));
    input.a_prior.sigma.rate = arma::vec(2, arma::fill::value(0.0001));
    input.a_prior.initial_state.mu = arma::vec(2, arma::fill::zeros);
    input.a_prior.initial_state.v_inv = arma::eye<arma::mat>(2, 2) * 0.01;

    input.u_scale_prior.shape = arma::vec(1, arma::fill::value(3.0));
    input.u_scale_prior.rate = arma::vec(1, arma::fill::value(0.2));

    input.initial.a = arma::mat(2, sample.tt, arma::fill::zeros);
    input.initial.a_sigma_inv = arma::diagmat(arma::vec(2, arma::fill::value(100.0)));
    input.initial.a_init = arma::vec(2, arma::fill::zeros);
    input.initial.w = arma::mat(sample.tt, 1, arma::fill::ones);
    input.initial.u_scale = arma::vec(1, arma::fill::ones);

    bayests::NullReporter reporter;
    const auto draws = bayests::VarTvpAldSampler{}.draw_coefficients(input, reporter);

    check("the path has one coefficient vector per period",
          draws.a.n_rows == static_cast<arma::uword>(2 * sample.tt));
    check("every drawn coefficient is finite", draws.a.is_finite());
    check("every drawn scale is positive", draws.u_scale.min() > 0.0);

    // The same defining property, scored period by period under that period's
    // own coefficients.
    const arma::vec a_mean = arma::mean(draws.a, 1);
    int below = 0;
    for (int t = 0; t < sample.tt; t++)
    {
        const double fitted = sample.z(t, 0) * a_mean(2 * t) + sample.z(t, 1) * a_mean(2 * t + 1);
        if (sample.y(t, 0) - fitted < 0.0)
        {
            below++;
        }
    }
    check_close("the share of residuals below the fit is q",
                static_cast<double>(below) / static_cast<double>(sample.tt), quantile, 0.08);
}

void the_log_likelihood_is_the_asymmetric_laplace()
{
    std::printf("the log likelihood\n");

    arma::arma_rng::set_seed(11);
    const Sample sample = make_sample(120, 1.0, 0.5);
    const auto input = make_input(sample, 0.4, 100, 50);

    bayests::NullReporter reporter;
    const auto draws = bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter);
    const arma::mat loglik = bayests::VarNormalAldSampler{}.log_likelihood(input, draws);

    check("the log likelihood is draws x periods",
          loglik.n_rows == draws.iterations() &&
              loglik.n_cols == static_cast<arma::uword>(sample.tt));
    check("the log likelihood is finite", loglik.is_finite());

    // Checked against the density written out by hand, on the first draw and
    // the first period, so a sign slip in the check function cannot hide.
    const double q = 0.4;
    const arma::vec a = draws.a.col(0);
    const double residual = sample.y(0, 0) - arma::dot(sample.z.row(0), a);
    const double scale = draws.u_scale(0, 0);
    const double v = residual / scale;
    const double expected =
        std::log(q) + std::log(1.0 - q) - std::log(scale) - v * (q - (v < 0.0 ? 1.0 : 0.0));
    check_close("it is the density written out by hand", loglik(0, 0), expected, 1e-12);
}

void the_forecast_is_refused()
{
    std::printf("the forecast\n");

    arma::arma_rng::set_seed(5);
    const Sample sample = make_sample(60, 1.0, 0.5);
    const auto input = make_input(sample, 0.5, 60, 20);

    bayests::NullReporter reporter;
    const auto draws = bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter);

    bool threw = false;
    std::string message;
    try
    {
        bayests::VarNormalAldSampler{}.forecast(input, draws, reporter);
    }
    catch (const std::invalid_argument &e)
    {
        threw = true;
        message = e.what();
    }
    check("a forecast is refused", threw);
    check("the refusal says why", message.find("quantile of the iterated") != std::string::npos,
          message);
}

void expect_rejected(const std::string &what, const bayests::VarNormalAldInput &input)
{
    bool threw = false;
    std::string message;
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &e)
    {
        threw = true;
        message = e.what();
    }
    check(what, threw, threw ? message : "accepted");
}

void the_impossible_inputs_are_refused()
{
    std::printf("the refusals\n");

    arma::arma_rng::set_seed(1);
    const Sample sample = make_sample(40, 1.0, 0.5);
    const auto base = make_input(sample, 0.3, 20, 10);

    bool accepted = true;
    try
    {
        base.validate();
    }
    catch (const std::exception &)
    {
        accepted = false;
    }
    check("a well-formed input is accepted", accepted);

    auto with = [&](auto mutate) {
        auto copy = base;
        mutate(copy);
        return copy;
    };

    expect_rejected("a quantile of zero",
                    with([](auto &in) { in.spec.quantile = 0.0; }));
    expect_rejected("a quantile of one",
                    with([](auto &in) { in.spec.quantile = 1.0; }));
    expect_rejected("a quantile above one",
                    with([](auto &in) { in.spec.quantile = 1.5; }));
    expect_rejected("a covariance block",
                    with([](auto &in) { in.spec.k = 2; in.spec.covar = true; }));
    expect_rejected("a forecast horizon",
                    with([](auto &in) { in.spec.h = 4; }));
    expect_rejected("stochastic search variable selection",
                    with([](auto &in) { in.spec.varsel = bayests::VarSelection::ssvs; }));
    expect_rejected("a latent scale of zero",
                    with([](auto &in) { in.initial.w(3, 0) = 0.0; }));
    expect_rejected("a negative scale",
                    with([](auto &in) { in.initial.u_scale(0) = -1.0; }));
    expect_rejected("a latent scale path of the wrong length",
                    with([](auto &in) { in.initial.w = arma::mat(5, 1, arma::fill::ones); }));
}

} // namespace

int main()
{
    std::printf("unit_var_ald\n");

    the_shape_constants_are_exact_at_the_median();
    it_estimates_the_quantile_it_was_asked_for();
    the_coefficients_are_ordered_as_the_quantiles_are();
    the_time_varying_model_estimates_the_same_quantile();
    the_log_likelihood_is_the_asymmetric_laplace();
    the_forecast_is_refused();
    the_impossible_inputs_are_refused();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
