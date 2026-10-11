// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_volatility_break.cpp
/// @brief Checks the volatility break of Lenza and Primiceri (2022) in
/// VarNormalWishart.
///
/// From a known period the error covariance is s_t^2 Sigma, with free scales
/// for the first periods and a decay after them. The checks, in order: the
/// path; the gamma restricted to (0, 1] that the exact draw of a free scale
/// reduces to, against quadrature; that draw itself, and the Metropolis-Hastings
/// step of the last scale and the decay rate, each against the conditional
/// they target, computed by quadrature at fixed residuals; the log likelihood
/// and the score against the dense normal; the forecast's spread, and a
/// scenario's, against the scale the path puts on each horizon; recovery on a
/// simulated VAR whose shocks jump, and that ignoring the jump distorts the
/// coefficients where modelling it does not; and what is refused is refused.

#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_wishart.h"
#include "core/models/volatility_break_support.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-70s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_below(const std::string &what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-70s %s   %.3e < %.3e\n", what.c_str(), ok ? "ok" : "FAILED", got, bound);
    if (!ok)
    {
        failures++;
    }
}

constexpr arma::uword kK = 3;
constexpr arma::uword kP = 1;
constexpr arma::uword kNx = kK * kP + 1;

void test_path()
{
    std::printf("the path\n");
    bayests::VarSpec spec;
    spec.volatility_break = 5;
    spec.volatility_scales = 3;
    const arma::vec theta = {10.0, 40.0, 9.0, 0.5};
    const arma::vec s = bayests::core::volatility_path(spec, theta, 0, 9);
    const arma::vec want = {1, 1, 1, 1, 10, 40, 9, 5, 3};
    check_below("ones, three free scales, then 1 + (s2 - 1) rho^j", arma::abs(s - want).max(), 1e-14);
    const arma::vec later = bayests::core::volatility_path(spec, theta, 7, 3);
    check_below("and the same path from an offset, into a horizon",
                arma::abs(later - arma::vec({5.0, 3.0, 2.0})).max(), 1e-14);
    spec.volatility_break = 0;
    check("no break is ones", arma::all(bayests::core::volatility_path(spec, theta, 0, 9) == 1.0));
}

/// The mean of Gamma(a, rate b) restricted to (0, 1], by the midpoint rule.
double truncated_gamma_mean(const double a, const double b)
{
    const int n = 200000;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++)
    {
        const double w = (i + 0.5) / n;
        const double f = std::exp((a - 1.0) * std::log(w) - b * w);
        num += w * f;
        den += f;
    }
    return num / den;
}

void test_truncated_gamma()
{
    std::printf("a gamma restricted to (0, 1]\n");
    arma::arma_rng::set_seed(11);
    // Both samplers: a small rate takes the power proposal, a large one the
    // gamma; the third sits near where the two cross.
    const double cases[][2] = {{3.5, 0.2}, {3.5, 60.0}, {3.5, 3.0}, {10.5, 6.0}};
    for (const auto &c : cases)
    {
        const int n = 40000;
        arma::vec w(n);
        for (int i = 0; i < n; i++)
        {
            w(i) = bayests::core::draw_gamma_below_one(c[0], c[1]);
        }
        const double want = truncated_gamma_mean(c[0], c[1]);
        const double se = arma::stddev(w) / std::sqrt(static_cast<double>(n));
        check("shape " + std::to_string(c[0]).substr(0, 4) + ", rate " +
                  std::to_string(c[1]).substr(0, 4) + ": in (0, 1], mean within 4 se of quadrature",
              w.max() <= 1.0 && w.min() > 0.0 && std::abs(arma::mean(w) - want) < 4.0 * se);
    }
}

/// Residuals for a break at period 1 (zero-based) over `tt` periods: unit
/// residuals scaled so that period t's quadratic form is q(t).
arma::mat residuals_with(const arma::vec &q)
{
    arma::mat u(kK, q.n_elem, arma::fill::zeros);
    for (arma::uword t = 0; t < q.n_elem; t++)
    {
        u(0, t) = std::sqrt(q(t));
    }
    return u;
}

void test_conditionals()
{
    std::printf("the draw of the scales, against their conditionals at fixed residuals\n");
    bayests::VarSpec spec;
    spec.k = static_cast<int>(kK);
    spec.volatility_break = 2;
    spec.volatility_scales = 2;
    bayests::VolatilityBreakPrior prior;
    prior.shape = 1.0;
    prior.rho_shape1 = 3.0;
    prior.rho_shape2 = 1.5;

    // Break at period 1: s0 moves period 1 alone, s1 period 2, and s1 and rho
    // the four after it. q is what scales of about 6, 15, 1 + 14 rho^j produce.
    const arma::vec q = {2.0, 110.0, 700.0, 300.0, 120.0, 50.0, 25.0};
    const arma::uword tt = q.n_elem;
    const arma::mat u = residuals_with(q);
    const arma::mat precision = arma::eye<arma::mat>(kK, kK);

    bayests::core::VolatilityBreak block(spec, prior, arma::vec(), tt, kK);
    arma::arma_rng::set_seed(21);
    const int burn = 4000, n = 60000;
    arma::mat kept(3, n);
    for (int i = 0; i < burn + n; i++)
    {
        block.update(u, precision, i < burn);
        if (i >= burn)
        {
            kept.col(i - burn) = block.theta();
        }
    }

    // s0: p(s) proportional to s^-(k + shape + 1) exp(-q / (2 s^2)) on s >= 1.
    const double k = static_cast<double>(kK);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < 400000; i++)
    {
        const double s = 1.0 + (i + 0.5) * 1e-4;
        const double f = std::exp(-(k + prior.shape + 1.0) * std::log(s) - q(1) / (2.0 * s * s));
        num += s * f;
        den += f;
    }
    const double s0_mean = num / den;
    const double s0_got = arma::mean(kept.row(0));
    std::printf("    s0: chain %.4f, quadrature %.4f\n", s0_got, s0_mean);
    check_below("s0, an exact draw, has the conditional's mean (relative error)",
                std::abs(s0_got / s0_mean - 1.0), 0.01);

    // (s1, rho): the same target on a grid in s1 and rho.
    double m_s = 0.0, m_r = 0.0, z = 0.0;
    double log_max = -1e300;
    std::vector<double> logf;
    const int gs = 1200, gr = 600;
    for (int i = 0; i < gs; i++)
    {
        const double s1 = 1.0 + (i + 0.5) * 0.05;
        for (int j = 0; j < gr; j++)
        {
            const double rho = (j + 0.5) / gr;
            double lf = -(prior.shape + 1.0) * std::log(s1) + (prior.rho_shape1 - 1.0) * std::log(rho) +
                        (prior.rho_shape2 - 1.0) * std::log(1.0 - rho);
            for (arma::uword t = 2; t < tt; t++)
            {
                const double s = 1.0 + (s1 - 1.0) * std::pow(rho, static_cast<double>(t - 2));
                lf += -k * std::log(s) - q(t) / (2.0 * s * s);
            }
            logf.push_back(lf);
            log_max = std::max(log_max, lf);
        }
    }
    int at = 0;
    for (int i = 0; i < gs; i++)
    {
        const double s1 = 1.0 + (i + 0.5) * 0.05;
        for (int j = 0; j < gr; j++)
        {
            const double rho = (j + 0.5) / gr;
            const double f = std::exp(logf[at++] - log_max);
            m_s += s1 * f;
            m_r += rho * f;
            z += f;
        }
    }
    m_s /= z;
    m_r /= z;
    const double s1_got = arma::mean(kept.row(1));
    const double r_got = arma::mean(kept.row(2));
    std::printf("    s1: chain %.4f, quadrature %.4f; rho: chain %.4f, quadrature %.4f\n", s1_got, m_s,
                r_got, m_r);
    check_below("the Metropolis-Hastings step targets the conditional of s1 (relative error)",
                std::abs(s1_got / m_s - 1.0), 0.02);
    check_below("and of rho", std::abs(r_got / m_r - 1.0), 0.02);

    // A free scale past the sample is drawn from its Pareto prior: with shape
    // 3 its mean is 1.5.
    bayests::VarSpec late = spec;
    late.volatility_break = static_cast<int>(tt);
    late.volatility_scales = 3;
    bayests::VolatilityBreakPrior p3 = prior;
    p3.shape = 3.0;
    bayests::core::VolatilityBreak beyond(late, p3, arma::vec(), tt, kK);
    arma::arma_rng::set_seed(22);
    arma::vec s1_prior(40000);
    for (arma::uword i = 0; i < s1_prior.n_elem; i++)
    {
        beyond.update(u, precision, true);
        s1_prior(i) = beyond.theta()(1);
    }
    check_below("a free scale past the sample is drawn from its Pareto prior (mean 1.5)",
                std::abs(arma::mean(s1_prior) - 1.5), 0.02);
}

struct Sample
{
    arma::mat y;      ///< tt x k
    arma::mat z;      ///< SUR regressors
    arma::mat a_lag;  ///< k x k
    arma::vec c;      ///< intercept
    arma::vec s;      ///< the scale of every period
    arma::mat last;   ///< the last observation, for a forecast's regressors
};

/// A VAR(1) whose shocks jump at `break_at` (one-based) and then decay.
Sample simulate(const arma::uword tt, const int break_at, const arma::vec &theta, const int scales)
{
    Sample s;
    s.a_lag = {{0.6, 0.1, 0.0}, {0.0, 0.5, 0.1}, {0.1, 0.0, 0.4}};
    s.c = {0.2, -0.1, 0.1};
    bayests::VarSpec spec;
    spec.volatility_break = break_at;
    spec.volatility_scales = scales;
    s.s = bayests::core::volatility_path(spec, theta, 0, tt);
    const arma::mat chol_sigma = {{0.5, 0.0, 0.0}, {0.2, 0.4, 0.0}, {-0.1, 0.1, 0.3}};
    arma::arma_rng::set_seed(808);
    arma::mat path(kK, tt + 1, arma::fill::zeros);
    for (arma::uword t = 1; t <= tt; t++)
    {
        path.col(t) = s.c + s.a_lag * path.col(t - 1) + s.s(t - 1) * chol_sigma * arma::randn<arma::vec>(kK);
    }
    s.y = arma::trans(path.cols(1, tt));
    arma::mat x(tt, kNx);
    x.col(kNx - 1).ones();
    x.cols(0, kK - 1) = arma::trans(path.cols(0, tt - 1));
    s.z = arma::kron(x, arma::eye<arma::mat>(kK, kK));
    s.last = path.col(tt);
    return s;
}

bayests::VarNormalWishartInput input_for(const Sample &s, const int break_at, const int iterations,
                                         const int burnin)
{
    bayests::VarNormalWishartInput input;
    input.spec.k = static_cast<int>(kK);
    input.spec.p = static_cast<int>(kP);
    input.spec.n = 1;
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.spec.volatility_break = break_at;
    input.spec.volatility_scales = 3;
    input.train.y = s.y;
    input.train.z = s.z;
    const arma::uword n = kK * kNx;
    input.a_prior.mu = arma::zeros<arma::vec>(n);
    input.a_prior.v_inv = 0.01 * arma::eye<arma::mat>(n, n);
    input.u_sigma_prior.df = static_cast<int>(kK) + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    input.initial.a = arma::zeros<arma::vec>(n);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    input.volatility_prior.shape = 1.0;
    input.volatility_prior.rho_shape1 = 3.0357;
    input.volatility_prior.rho_shape2 = 1.5089;
    return input;
}

void test_recovery()
{
    std::printf("recovery on a VAR whose shocks jump\n");
    const arma::uword tt = 300;
    const int break_at = 281;
    const arma::vec truth = {12.0, 30.0, 8.0, 0.7};
    const Sample s = simulate(tt, break_at, truth, 3);
    bayests::NullReporter reporter;

    const auto input = input_for(s, break_at, 3000, 1500);
    arma::arma_rng::set_seed(1);
    const auto draws = bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);
    check("one column of the volatility draws per kept draw",
          draws.volatility.n_rows == 4 && draws.volatility.n_cols == 3000);

    const arma::vec lo = arma::quantile(draws.volatility, arma::vec({0.005}), 1);
    const arma::vec hi = arma::quantile(draws.volatility, arma::vec({0.995}), 1);
    const arma::vec med = arma::median(draws.volatility, 1);
    std::printf("    medians %.2f %.2f %.2f %.3f, truth 12 30 8 0.7\n", med(0), med(1), med(2), med(3));
    check("every parameter's 99% interval covers the truth",
          arma::all(lo <= truth) && arma::all(hi >= truth));

    // The coefficients: what the jump does to them when it is ignored.
    auto ignore = input;
    ignore.spec.volatility_break = 0;
    arma::arma_rng::set_seed(1);
    const auto naive = bayests::VarNormalWishartSampler{}.draw_coefficients(ignore, reporter);
    arma::mat truth_a(kK, kNx);
    truth_a.cols(0, kK - 1) = s.a_lag;
    truth_a.col(kNx - 1) = s.c;
    const double err_break = arma::abs(arma::reshape(arma::mean(draws.a, 1), kK, kNx) - truth_a).max();
    const double err_naive = arma::abs(arma::reshape(arma::mean(naive.a, 1), kK, kNx) - truth_a).max();
    // The benchmark is what the clean periods alone say: OLS on the sample
    // before the break, whose own sampling error is the floor for both.
    const arma::uword pre = static_cast<arma::uword>(break_at - 1);
    arma::mat x(pre, kNx);
    for (arma::uword t = 0; t < pre; t++)
    {
        for (arma::uword j = 0; j < kNx; j++)
        {
            x(t, j) = s.z(t * kK, j * kK);
        }
    }
    const double err_clean =
        arma::abs(arma::trans(arma::solve(x, s.y.head_rows(pre))) - truth_a).max();
    std::printf("    worst coefficient error: %.3f with the break, %.3f ignoring it, %.3f by OLS "
                "before it\n",
                err_break, err_naive, err_clean);
    check_below("modelled, the coefficients are as close as the clean periods allow",
                err_break, err_clean + 0.05);
    check("ignored, the jump pulls them further away", err_naive > err_break);
}

/// A posterior by hand: every draw the same coefficients and precision, and
/// the volatility parameters given.
bayests::VarNormalWishartDraws fixed_draws(const arma::mat &a, const arma::mat &sigma,
                                           const arma::vec &theta, const arma::uword n)
{
    bayests::VarNormalWishartDraws d;
    d.a = arma::repmat(arma::vectorise(a), 1, n);
    d.u_sigma_inv = arma::repmat(arma::vectorise(arma::inv_sympd(sigma)), 1, n);
    d.volatility = arma::repmat(theta, 1, n);
    return d;
}

void test_likelihood_and_score()
{
    std::printf("the log likelihood and the score, against the dense normal\n");
    const arma::uword tt = 40;
    const int break_at = 31;
    const arma::vec theta = {6.0, 14.0, 5.0, 0.6};
    const Sample s = simulate(tt + 4, break_at, theta, 3);
    const arma::mat sigma = {{0.25, 0.08, -0.05}, {0.08, 0.2, 0.04}, {-0.05, 0.04, 0.14}};
    arma::mat a(kK, kNx);
    a.cols(0, kK - 1) = s.a_lag;
    a.col(kNx - 1) = s.c;

    // Train on the first tt periods, score the four after.
    Sample train = s;
    train.y = s.y.head_rows(tt);
    train.z = s.z.head_rows(tt * kK);
    auto input = input_for(train, break_at, 2, 0);
    const auto draws = fixed_draws(a, sigma, theta, 2);
    const arma::mat ll = bayests::VarNormalWishartSampler{}.log_likelihood(input, draws);

    bayests::VarSpec spec = input.spec;
    const arma::vec path = bayests::core::volatility_path(spec, theta, 0, tt + 4);
    const auto dense = [&](const arma::uword t) {
        // The simulation starts from zero, so the first period's lag is zero.
        const arma::vec lag =
            t == 0 ? arma::vec(kK, arma::fill::zeros) : arma::vec(arma::trans(s.y.row(t - 1)));
        const arma::vec x = arma::join_cols(lag, arma::vec({1.0}));
        const arma::vec e = arma::trans(s.y.row(t)) - a * x;
        const arma::mat cov = path(t) * path(t) * sigma;
        double logdet, sign;
        arma::log_det(logdet, sign, cov);
        return -0.5 * (kK * std::log(2.0 * arma::datum::pi) + logdet +
                       arma::as_scalar(arma::trans(e) * arma::solve(cov, e)));
    };
    double worst = 0.0;
    for (arma::uword t = 0; t < tt; t++)
    {
        worst = std::max(worst, std::abs(ll(0, t) - dense(t)));
    }
    check_below("every period's log density is the normal with s_t^2 Sigma", worst, 1e-10);

    input.spec.h = 4;
    input.test.y = s.y.rows(tt, tt + 3);
    input.forecast.x = arma::mat(4, kNx, arma::fill::zeros);
    input.forecast.x.col(kNx - 1).ones();
    input.forecast.x.submat(0, 0, 0, kK - 1) = s.y.row(tt - 1);
    const arma::mat score = bayests::VarNormalWishartSampler{}.predictive_log_density(input, draws);
    worst = 0.0;
    for (arma::uword i = 0; i < 4; i++)
    {
        worst = std::max(worst, std::abs(score(0, i) - dense(tt + i)));
    }
    check_below("and the score places the horizon on the path: periods tt and on", worst, 1e-10);
}

void test_forecast()
{
    std::printf("the forecast's spread follows the path into the horizon\n");
    const arma::uword tt = 30;
    const int break_at = 29; // the sample ends on s1, so the horizon decays from s2
    const arma::vec theta = {6.0, 20.0, 9.0, 0.5};
    const Sample s = simulate(tt, break_at, theta, 3);
    auto input = input_for(s, break_at, 2, 0);
    const int h = 4;
    input.spec.h = h;
    input.forecast.x = arma::mat(h, kNx, arma::fill::zeros);
    input.forecast.x.col(kNx - 1).ones();

    // Coefficients zero, so a forecast is its error alone.
    const arma::mat sigma = {{1.0, 0.6, 0.0}, {0.6, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const arma::uword n = 40000;
    const auto draws = fixed_draws(arma::zeros<arma::mat>(kK, kNx), sigma, theta, n);
    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(5);
    const arma::mat f = bayests::VarNormalWishartSampler{}.forecast(input, draws, reporter).values;
    bayests::VarSpec spec = input.spec;
    const arma::vec want = bayests::core::volatility_path(spec, theta, tt, h);
    double worst = 0.0;
    for (int i = 0; i < h; i++)
    {
        worst = std::max(worst, std::abs(arma::stddev(f.row(i * kK)) / want(i) - 1.0));
    }
    std::printf("    horizon scales %.2f %.2f %.2f %.2f\n", want(0), want(1), want(2), want(3));
    check_below("each horizon's standard deviation is s_{T+i} (relative error)", worst, 0.02);

    // A scenario: the first variable pinned at zero at every horizon, so the
    // second keeps s_{T+i} sqrt(1 - 0.6^2) of its spread.
    bayests::Constraints c;
    c.row = arma::regspace<arma::uvec>(0, h - 1);
    c.period = arma::regspace<arma::uvec>(0, h - 1);
    c.variable = arma::uvec(h, arma::fill::zeros);
    c.weight = arma::vec(h, arma::fill::ones);
    c.value = arma::vec(h, arma::fill::zeros);
    c.group = arma::uvec(h, arma::fill::zeros);
    input.forecast.constraints = c;
    const auto few = fixed_draws(arma::zeros<arma::mat>(kK, kNx), sigma, theta, 8000);
    arma::arma_rng::set_seed(6);
    const arma::mat g = bayests::VarNormalWishartSampler{}.forecast(input, few, reporter).values;
    worst = 0.0;
    double pinned = 0.0;
    for (int i = 0; i < h; i++)
    {
        pinned = std::max(pinned, arma::abs(g.row(i * kK)).max());
        worst = std::max(worst, std::abs(arma::stddev(g.row(i * kK + 1)) / (want(i) * 0.8) - 1.0));
    }
    check_below("a scenario holds its pins", pinned, 1e-8);
    check_below("and the conditional spread is s_{T+i} sqrt(1 - r^2) (relative error)", worst, 0.04);
}

template <typename Input>
bool refuses(const Input &input, const std::string &contains)
{
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &e)
    {
        const bool ok = std::string(e.what()).find(contains) != std::string::npos;
        if (!ok)
        {
            std::printf("    refused, but: %s\n", e.what());
        }
        return ok;
    }
    return false;
}

void test_refusals()
{
    std::printf("refusals\n");
    const arma::vec theta = {6.0, 20.0, 9.0, 0.5};
    const Sample s = simulate(40, 31, theta, 3);
    const auto good = input_for(s, 31, 10, 10);
    bool valid = true;
    try
    {
        good.validate();
    }
    catch (const std::exception &e)
    {
        std::printf("    %s\n", e.what());
        valid = false;
    }
    check("a well-formed break validates", valid);

    auto late = good;
    late.spec.volatility_break = 41;
    check("a break past the sample", refuses(late, "inside the sample"));
    auto negative = good;
    negative.spec.volatility_break = -1;
    check("a negative break", refuses(negative, "counted from one"));
    auto none = good;
    none.spec.volatility_scales = 0;
    check("no free scale", refuses(none, "volatility_scales"));
    auto flat = good;
    flat.volatility_prior.shape = 0.0;
    check("a Pareto shape that is not positive", refuses(flat, "/priors/volatility/shape"));
    auto beta = good;
    beta.volatility_prior.rho_shape2 = -1.0;
    check("a beta that is not proper", refuses(beta, "rho_shape2"));
    auto short_start = good;
    short_start.initial.volatility = {2.0, 2.0};
    check("a start of the wrong length", refuses(short_start, "volatility_scales + 1"));
    auto low_start = good;
    low_start.initial.volatility = {2.0, 1.0, 2.0, 0.5};
    check("a start with a scale not above one", refuses(low_start, "above one"));
    auto rho_start = good;
    rho_start.initial.volatility = {2.0, 2.0, 2.0, 1.0};
    check("a start with a decay rate outside (0, 1)", refuses(rho_start, "(0, 1)"));
    auto steady = good;
    steady.spec.steady_state = true;
    steady.mu_prior.mu = arma::zeros<arma::vec>(kK);
    steady.mu_prior.v_inv = arma::eye<arma::mat>(kK, kK);
    steady.initial.mu = arma::zeros<arma::vec>(kK);
    check("beside the steady-state prior", refuses(steady, "steady_state"));
    auto constrained = good;
    constrained.train.constraints.row = {0};
    constrained.train.constraints.period = {0};
    constrained.train.constraints.variable = {0};
    constrained.train.constraints.weight = {1.0};
    constrained.train.constraints.value = {s.y(0, 0)};
    constrained.train.constraints.group = {0};
    check("beside training constraints", refuses(constrained, "volatility_break"));

    bayests::VarNormalGammaInput gamma;
    gamma.spec.k = static_cast<int>(kK);
    gamma.spec.volatility_break = 31;
    check("VarNormalGamma refuses it by name", refuses(gamma, "does not read /model/volatility_break"));
}

} // namespace

int main()
{
    test_path();
    test_truncated_gamma();
    test_conditionals();
    test_likelihood_and_score();
    test_forecast();
    test_recovery();
    test_refusals();
    if (failures > 0)
    {
        std::printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
