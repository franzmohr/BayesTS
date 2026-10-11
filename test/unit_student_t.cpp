// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_student_t.cpp
/// @brief Checks Student t errors in VarNormalWishart.
///
/// The errors are a scale mixture, u_t ~ N(0, lambda_t s_t^2 Sigma) with
/// lambda_t ~ IG(nu / 2, nu / 2). The checks, in order: the multivariate t
/// density; the draw of lambda_t against its inverse gamma; the
/// Metropolis-Hastings step of nu, run with the lambdas, against the marginal
/// posterior of nu by quadrature; the log likelihood and the score against
/// the t density with and without a volatility break; the forecast's
/// variance against nu / (nu - 2) s^2; on a simulated VAR, that the lambdas
/// find the outliers planted in it and that the coefficients are then closer
/// to the truth than under normal errors, and that a break and t errors
/// together still recover the break; and what is refused is refused.

#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_wishart.h"
#include "core/models/completion_support.h"
#include "core/models/student_t_support.h"
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
constexpr arma::uword kNx = kK + 1;

/// k draws of the multivariate t with nu degrees of freedom and scale chol chol'.
arma::vec draw_t(const arma::mat &chol, const double nu)
{
    const double lambda = 1.0 / arma::randg<double>(arma::distr_param(0.5 * nu, 2.0 / nu));
    return std::sqrt(lambda) * chol * arma::randn<arma::vec>(chol.n_rows);
}

void test_density()
{
    std::printf("the multivariate t density\n");
    // k = 1 integrates to one, and a large nu is the normal.
    double total = 0.0;
    const double step = 1e-3;
    for (double x = -400.0; x < 400.0; x += step)
    {
        total += std::exp(bayests::core::student_t_log_density(x * x / 4.0, std::log(4.0), 1.0, 3.0)) * step;
    }
    check_below("a univariate t with scale 2 integrates to one", std::abs(total - 1.0), 2e-3);
    const double q = 1.7, logdet = 0.3;
    const double normal = -0.5 * (3.0 * std::log(2.0 * arma::datum::pi) + logdet + q);
    check_below("with nu = 1e8 it is the normal",
                std::abs(bayests::core::student_t_log_density(q, logdet, 3.0, 1e8) - normal), 1e-6);
}

void test_lambda()
{
    std::printf("the mixing scales, against their inverse gamma\n");
    bayests::VarSpec spec;
    spec.student_t = true;
    bayests::StudentTPrior prior;
    prior.df = 5.0;
    const arma::mat e = {{0.1, 2.0, 6.0}, {0.0, -1.0, 2.0}, {0.3, 0.5, -3.0}};
    const arma::mat precision = arma::eye<arma::mat>(kK, kK);
    bayests::core::StudentTErrors block(spec, prior, 0.0, 3, kK);
    arma::arma_rng::set_seed(31);
    const int n = 60000;
    arma::mat kept(3, n);
    for (int i = 0; i < n; i++)
    {
        block.update(e, precision, false);
        kept.col(i) = block.lambda();
    }
    const arma::rowvec q = arma::sum(e % e, 0);
    double worst = 0.0;
    for (arma::uword t = 0; t < 3; t++)
    {
        const double want = 0.5 * (5.0 + q(t)) / (0.5 * (5.0 + 3.0) - 1.0);
        worst = std::max(worst, std::abs(arma::mean(kept.row(t)) / want - 1.0));
    }
    check_below("each lambda_t has the mean of IG((nu + k) / 2, (nu + q_t) / 2) (rel. error)", worst,
                0.02);
    check("a fixed nu stays fixed", block.df() == 5.0);
}

void test_nu()
{
    std::printf("the degrees of freedom, against their marginal posterior\n");
    const arma::uword tt = 300;
    const arma::mat chol = arma::eye<arma::mat>(kK, kK);
    arma::arma_rng::set_seed(41);
    arma::mat e(kK, tt);
    for (arma::uword t = 0; t < tt; t++)
    {
        e.col(t) = draw_t(chol, 5.0);
    }
    bayests::VarSpec spec;
    spec.student_t = true;
    bayests::StudentTPrior prior;
    prior.shape = 2.0;
    prior.rate = 0.1;
    bayests::core::StudentTErrors block(spec, prior, 0.0, tt, kK);
    const arma::mat precision = arma::eye<arma::mat>(kK, kK);
    const int burn = 3000, n = 40000;
    arma::vec nu(n);
    for (int i = 0; i < burn + n; i++)
    {
        block.update(e, precision, i < burn);
        if (i >= burn)
        {
            nu(i - burn) = block.df();
        }
    }

    // p(nu | e) is the gamma prior times the t density of every residual.
    const arma::rowvec q = arma::sum(e % e, 0);
    double num = 0.0, den = 0.0, lmax = -1e300;
    std::vector<double> logf;
    for (int i = 0; i < 20000; i++)
    {
        const double v = 0.5 + (i + 0.5) * 0.005;
        double lf = (prior.shape - 1.0) * std::log(v) - prior.rate * v;
        for (arma::uword t = 0; t < tt; t++)
        {
            lf += bayests::core::student_t_log_density(q(t), 0.0, static_cast<double>(kK), v);
        }
        logf.push_back(lf);
        lmax = std::max(lmax, lf);
    }
    for (int i = 0; i < 20000; i++)
    {
        const double v = 0.5 + (i + 0.5) * 0.005;
        const double f = std::exp(logf[static_cast<size_t>(i)] - lmax);
        num += v * f;
        den += f;
    }
    const double want = num / den;
    std::printf("    nu: chain %.3f, quadrature %.3f\n", arma::mean(nu), want);
    check_below("the chain's nu has the marginal posterior's mean (relative error)",
                std::abs(arma::mean(nu) / want - 1.0), 0.03);
}

struct Sample
{
    arma::mat y;
    arma::mat z;
    arma::mat truth; ///< [A_1 c], k x n_x
};

/// A VAR(1) with t errors, a volatility break if `theta` is not empty, and
/// shocks multiplied by `boost` in the periods `outliers` lists.
Sample simulate(const arma::uword tt, const double nu, const int break_at, const arma::vec &theta,
                const std::vector<arma::uword> &outliers, const double boost, const int seed)
{
    Sample s;
    const arma::mat a_lag = {{0.6, 0.1, 0.0}, {0.0, 0.5, 0.1}, {0.1, 0.0, 0.4}};
    const arma::vec c = {0.2, -0.1, 0.1};
    s.truth = arma::join_rows(a_lag, c);
    bayests::VarSpec spec;
    spec.volatility_break = theta.is_empty() ? 0 : break_at;
    spec.volatility_scales = 3;
    const arma::vec scale = bayests::core::volatility_path(spec, theta, 0, tt);
    const arma::mat chol = {{0.5, 0.0, 0.0}, {0.2, 0.4, 0.0}, {-0.1, 0.1, 0.3}};
    arma::arma_rng::set_seed(seed);
    arma::mat path(kK, tt + 1, arma::fill::zeros);
    for (arma::uword t = 1; t <= tt; t++)
    {
        double b = 1.0;
        for (const arma::uword o : outliers)
        {
            b = (o == t - 1) ? boost : b;
        }
        path.col(t) = c + a_lag * path.col(t - 1) + b * scale(t - 1) * draw_t(chol, nu);
    }
    s.y = arma::trans(path.cols(1, tt));
    arma::mat x(tt, kNx);
    x.col(kNx - 1).ones();
    x.cols(0, kK - 1) = arma::trans(path.cols(0, tt - 1));
    s.z = arma::kron(x, arma::eye<arma::mat>(kK, kK));
    return s;
}

bayests::VarNormalWishartInput input_for(const Sample &s, const int iterations, const int burnin)
{
    bayests::VarNormalWishartInput input;
    input.spec.k = static_cast<int>(kK);
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.spec.student_t = true;
    input.train.y = s.y;
    input.train.z = s.z;
    const arma::uword n = kK * kNx;
    input.a_prior.mu = arma::zeros<arma::vec>(n);
    input.a_prior.v_inv = 0.01 * arma::eye<arma::mat>(n, n);
    input.u_sigma_prior.df = static_cast<int>(kK) + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    input.initial.a = arma::zeros<arma::vec>(n);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    input.student_t_prior.shape = 2.0;
    input.student_t_prior.rate = 0.1;
    input.volatility_prior.shape = 1.0;
    input.volatility_prior.rho_shape1 = 3.0357;
    input.volatility_prior.rho_shape2 = 1.5089;
    return input;
}

double coefficient_error(const arma::mat &a, const arma::mat &truth)
{
    return arma::abs(arma::reshape(arma::mean(a, 1), kK, kNx) - truth).max();
}

void test_recovery()
{
    std::printf("recovery: outliers found, and the coefficients kept\n");
    bayests::NullReporter reporter;
    const std::vector<arma::uword> outliers = {80, 160, 240};
    const Sample s = simulate(300, 6.0, 0, arma::vec(), outliers, 15.0, 808);
    const auto input = input_for(s, 3000, 1500);
    arma::arma_rng::set_seed(1);
    const auto draws = bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);
    check("one df and tt lambdas per kept draw",
          draws.student_t_df.n_cols == 3000 && draws.student_t_lambda.n_rows == 300);
    const arma::vec lambda = arma::mean(draws.student_t_lambda, 1);
    const arma::uvec top = arma::sort_index(lambda, "descend");
    bool found = true;
    for (arma::uword i = 0; i < 3; i++)
    {
        found = found && (top(i) == 80 || top(i) == 160 || top(i) == 240);
    }
    std::printf("    largest mean lambdas at periods %llu %llu %llu: %.1f %.1f %.1f, median %.2f\n",
                static_cast<unsigned long long>(top(0)), static_cast<unsigned long long>(top(1)),
                static_cast<unsigned long long>(top(2)), lambda(top(0)), lambda(top(1)),
                lambda(top(2)), arma::median(lambda));
    check("the three planted outliers have the three largest lambdas", found);

    auto normal = input;
    normal.spec.student_t = false;
    arma::arma_rng::set_seed(1);
    const auto gauss = bayests::VarNormalWishartSampler{}.draw_coefficients(normal, reporter);
    const double err_t = coefficient_error(draws.a, s.truth);
    const double err_n = coefficient_error(gauss.a, s.truth);
    std::printf("    worst coefficient error: %.3f with t errors, %.3f with normal errors\n", err_t,
                err_n);
    check("t errors keep the coefficients closer to the truth", err_t < err_n);

    std::printf("a volatility break and t errors together\n");
    const arma::vec theta = {12.0, 30.0, 8.0, 0.7};
    const Sample b = simulate(300, 6.0, 281, theta, {}, 1.0, 909);
    auto both = input_for(b, 3000, 1500);
    both.spec.volatility_break = 281;
    both.spec.volatility_scales = 3;
    arma::arma_rng::set_seed(2);
    const auto joint = bayests::VarNormalWishartSampler{}.draw_coefficients(both, reporter);
    const arma::vec lo = arma::quantile(joint.volatility, arma::vec({0.005}), 1);
    const arma::vec hi = arma::quantile(joint.volatility, arma::vec({0.995}), 1);
    const arma::vec med = arma::median(joint.volatility, 1);
    const arma::vec nu_q = arma::quantile(arma::vectorise(joint.student_t_df), arma::vec({0.005, 0.5, 0.995}));
    std::printf("    break medians %.2f %.2f %.2f %.3f, truth 12 30 8 0.7; nu median %.2f [%.2f, %.2f], "
                "truth 6\n",
                med(0), med(1), med(2), med(3), nu_q(1), nu_q(0), nu_q(2));
    check("the break's 99% intervals still cover the truth",
          arma::all(lo <= theta) && arma::all(hi >= theta));
    check("and nu's covers 6", nu_q(0) <= 6.0 && nu_q(2) >= 6.0);
}

bayests::VarNormalWishartDraws fixed_draws(const arma::mat &a, const arma::mat &sigma, const double nu,
                                           const arma::vec &theta, const arma::uword n)
{
    bayests::VarNormalWishartDraws d;
    d.a = arma::repmat(arma::vectorise(a), 1, n);
    d.u_sigma_inv = arma::repmat(arma::vectorise(arma::inv_sympd(sigma)), 1, n);
    d.student_t_df = arma::mat(1, n, arma::fill::value(nu));
    if (!theta.is_empty())
    {
        d.volatility = arma::repmat(theta, 1, n);
    }
    return d;
}

void test_likelihood_and_forecast()
{
    std::printf("the log likelihood and the score, against the t density\n");
    const arma::uword tt = 40;
    const Sample s = simulate(tt + 4, 5.0, 31, arma::vec({6.0, 14.0, 5.0, 0.6}), {}, 1.0, 55);
    const arma::mat sigma = {{0.25, 0.08, -0.05}, {0.08, 0.2, 0.04}, {-0.05, 0.04, 0.14}};
    for (const bool with_break : {false, true})
    {
        const arma::vec theta = with_break ? arma::vec({6.0, 14.0, 5.0, 0.6}) : arma::vec();
        Sample train = s;
        train.y = s.y.head_rows(tt);
        train.z = s.z.head_rows(tt * kK);
        auto input = input_for(train, 2, 0);
        input.spec.volatility_break = with_break ? 31 : 0;
        const auto draws = fixed_draws(s.truth, sigma, 4.5, theta, 2);
        const arma::mat ll = bayests::VarNormalWishartSampler{}.log_likelihood(input, draws);
        bayests::VarSpec spec = input.spec;
        const arma::vec path = bayests::core::volatility_path(spec, theta, 0, tt + 4);
        const auto dense = [&](const arma::uword t) {
            const arma::vec lag =
                t == 0 ? arma::vec(kK, arma::fill::zeros) : arma::vec(arma::trans(s.y.row(t - 1)));
            const arma::vec e = arma::trans(s.y.row(t)) - s.truth * arma::join_cols(lag, arma::vec({1.0}));
            const arma::mat cov = path(t) * path(t) * sigma;
            double logdet, sign;
            arma::log_det(logdet, sign, cov);
            const double q = arma::as_scalar(arma::trans(e) * arma::solve(cov, e));
            // The t density written out again, from its textbook form.
            return std::lgamma(0.5 * (4.5 + kK)) - std::lgamma(0.5 * 4.5) -
                   0.5 * kK * std::log(4.5 * arma::datum::pi) - 0.5 * logdet -
                   0.5 * (4.5 + kK) * std::log(1.0 + q / 4.5);
        };
        double worst = 0.0;
        for (arma::uword t = 0; t < tt; t++)
        {
            worst = std::max(worst, std::abs(ll(0, t) - dense(t)));
        }
        const std::string tag = with_break ? " with a break" : "";
        check_below("every period's log density is the multivariate t" + tag, worst, 1e-10);

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
        check_below("and so is the score" + tag, worst, 1e-10);
    }

    std::printf("the forecast's variance\n");
    const arma::uword n = 60000;
    const double nu = 8.0;
    const arma::vec theta = {6.0, 20.0, 9.0, 0.5};
    Sample train = simulate(30, 5.0, 29, theta, {}, 1.0, 66);
    auto input = input_for(train, 2, 0);
    input.spec.volatility_break = 29;
    input.spec.h = 3;
    input.forecast.x = arma::mat(3, kNx, arma::fill::zeros);
    input.forecast.x.col(kNx - 1).ones();
    const auto draws = fixed_draws(arma::zeros<arma::mat>(kK, kNx), arma::eye<arma::mat>(kK, kK), nu,
                                   theta, n);
    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(7);
    const arma::mat f = bayests::VarNormalWishartSampler{}.forecast(input, draws, reporter).values;
    bayests::VarSpec spec = input.spec;
    const arma::vec s_h = bayests::core::volatility_path(spec, theta, 30, 3);
    double worst = 0.0;
    for (arma::uword i = 0; i < 3; i++)
    {
        const double want = s_h(i) * s_h(i) * nu / (nu - 2.0);
        worst = std::max(worst, std::abs(arma::var(f.row(i * kK)) / want - 1.0));
    }
    check_below("each horizon's variance is s_{T+i}^2 nu / (nu - 2) (relative error)", worst, 0.05);
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
    const Sample s = simulate(40, 5.0, 0, arma::vec(), {}, 1.0, 77);
    const auto good = input_for(s, 10, 10);
    bool valid = true;
    try
    {
        good.validate();
        auto fixed = good;
        fixed.student_t_prior = bayests::StudentTPrior{};
        fixed.student_t_prior.df = 5.0;
        fixed.validate();
    }
    catch (const std::exception &e)
    {
        std::printf("    %s\n", e.what());
        valid = false;
    }
    check("a gamma prior on nu, and a fixed nu, both validate", valid);

    auto both = good;
    both.student_t_prior.df = 5.0;
    check("both a fixed nu and a prior on it", refuses(both, "exactly one"));
    auto neither = good;
    neither.student_t_prior = bayests::StudentTPrior{};
    check("neither", refuses(neither, "exactly one"));
    auto negative = good;
    negative.student_t_prior.rate = -0.1;
    check("a rate that is not positive", refuses(negative, "/priors/student_t/rate"));
    auto start = good;
    start.initial.student_t_df = -1.0;
    check("a start that is not positive", refuses(start, "/initial/student_t_df"));
    auto steady = good;
    steady.spec.steady_state = true;
    steady.mu_prior.mu = arma::zeros<arma::vec>(kK);
    steady.mu_prior.v_inv = arma::eye<arma::mat>(kK, kK);
    steady.initial.mu = arma::zeros<arma::vec>(kK);
    check("beside the steady-state prior", refuses(steady, "steady_state"));
    auto scenario = good;
    scenario.spec.h = 2;
    scenario.forecast.x = arma::mat(2, kNx, arma::fill::ones);
    scenario.forecast.constraints.value = {0.0};
    scenario.forecast.constraints.group = {0};
    scenario.forecast.constraints.row = {0};
    scenario.forecast.constraints.period = {0};
    scenario.forecast.constraints.variable = {0};
    scenario.forecast.constraints.weight = {1.0};
    check("beside a scenario", refuses(scenario, "/data/forecast/constraints"));

    auto partial = good;
    partial.spec.h = 2;
    partial.forecast.x = arma::mat(2, kNx, arma::fill::ones);
    partial.test.y = arma::mat(2, kK, arma::fill::zeros);
    partial.test.constraints.value = {0.0};
    partial.test.constraints.group = {0};
    partial.test.constraints.row = {0};
    partial.test.constraints.period = {0};
    partial.test.constraints.variable = {0};
    partial.test.constraints.weight = {1.0};
    check("a score against a horizon realised in part", refuses(partial, "realised in part"));
    auto scored = good;
    scored.spec.h = 2;
    scored.forecast.x = arma::mat(2, kNx, arma::fill::ones);
    scored.test.y = arma::mat(2, kK, arma::fill::zeros);
    scored.train.constraints = bayests::core::pin_every_entry(s.y);
    check("a score from a panel not observed whole", refuses(scored, "not observed whole"));
    auto training = good;
    training.train.constraints = bayests::core::pin_every_entry(s.y);
    bool reads = true;
    try
    {
        training.validate();
    }
    catch (const std::exception &e)
    {
        std::printf("    %s\n", e.what());
        reads = false;
    }
    check("training constraints alone validate: the panel is completed", reads);

    bayests::VarNormalGammaInput gamma;
    gamma.spec.k = static_cast<int>(kK);
    gamma.spec.student_t = true;
    check("VarNormalGamma refuses it by name", refuses(gamma, "does not read /model/student_t"));
}

} // namespace

int main()
{
    test_density();
    test_lambda();
    test_nu();
    test_likelihood_and_forecast();
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
