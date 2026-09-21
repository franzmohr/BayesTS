// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_noncentred.cpp
/// @brief The non-centred random walk and the Savage-Dickey ordinates it writes.
///
/// Two kinds of check. The first three are about the conditional draw of
/// \f$(x_0, \omega)\f$ in src/core/models/noncentred_support.h: that the log
/// ordinates it hands back are the densities at zero of the marginals of the
/// normal it draws from -- an identity, checked to rounding -- and that the
/// draws have that normal's mean, a statistical statement made at five Monte
/// Carlo standard errors.
///
/// The last runs VarTvpStochvol end to end on three simulated samples and asks
/// whether the Bayes factors of Chan (2018) come out on the side the data were
/// generated on: against time variation where nothing moves, for it in the
/// volatility of the one equation whose variance jumps, and for it in the one
/// intercept that shifts. "Found to move" asks for a log Bayes factor above 3
/// and "not found to move" for one below 1 -- no more than a bare mention's
/// worth of evidence either way. Over eight seeds the smallest of the former was
/// 4.7 and the largest of the latter -0.1.
///
/// What is deliberately not asked is that the whole coefficient block be found
/// to move when one intercept does. The joint ordinate tests "every coefficient
/// moves" against "none does", and five that stand still cost about a log point
/// each: over the same eight seeds that Bayes factor ranged from -2.7 to 13.1.
/// That is the test doing what it says, and why the per-state ordinates are
/// written beside the joint one.
///
/// VarTvpGamma is run on the shifting intercept as well, with a covariance
/// block on, which puts its single-block path through the helper: the errors
/// are independent, so no covariance coefficient should be found to move. Over
/// eight seeds the shifting intercept's log Bayes factor ranged from 5.1 to 14.9
/// and the largest of the others reached 0.84, so "not found to move" asks for
/// below 2 here -- still short of the 3 that "found to move" needs.
///
/// VecTvpStochvol is run on a cointegrated pair whose first error variance
/// jumps, with every block non-centred: the jump has to be found in that
/// volatility and not in the other, while the cointegration space keeps its
/// fixed state variance. VecTvpGamma is run on the same pair with unit error
/// variances and a constant in the first equation that shifts half way -- by
/// twice as much as the VAR's, for the reason given at the check.
///
/// Last, what validate() refuses: a file that gives both priors on how far a
/// random walk moves, and an omega_v it could not use.

#include "bayests/reporter.h"
#include "bayests/var_tvp_gamma.h"
#include "bayests/var_tvp_stochvol.h"
#include "bayests/vec_tvp_gamma.h"
#include "bayests/vec_tvp_stochvol.h"
#include "core/models/noncentred_support.h"

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
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

bool close(double a, double b, double tol = 1e-9)
{
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

/// A random symmetric positive definite matrix, well away from singular.
arma::mat random_spd(arma::uword n)
{
    const arma::mat a = arma::randn<arma::mat>(n, n);
    return a.t() * a + arma::eye<arma::mat>(n, n);
}

void the_ordinates_are_the_marginal_densities()
{
    std::printf("the ordinates at zero\n");

    const arma::uword n = 3;
    const arma::mat data_precision = random_spd(2 * n);
    const arma::vec data_rhs = arma::randn<arma::vec>(2 * n);

    bayests::NormalPrior x0_prior;
    x0_prior.mu = arma::randn<arma::vec>(n);
    x0_prior.v_inv = random_spd(n);
    const arma::vec omega_v = {0.1, 0.5, 2.0};

    const bayests::core::NoncentredCoefficients c =
        bayests::core::draw_noncentred_coefficients(data_precision, data_rhs, x0_prior, omega_v);

    // The same normal assembled the long way round.
    arma::mat precision = data_precision;
    precision.submat(0, 0, n - 1, n - 1) += x0_prior.v_inv;
    precision.submat(n, n, 2 * n - 1, 2 * n - 1) += arma::diagmat(1.0 / omega_v);
    arma::vec rhs = data_rhs;
    rhs.head(n) += x0_prior.v_inv * x0_prior.mu;
    const arma::mat cov = arma::inv(precision);
    const arma::vec mean = cov * rhs;

    bool per_state = true;
    for (arma::uword i = 0; i < n; i++)
    {
        const double sd = std::sqrt(cov(n + i, n + i));
        const double expected = arma::log_normpdf(0.0, mean(n + i), sd);
        per_state = per_state && close(c.log_zero(i), expected);
    }
    check("each state's is its marginal density at zero", per_state);

    const arma::mat omega_cov = cov.submat(n, n, 2 * n - 1, 2 * n - 1);
    const arma::vec omega_mean = mean.tail(n);
    const double expected_joint =
        -0.5 * (n * std::log(2 * arma::datum::pi) + std::log(arma::det(omega_cov)) +
                arma::as_scalar(omega_mean.t() * arma::inv(omega_cov) * omega_mean));
    check("the joint one is the block's density at zero", close(c.log_zero_joint, expected_joint));
}

void the_joint_ordinate_is_the_sum_for_independent_states()
{
    std::printf("independent states\n");

    // (x0_i, omega_i) pairs that nothing couples: the log-volatility's layout,
    // with a diagonal prior on where each starts.
    const arma::uword n = 4;
    arma::mat data_precision = arma::zeros<arma::mat>(2 * n, 2 * n);
    for (arma::uword i = 0; i < n; i++)
    {
        const arma::mat block = random_spd(2);
        data_precision(i, i) = block(0, 0);
        data_precision(i, n + i) = block(0, 1);
        data_precision(n + i, i) = block(1, 0);
        data_precision(n + i, n + i) = block(1, 1);
    }

    bayests::NormalPrior x0_prior;
    x0_prior.mu = arma::randn<arma::vec>(n);
    x0_prior.v_inv = arma::diagmat(arma::vec(n, arma::fill::value(0.5)));

    const bayests::core::NoncentredCoefficients c = bayests::core::draw_noncentred_coefficients(
        data_precision, arma::randn<arma::vec>(2 * n), x0_prior, arma::vec(n, arma::fill::value(0.2)));

    check("the joint ordinate is the sum of the per-state ones",
          close(c.log_zero_joint, arma::accu(c.log_zero)));
}

void the_draw_has_the_conditional_mean()
{
    std::printf("the draw of (x0, omega)\n");

    const arma::uword n = 2;
    const arma::mat data_precision = random_spd(2 * n);
    const arma::vec data_rhs = arma::randn<arma::vec>(2 * n);
    bayests::NormalPrior x0_prior;
    x0_prior.mu = arma::zeros<arma::vec>(n);
    x0_prior.v_inv = arma::eye<arma::mat>(n, n);
    const arma::vec omega_v(n, arma::fill::value(0.3));

    arma::mat precision = data_precision;
    precision.submat(0, 0, n - 1, n - 1) += x0_prior.v_inv;
    precision.submat(n, n, 2 * n - 1, 2 * n - 1) += arma::diagmat(1.0 / omega_v);
    const arma::mat cov = arma::inv(precision);
    const arma::vec mean = cov * data_rhs;

    const int draws = 20000;
    arma::mat sample(2 * n, draws);
    for (int d = 0; d < draws; d++)
    {
        const bayests::core::NoncentredCoefficients c = bayests::core::draw_noncentred_coefficients(
            data_precision, data_rhs, x0_prior, omega_v);
        sample.col(d) = arma::join_cols(c.x0, c.omega);
    }

    const arma::vec se = arma::sqrt(cov.diag() / draws);
    const arma::vec gap = arma::abs(arma::mean(sample, 1) - mean) / se;
    check("every coefficient's mean within five Monte Carlo standard errors", gap.max() < 5.0);
}

/// log of the average of exp(x), without the underflow the ordinates invite.
double log_mean_exp(const arma::rowvec &x)
{
    const double m = x.max();
    return m + std::log(arma::mean(arma::exp(x - m)));
}

/// log BF of the time-varying model against the constant one, per the
/// Savage-Dickey ratio: prior ordinate over posterior ordinate, one state.
double log_bf(const bayests::NoncentredStateDraws &d, arma::uword i, double omega_v)
{
    const double prior = -0.5 * std::log(2 * arma::datum::pi * omega_v);
    return prior - log_mean_exp(d.log_zero.row(i));
}

/// The same for the whole block, from the joint ordinate.
double log_bf_joint(const bayests::NoncentredStateDraws &d, double omega_v)
{
    const double n = static_cast<double>(d.log_zero.n_rows);
    const double prior = -0.5 * n * std::log(2 * arma::datum::pi * omega_v);
    return prior - log_mean_exp(d.log_zero_joint.row(0));
}

/// Two-variable VAR(1) with an intercept. `sd` is T x 2, the error standard
/// deviation of each equation per period; `shift` is added to the first
/// equation's intercept from the middle of the sample on.
bayests::VarTvpStochvolInput simulate(const arma::mat &sd, double shift)
{
    const arma::uword k = 2;
    const arma::uword tt = sd.n_rows;
    const arma::mat a1 = {{0.5, 0.1}, {0.0, 0.4}};

    arma::mat y(tt + 1, k, arma::fill::zeros);
    for (arma::uword t = 1; t <= tt; t++)
    {
        arma::vec c = arma::zeros<arma::vec>(k);
        if (t > tt / 2)
        {
            c(0) = shift;
        }
        y.row(t) = arma::trans(c + a1 * arma::trans(y.row(t - 1)) +
                               arma::trans(sd.row(t - 1)) % arma::randn<arma::vec>(k));
    }

    // z_t = x_t' kron I_k with x_t = (y_{t-1}', 1): the SUR layout.
    const arma::uword nparams = 3 * k;
    arma::mat z(k * tt, nparams);
    for (arma::uword t = 0; t < tt; t++)
    {
        const arma::rowvec x = arma::join_rows(y.row(t), arma::rowvec{1.0});
        z.rows(k * t, k * (t + 1) - 1) = arma::kron(x, arma::eye<arma::mat>(k, k));
    }

    bayests::VarTvpStochvolInput input;
    input.spec.k = static_cast<int>(k);
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.iterations = 3000;
    input.spec.burnin = 1000;

    input.train.y = y.rows(1, tt);
    input.train.z = z;

    input.a_prior.omega_v = arma::vec(nparams, arma::fill::value(0.01));
    input.a_prior.initial_state.mu = arma::zeros<arma::vec>(nparams);
    input.a_prior.initial_state.v_inv = arma::eye<arma::mat>(nparams, nparams) * 0.1;

    input.u_sigma_prior.offset = arma::vec(k, arma::fill::value(1e-4));
    input.u_sigma_prior.state.omega_v = arma::vec(k, arma::fill::value(0.1));
    input.u_sigma_prior.state.initial_state.mu = arma::zeros<arma::vec>(k);
    input.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(k, k) * 0.1;

    input.initial.a = arma::zeros<arma::mat>(nparams, tt);
    input.initial.a_sigma_inv = arma::eye<arma::mat>(nparams, nparams) * 100.0;
    input.initial.a_init = arma::zeros<arma::vec>(nparams);
    input.initial.h = arma::zeros<arma::mat>(tt, k);
    input.initial.h_init = arma::zeros<arma::vec>(k);
    input.initial.h_sigma = arma::vec(k, arma::fill::value(0.1));

    return input;
}

bool run(const bayests::VarTvpStochvolInput &input, bayests::VarTvpStochvolDraws &draws)
{
    bayests::NullReporter reporter;
    try
    {
        draws = bayests::VarTvpStochvolSampler{}.draw_coefficients(input, reporter);
        return true;
    }
    catch (const std::exception &e)
    {
        std::printf("    threw: %s\n", e.what());
        return false;
    }
}

void the_bayes_factors_point_the_right_way()
{
    const arma::uword tt = 200;

    {
        std::printf("nothing moves\n");
        bayests::VarTvpStochvolDraws d;
        const bool ran = run(simulate(arma::ones<arma::mat>(tt, 2), 0.0), d);
        check("the chain runs to the end", ran);
        if (ran)
        {
            const double h1 = log_bf(d.h_noncentred, 0, 0.1);
            const double h2 = log_bf(d.h_noncentred, 1, 0.1);
            const double a = log_bf_joint(d.a_noncentred, 0.01);
            std::printf("    log BF: volatility %.2f, %.2f; coefficients jointly %.2f\n", h1, h2, a);
            check("neither volatility is found to move", h1 < 1.0 && h2 < 1.0);
            check("the coefficients are not found to move", a < 0.0);
            check("sigma is written as omega squared",
                  arma::approx_equal(d.h_sigma, arma::square(d.h_noncentred.omega), "absdiff",
                                     1e-12));
            check("omega visits both signs", d.h_noncentred.omega.min() < 0.0 &&
                                                 d.h_noncentred.omega.max() > 0.0);
        }
    }

    {
        std::printf("the first equation's variance jumps\n");
        arma::mat sd = arma::ones<arma::mat>(tt, 2);
        sd.submat(0, 0, tt / 2 - 1, 0).fill(0.5);
        sd.submat(tt / 2, 0, tt - 1, 0).fill(3.0);
        bayests::VarTvpStochvolDraws d;
        const bool ran = run(simulate(sd, 0.0), d);
        check("the chain runs to the end", ran);
        if (ran)
        {
            const double h1 = log_bf(d.h_noncentred, 0, 0.1);
            const double h2 = log_bf(d.h_noncentred, 1, 0.1);
            std::printf("    log BF: volatility %.2f, %.2f\n", h1, h2);
            check("the first volatility is found to move", h1 > 3.0);
            check("the second is not", h2 < 1.0);
        }
    }

    {
        std::printf("the first equation's intercept shifts\n");
        bayests::VarTvpStochvolDraws d;
        const bool ran = run(simulate(arma::ones<arma::mat>(tt, 2), 4.0), d);
        check("the chain runs to the end", ran);
        if (ran)
        {
            // Coefficients are ordered vec(A) with x_t = (y_{t-1}', 1), so the
            // intercepts come last: the first equation's is at 2k = 4.
            const double c1 = log_bf(d.a_noncentred, 4, 0.01);
            const double c2 = log_bf(d.a_noncentred, 5, 0.01);
            std::printf("    log BF: intercepts %.2f, %.2f\n", c1, c2);
            check("the shifting intercept is found to move", c1 > 3.0);
            check("the other is not", c2 < 1.0);
        }
    }
}

/// The same sample and coefficient prior as simulate(), for VarTvpGamma: a
/// constant error precision with a gamma prior, and a covariance block.
bayests::VarTvpGammaInput as_gamma(const bayests::VarTvpStochvolInput &sv)
{
    const arma::uword k = static_cast<arma::uword>(sv.spec.k);
    const arma::uword tt = sv.train.y.n_rows;

    bayests::VarTvpGammaInput input;
    input.spec = sv.spec;
    input.spec.covar = true;
    input.train = sv.train;
    input.a_prior = sv.a_prior;
    input.initial.a = sv.initial.a;
    input.initial.a_sigma_inv = sv.initial.a_sigma_inv;
    input.initial.a_init = sv.initial.a_init;

    const arma::uword n_psi = k * (k - 1) / 2;
    input.psi_prior.omega_v = arma::vec(n_psi, arma::fill::value(0.01));
    input.psi_prior.initial_state.mu = arma::zeros<arma::vec>(n_psi);
    input.psi_prior.initial_state.v_inv = arma::eye<arma::mat>(n_psi, n_psi);
    input.initial.psi = arma::zeros<arma::mat>(n_psi, tt);
    input.initial.psi_sigma_inv = arma::eye<arma::mat>(n_psi, n_psi) * 100.0;
    input.initial.psi_init = arma::zeros<arma::vec>(n_psi);

    input.u_sigma_prior.shape = arma::vec(k, arma::fill::value(3.0));
    input.u_sigma_prior.rate = arma::vec(k, arma::fill::value(2.0));
    input.initial.u_omega_inv = arma::eye<arma::mat>(k, k);

    return input;
}

void the_gamma_model_finds_the_shifting_intercept()
{
    std::printf("VarTvpGamma: the first equation's intercept shifts\n");

    const bayests::VarTvpGammaInput input =
        as_gamma(simulate(arma::ones<arma::mat>(200, 2), 4.0));

    bayests::NullReporter reporter;
    bayests::VarTvpGammaDraws d;
    bool ran = false;
    try
    {
        d = bayests::VarTvpGammaSampler{}.draw_coefficients(input, reporter);
        ran = true;
    }
    catch (const std::exception &e)
    {
        std::printf("    threw: %s\n", e.what());
    }
    check("the chain runs to the end", ran);
    if (!ran)
    {
        return;
    }

    const double c1 = log_bf(d.a_noncentred, 4, 0.01);
    const double c2 = log_bf(d.a_noncentred, 5, 0.01);
    const double p1 = log_bf(d.psi_noncentred, 0, 0.01);
    std::printf("    log BF: intercepts %.2f, %.2f; covariance %.2f\n", c1, c2, p1);
    check("the shifting intercept is found to move", c1 > 3.0);
    check("the other is not", c2 < 2.0);
    check("the covariance coefficient is not", p1 < 2.0);
    check("psi sigma is written as omega squared",
          arma::approx_equal(d.psi_sigma, arma::square(d.psi_noncentred.omega), "absdiff",
                             1e-12));
}

/// A cointegrated pair, dy_t = alpha beta' y_{t-1} + c + u_t with beta = (1, -1)',
/// whose first error standard deviation jumps from 0.5 to 3 half way through.
bayests::VecTvpStochvolInput simulate_vec()
{
    const arma::uword k = 2;
    const arma::uword tt = 200;
    const arma::vec alpha = {-0.3, 0.1};
    const arma::vec beta = {1.0, -1.0};

    arma::mat levels(tt + 1, k, arma::fill::zeros);
    for (arma::uword t = 1; t <= tt; t++)
    {
        const arma::vec sd = {t > tt / 2 ? 3.0 : 0.5, 1.0};
        const arma::vec prev = arma::trans(levels.row(t - 1));
        levels.row(t) = arma::trans(prev + alpha * arma::dot(beta, prev) +
                                    sd % arma::randn<arma::vec>(k));
    }

    // Loadings first, then the unrestricted constant: n_a = k rank + k.
    const arma::uword n_a = 2 * k;
    arma::mat z(k * tt, n_a, arma::fill::zeros);
    for (arma::uword t = 0; t < tt; t++)
    {
        z.submat(k * t, k, k * (t + 1) - 1, n_a - 1) = arma::eye<arma::mat>(k, k);
    }

    bayests::VecTvpStochvolInput input;
    input.spec.k = static_cast<int>(k);
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.rank = 1;
    input.spec.k_beta = static_cast<int>(k);
    input.spec.iterations = 3000;
    input.spec.burnin = 1000;

    input.train.y = arma::diff(levels);
    input.train.w = levels.rows(0, tt - 1);
    input.train.z = z;

    input.a_prior.omega_v = arma::vec(n_a, arma::fill::value(0.001));
    input.a_prior.initial_state.mu = arma::zeros<arma::vec>(n_a);
    input.a_prior.initial_state.v_inv = arma::eye<arma::mat>(n_a, n_a) * 0.1;

    const double rho = input.beta_prior.rho;
    input.beta_prior.initial_state.mu = arma::zeros<arma::vec>(k);
    input.beta_prior.initial_state.v_inv = (1.0 - rho * rho) * arma::eye<arma::mat>(k, k);

    input.u_sigma_prior.offset = arma::vec(k, arma::fill::value(1e-4));
    input.u_sigma_prior.state.omega_v = arma::vec(k, arma::fill::value(0.1));
    input.u_sigma_prior.state.initial_state.mu = arma::zeros<arma::vec>(k);
    input.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(k, k) * 0.1;

    input.initial.a = arma::zeros<arma::mat>(n_a, tt);
    input.initial.a_sigma_inv = arma::eye<arma::mat>(n_a, n_a) * 1000.0;
    input.initial.a_init = arma::zeros<arma::vec>(n_a);
    input.initial.beta = arma::repmat(beta, 1, tt);
    input.initial.beta_init = beta;
    input.initial.h = arma::zeros<arma::mat>(tt, k);
    input.initial.h_init = arma::zeros<arma::vec>(k);
    input.initial.h_sigma = arma::vec(k, arma::fill::value(0.1));

    return input;
}

void the_vec_finds_the_volatility_that_jumps()
{
    std::printf("VecTvpStochvol: the first equation's variance jumps\n");

    bayests::NullReporter reporter;
    bayests::VecTvpStochvolDraws d;
    bool ran = false;
    try
    {
        d = bayests::VecTvpStochvolSampler{}.draw_coefficients(simulate_vec(), reporter);
        ran = true;
    }
    catch (const std::exception &e)
    {
        std::printf("    threw: %s\n", e.what());
    }
    check("the chain runs to the end", ran);
    if (!ran)
    {
        return;
    }

    const double h1 = log_bf(d.h_noncentred, 0, 0.1);
    const double h2 = log_bf(d.h_noncentred, 1, 0.1);
    std::printf("    log BF: volatility %.2f, %.2f\n", h1, h2);
    check("the first volatility is found to move", h1 > 3.0);
    check("the second is not", h2 < 2.0);
    check("every coefficient has its ordinates", d.a_noncentred.log_zero.n_rows == 4 &&
                                                    d.a_noncentred.log_zero.is_finite());
    check("a sigma is written as omega squared",
          arma::approx_equal(d.a_sigma, arma::square(d.a_noncentred.omega), "absdiff", 1e-12));
}

/// The cointegrated pair of simulate_vec() with unit error variances and the
/// first equation's constant shifting by `shift` half way, for VecTvpGamma with
/// a covariance block.
bayests::VecTvpGammaInput simulate_vec_gamma(double shift)
{
    const arma::uword k = 2;
    const arma::uword tt = 200;
    const arma::vec alpha = {-0.3, 0.1};
    const arma::vec beta = {1.0, -1.0};

    arma::mat levels(tt + 1, k, arma::fill::zeros);
    for (arma::uword t = 1; t <= tt; t++)
    {
        const arma::vec c = {t > tt / 2 ? shift : 0.0, 0.0};
        const arma::vec prev = arma::trans(levels.row(t - 1));
        levels.row(t) = arma::trans(prev + alpha * arma::dot(beta, prev) + c +
                                    arma::randn<arma::vec>(k));
    }

    const arma::uword n_a = 2 * k;
    arma::mat z(k * tt, n_a, arma::fill::zeros);
    for (arma::uword t = 0; t < tt; t++)
    {
        z.submat(k * t, k, k * (t + 1) - 1, n_a - 1) = arma::eye<arma::mat>(k, k);
    }

    bayests::VecTvpGammaInput input;
    input.spec.k = static_cast<int>(k);
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.rank = 1;
    input.spec.k_beta = static_cast<int>(k);
    input.spec.covar = true;
    input.spec.iterations = 3000;
    input.spec.burnin = 1000;

    input.train.y = arma::diff(levels);
    input.train.w = levels.rows(0, tt - 1);
    input.train.z = z;

    input.a_prior.omega_v = arma::vec(n_a, arma::fill::value(0.01));
    input.a_prior.initial_state.mu = arma::zeros<arma::vec>(n_a);
    input.a_prior.initial_state.v_inv = arma::eye<arma::mat>(n_a, n_a) * 0.1;

    const double rho = input.beta_prior.rho;
    input.beta_prior.initial_state.mu = arma::zeros<arma::vec>(k);
    input.beta_prior.initial_state.v_inv = (1.0 - rho * rho) * arma::eye<arma::mat>(k, k);

    input.psi_prior.omega_v = arma::vec(1, arma::fill::value(0.01));
    input.psi_prior.initial_state.mu = arma::zeros<arma::vec>(1);
    input.psi_prior.initial_state.v_inv = arma::eye<arma::mat>(1, 1);

    input.u_sigma_prior.shape = arma::vec(k, arma::fill::value(3.0));
    input.u_sigma_prior.rate = arma::vec(k, arma::fill::value(2.0));

    input.initial.a = arma::zeros<arma::mat>(n_a, tt);
    input.initial.a_sigma_inv = arma::eye<arma::mat>(n_a, n_a) * 100.0;
    input.initial.a_init = arma::zeros<arma::vec>(n_a);
    input.initial.beta = arma::repmat(beta, 1, tt);
    input.initial.beta_init = beta;
    input.initial.psi = arma::zeros<arma::mat>(1, tt);
    input.initial.psi_sigma_inv = arma::eye<arma::mat>(1, 1) * 100.0;
    input.initial.psi_init = arma::zeros<arma::vec>(1);
    input.initial.u_omega_inv = arma::eye<arma::mat>(k, k);

    return input;
}

void the_vec_gamma_model_finds_the_shifting_constant()
{
    std::printf("VecTvpGamma: the first equation's constant shifts\n");

    bayests::NullReporter reporter;
    bayests::VecTvpGammaDraws d;
    bool ran = false;
    try
    {
        d = bayests::VecTvpGammaSampler{}.draw_coefficients(simulate_vec_gamma(4.0), reporter);
        ran = true;
    }
    catch (const std::exception &e)
    {
        std::printf("    threw: %s\n", e.what());
    }
    check("the chain runs to the end", ran);
    if (!ran)
    {
        return;
    }

    // Loadings first, then the constants: the first equation's is at k = 2.
    //
    // The shift is twice the VAR's. A constant that moves shifts the
    // equilibrium of the error correction term with it, and a cointegration
    // space that follows a random walk of unit variance can take part of that
    // up: at a shift of 2 the constant's log Bayes factor was 0.96, at 4 it is
    // 6.7 and at 6 it is 9.7, the other constant staying below 0.4 throughout.
    const double c1 = log_bf(d.a_noncentred, 2, 0.01);
    const double c2 = log_bf(d.a_noncentred, 3, 0.01);
    const double p1 = log_bf(d.psi_noncentred, 0, 0.01);
    std::printf("    log BF: constants %.2f, %.2f; covariance %.2f\n", c1, c2, p1);
    check("the shifting constant is found to move", c1 > 3.0);
    check("the other is not", c2 < 2.0);
    check("the covariance coefficient is not", p1 < 2.0);
    check("psi sigma is written as omega squared",
          arma::approx_equal(d.psi_sigma, arma::square(d.psi_noncentred.omega), "absdiff",
                             1e-12));
}

/// Whether validate() refuses `input`, and with a message that mentions `word`.
bool refused(const bayests::VarTvpStochvolInput &input, const std::string &word)
{
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &e)
    {
        return std::string(e.what()).find(word) != std::string::npos;
    }
    return false;
}

void the_prior_is_one_or_the_other()
{
    std::printf("what validate() refuses\n");

    const bayests::VarTvpStochvolInput good = simulate(arma::ones<arma::mat>(20, 2), 0.0);
    check("a non-centred file is accepted", !refused(good, ""));

    bayests::VarTvpStochvolInput both = good;
    both.a_prior.sigma.shape = arma::vec(6, arma::fill::value(3.0));
    both.a_prior.sigma.rate = arma::vec(6, arma::fill::value(0.01));
    check("omega_v beside shape and rate is refused", refused(both, "omega_v"));

    bayests::VarTvpStochvolInput short_v = good;
    short_v.u_sigma_prior.state.omega_v = arma::vec(1, arma::fill::value(0.1));
    check("an omega_v of the wrong length is refused", refused(short_v, "log-volatility"));

    bayests::VarTvpStochvolInput zero_v = good;
    zero_v.a_prior.omega_v(2) = 0.0;
    check("a zero prior variance is refused", refused(zero_v, "greater than 0"));
}

} // namespace

int main()
{
    std::printf("unit_noncentred\n");
    arma::arma_rng::set_seed(20260921);

    the_ordinates_are_the_marginal_densities();
    the_joint_ordinate_is_the_sum_for_independent_states();
    the_draw_has_the_conditional_mean();
    the_bayes_factors_point_the_right_way();
    the_gamma_model_finds_the_shifting_intercept();
    the_vec_finds_the_volatility_that_jumps();
    the_vec_gamma_model_finds_the_shifting_constant();
    the_prior_is_one_or_the_other();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
