// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_steady_state.cpp
/// @brief Checks the steady-state prior of Villani (2009).
///
/// Under the steady-state prior the prior is on the unconditional mean mu and
/// the intercept is (I - sum_j A_j) mu, written into `a` so that everything
/// downstream reads a draw as any other. That is an identity every draw has to
/// satisfy exactly. Beyond it: a loose prior recovers the mean of a simulated
/// VAR; a tight prior at a wrong mean pulls the posterior of mu onto itself,
/// which is what says the prior is on mu and not on the intercept; with the
/// stationarity condition every kept draw is stationary; VarNormalGamma reads
/// it as VarNormalWishart does. And what is refused is refused.

#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_wishart.h"
#include "bayests/var_tvp_wishart.h"
#include "core/models/shrinkage_support.h"

#include <cmath>
#include <cstdio>
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

void check_below(const std::string &what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-66s %s   %.3e < %.3e\n", what.c_str(), ok ? "ok" : "FAILED", got, bound);
    if (!ok)
    {
        failures++;
    }
}

constexpr arma::uword kK = 3;
constexpr arma::uword kP = 1;
constexpr arma::uword kNx = kK * kP + 1;

struct Sample
{
    arma::mat y;
    arma::mat z;
    arma::vec mu;
    arma::mat a_lag;
};

/// A VAR(1) around a known mean.
Sample simulate(const arma::uword tt)
{
    Sample s;
    s.mu = {2.0, -1.0, 0.5};
    s.a_lag = {{0.6, 0.1, 0.0}, {0.0, 0.5, 0.1}, {0.1, 0.0, 0.4}};
    arma::arma_rng::set_seed(808);
    arma::vec previous = s.mu;
    arma::mat path(kK, tt + 1);
    path.col(0) = previous;
    for (arma::uword t = 1; t <= tt; t++)
    {
        previous = s.mu + s.a_lag * (previous - s.mu) + 0.5 * arma::randn<arma::vec>(kK);
        path.col(t) = previous;
    }
    s.y = arma::trans(path.cols(1, tt));
    arma::mat x(tt, kNx);
    x.col(kNx - 1).ones();
    x.cols(0, kK - 1) = arma::trans(path.cols(0, tt - 1));
    s.z = arma::kron(x, arma::eye<arma::mat>(kK, kK));
    return s;
}

template <typename Input>
void fill(Input &input, const Sample &s, const int iterations, const int burnin)
{
    input.spec.k = static_cast<int>(kK);
    input.spec.p = static_cast<int>(kP);
    input.spec.n = 1;
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.spec.steady_state = true;
    input.train.y = s.y;
    input.train.z = s.z;
    const arma::uword n = kK * kNx;
    input.a_prior.mu = arma::zeros<arma::vec>(n);
    input.a_prior.v_inv = arma::eye<arma::mat>(n, n);
    input.mu_prior.mu = arma::zeros<arma::vec>(kK);
    input.mu_prior.v_inv = 0.01 * arma::eye<arma::mat>(kK, kK);
    input.initial.a = arma::zeros<arma::vec>(n);
    input.initial.mu = arma::zeros<arma::vec>(kK);
}

bayests::VarNormalWishartInput wishart_input(const Sample &s, const int iterations, const int burnin)
{
    bayests::VarNormalWishartInput input;
    fill(input, s, iterations, burnin);
    input.u_sigma_prior.df = static_cast<int>(kK) + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return input;
}

/// Every draw's intercept is (I - sum_j A_j) mu, to rounding.
template <typename Draws>
double worst_intercept(const Draws &draws)
{
    double worst = 0.0;
    for (arma::uword d = 0; d < draws.a.n_cols; d++)
    {
        const arma::mat a = arma::reshape(draws.a.col(d), kK, kNx);
        const arma::vec want = (arma::eye<arma::mat>(kK, kK) - a.cols(0, kK - 1)) * draws.mu.col(d);
        worst = std::max(worst, arma::abs(a.col(kNx - 1) - want).max());
    }
    return worst;
}

void test_wishart()
{
    std::printf("VarNormalWishart under a steady-state prior\n");
    const Sample s = simulate(200);
    bayests::NullReporter reporter;

    const auto loose_input = wishart_input(s, 2000, 500);
    arma::arma_rng::set_seed(1);
    const auto loose = bayests::VarNormalWishartSampler{}.draw_coefficients(loose_input, reporter);
    check_below("every draw's intercept is (I - sum_j A_j) mu", worst_intercept(loose), 1e-12);

    const arma::vec mean = arma::mean(loose.mu, 1);
    const arma::vec sd = arma::stddev(loose.mu, 0, 1);
    std::printf("    posterior mean of mu: %.3f %.3f %.3f, truth 2 -1 0.5\n", mean(0), mean(1),
                mean(2));
    check_below("a loose prior recovers the mean, within four posterior sds",
                (arma::abs(mean - s.mu) / sd).max(), 4.0);

    auto tight_input = loose_input;
    tight_input.mu_prior.v_inv = 1e6 * arma::eye<arma::mat>(kK, kK);
    arma::arma_rng::set_seed(1);
    const auto tight = bayests::VarNormalWishartSampler{}.draw_coefficients(tight_input, reporter);
    check_below("a tight prior at zero holds mu at zero: the prior is on mu",
                arma::abs(arma::mean(tight.mu, 1)).max(), 0.01);

    auto stationary_input = loose_input;
    stationary_input.spec.stationary = true;
    arma::arma_rng::set_seed(1);
    const auto stationary =
        bayests::VarNormalWishartSampler{}.draw_coefficients(stationary_input, reporter);
    bool all = true;
    for (arma::uword d = 0; d < stationary.a.n_cols; d++)
    {
        all = all && bayests::core::is_stationary(stationary.a.col(d), kK, kP);
    }
    check("with the stationarity condition every draw is stationary", all);
}

void test_gamma()
{
    std::printf("VarNormalGamma under a steady-state prior\n");
    const Sample s = simulate(200);
    bayests::VarNormalGammaInput input;
    fill(input, s, 500, 200);
    input.u_sigma_prior.shape = arma::vec(kK, arma::fill::value(3.0));
    input.u_sigma_prior.rate = arma::vec(kK, arma::fill::value(1.0));
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(2);
    const auto draws = bayests::VarNormalGammaSampler{}.draw_coefficients(input, reporter);
    check_below("every draw's intercept is (I - sum_j A_j) mu", worst_intercept(draws), 1e-12);
    check_below("and the mean is recovered", arma::abs(arma::mean(draws.mu, 1) - s.mu).max(), 0.5);
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
    const Sample s = simulate(40);
    const auto base = wishart_input(s, 10, 0);

    auto input = base;
    input.initial.mu.reset();
    check("a steady-state prior without a start is refused", refuses(input, "/initial/mu"));

    input = base;
    input.mu_prior.v_inv = -arma::eye<arma::mat>(kK, kK);
    check("so is a precision that is not positive definite", refuses(input, "positive definite"));

    input = base;
    input.a_prior.v_inv(0, kK * kK) = input.a_prior.v_inv(kK * kK, 0) = 0.1;
    check("so is a base prior coupling the lags and the intercept", refuses(input, "couples"));

    input = base;
    input.spec.shrinkage = bayests::Shrinkage::minnesota;
    input.a_shrinkage_prior.group = arma::ones<arma::uvec>(kK * kNx);
    input.a_shrinkage_prior.shape = {2.0};
    input.a_shrinkage_prior.rate = {1.0};
    check("and one beside shrinkage", refuses(input, "steady_state"));

    bayests::VarTvpWishartInput tvp;
    tvp.spec.k = static_cast<int>(kK);
    tvp.spec.steady_state = true;
    check("VarTvpWishart refuses it by name",
          refuses(tvp, "VarTvpWishart does not read /model/steady_state"));
}

} // namespace

int main()
{
    try
    {
        test_wishart();
        test_gamma();
        test_refusals();
    }
    catch (const std::exception &e)
    {
        std::printf("FAILED: threw: %s\n", e.what());
        failures++;
    }
    if (failures != 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
