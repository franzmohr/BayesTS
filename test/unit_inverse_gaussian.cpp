// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_inverse_gaussian.cpp
/// @brief Checks the inverse Gaussian draw against its moments and its support.
///
/// A draw has no exact identity to assert, so this is built the way
/// unit_stochvol.cpp is: moments that hold whatever the generator does, over
/// enough draws that the Monte Carlo error is far inside the bound, plus the
/// part of the support that a wrong root selection would leave.
///
/// The two moments do most of the work together. The mean alone is not enough:
/// the transformation has two roots, and returning the wrong one with the wrong
/// probability still gives a distribution on the right support with a mean that
/// can look close. It is the variance, mu^3 / lambda, that pins which root is
/// taken and how often -- so a sampler that inverts the selection rule passes
/// the first check and fails the second.
///
/// The reciprocal is checked as well, because that is what the caller actually
/// wants: the full conditional of an asymmetric Laplace scale is generalised
/// inverse Gaussian at index 1/2, which is the reciprocal of what this draws.
/// If 1/x has the mean the reciprocal law implies, the caller's inversion is
/// sound.

#include "core/algorithms/inverse_gaussian.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{

int failures = 0;

void check(const char *what, const bool ok)
{
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_close(const char *what, const double got, const double want, const double tol)
{
    const double relative = std::abs(got - want) / std::abs(want);
    const bool ok = relative < tol;
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAILED");
    std::printf("      got %.6f, want %.6f, relative %.5f, tolerance %.5f\n", got, want,
                relative, tol);
    if (!ok)
    {
        failures++;
    }
}

void check_rejects(const char *what, const arma::vec &mu, const arma::vec &lambda)
{
    bool threw = false;
    try
    {
        inverse_gaussian(mu, lambda);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    check(what, threw);
}

/// The first two moments of the draw, over `n` variates of one parameter pair.
void moments(const double mu, const double lambda, const arma::uword n, double &mean,
             double &variance)
{
    const arma::vec draws =
        inverse_gaussian(arma::vec(n, arma::fill::value(mu)), arma::vec(1, arma::fill::value(lambda)));
    mean = arma::mean(draws);
    variance = arma::var(draws);
}

void it_has_the_moments_of_an_inverse_gaussian()
{
    std::printf("the moments\n");
    arma::arma_rng::set_seed(20260904);

    const arma::uword n = 400000;

    // Three regimes: lambda well above mu (nearly normal), lambda near mu, and
    // lambda well below mu (heavily skewed, which is where a wrong root shows).
    const double mus[3] = {1.0, 2.0, 0.5};
    const double lambdas[3] = {8.0, 2.0, 0.25};
    const char *names[3] = {"mu = 1, lambda = 8", "mu = 2, lambda = 2", "mu = 0.5, lambda = 0.25"};

    for (int i = 0; i < 3; i++)
    {
        double mean = 0.0;
        double variance = 0.0;
        moments(mus[i], lambdas[i], n, mean, variance);

        const std::string mean_label = std::string("mean, ") + names[i];
        const std::string var_label = std::string("variance, ") + names[i];

        check_close(mean_label.c_str(), mean, mus[i], 0.02);
        check_close(var_label.c_str(), variance,
                    mus[i] * mus[i] * mus[i] / lambdas[i], 0.08);
    }
}

void the_reciprocal_is_the_law_the_caller_inverts()
{
    std::printf("the reciprocal\n");
    arma::arma_rng::set_seed(11);

    // If x ~ IG(mu, lambda) then E[1/x] = 1/mu + 1/lambda. That identity is
    // what makes the caller's `1 / inverse_gaussian(...)` the generalised
    // inverse Gaussian draw it is meant to be.
    const double mu = 1.5;
    const double lambda = 3.0;
    const arma::uword n = 400000;

    const arma::vec draws =
        inverse_gaussian(arma::vec(n, arma::fill::value(mu)), arma::vec(1, arma::fill::value(lambda)));

    check_close("E[1/x] is 1/mu + 1/lambda", arma::mean(1.0 / draws), 1.0 / mu + 1.0 / lambda,
                0.02);
}

void every_draw_is_inside_the_support()
{
    std::printf("the support\n");
    arma::arma_rng::set_seed(7);

    // Parameters spanning several orders of magnitude, which is where the
    // cancellation the closed form avoids would otherwise put a root at or
    // below zero.
    arma::vec mu(2000);
    arma::vec lambda(2000);
    for (arma::uword i = 0; i < mu.n_elem; i++)
    {
        mu[i] = std::pow(10.0, -4.0 + 8.0 * static_cast<double>(i) / static_cast<double>(mu.n_elem));
        lambda[i] = std::pow(10.0, 4.0 - 8.0 * static_cast<double>(i) / static_cast<double>(mu.n_elem));
    }

    const arma::vec draws = inverse_gaussian(mu, lambda);

    check("the draw has one element per mean", draws.n_elem == mu.n_elem);
    check("every draw is strictly positive", draws.min() > 0.0);
    check("every draw is finite", draws.is_finite());
}

void a_scalar_lambda_applies_to_every_draw()
{
    std::printf("the scalar shape\n");
    arma::arma_rng::set_seed(3);

    const arma::vec mu(50, arma::fill::value(2.0));
    const arma::vec scalar(1, arma::fill::value(4.0));
    const arma::vec repeated(50, arma::fill::value(4.0));

    arma::arma_rng::set_seed(3);
    const arma::vec from_scalar = inverse_gaussian(mu, scalar);
    arma::arma_rng::set_seed(3);
    const arma::vec from_repeated = inverse_gaussian(mu, repeated);

    check("a scalar lambda is the repeated one",
          arma::approx_equal(from_scalar, from_repeated, "absdiff", 0.0));
}

void the_impossible_arguments_are_refused()
{
    std::printf("the refusals\n");

    const arma::vec ok_mu(4, arma::fill::ones);
    const arma::vec ok_lambda(4, arma::fill::ones);

    check_rejects("an empty mean", arma::vec(), ok_lambda);
    check_rejects("a lambda of the wrong length", ok_mu, arma::vec(3, arma::fill::ones));

    arma::vec zero_mu = ok_mu;
    zero_mu[2] = 0.0;
    check_rejects("a mean of zero", zero_mu, ok_lambda);

    arma::vec negative_mu = ok_mu;
    negative_mu[0] = -1.0;
    check_rejects("a negative mean", negative_mu, ok_lambda);

    arma::vec zero_lambda = ok_lambda;
    zero_lambda[1] = 0.0;
    check_rejects("a shape of zero", ok_mu, zero_lambda);

    arma::vec nan_mu = ok_mu;
    nan_mu[3] = std::nan("");
    check_rejects("a mean that is not a number", nan_mu, ok_lambda);

    arma::vec inf_lambda = ok_lambda;
    inf_lambda[0] = std::numeric_limits<double>::infinity();
    check_rejects("an infinite shape", ok_mu, inf_lambda);
}

} // namespace

int main()
{
    std::printf("unit_inverse_gaussian\n");

    it_has_the_moments_of_an_inverse_gaussian();
    the_reciprocal_is_the_law_the_caller_inverts();
    every_draw_is_inside_the_support();
    a_scalar_lambda_applies_to_every_draw();
    the_impossible_arguments_are_refused();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
