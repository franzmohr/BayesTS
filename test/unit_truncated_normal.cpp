// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_truncated_normal.cpp
/// @brief Checks the truncated normal draw against its support and its moments.
///
/// Built the way unit_inverse_gaussian.cpp is: a draw has no exact identity to
/// assert, so what is checked is the support, which a wrong envelope violates
/// outright, and the first two moments, which a correct envelope applied over
/// the wrong region gets wrong while still landing inside the interval.
///
/// The five intervals are one per branch of the sampler, plus the mirror. Two of
/// them are the case the caller actually produces -- the autoregression of a
/// cointegration state equation, whose conditional mean can sit tens of standard
/// deviations outside a prior support that is itself a fraction of a standard
/// deviation wide. That is the regime in which a draw built on the truncated
/// quantile function returns the nearer endpoint every time, which looks like a
/// well behaved chain stuck at a boundary.
///
/// The moments are compared against the closed forms
///
///     E[x]   = mu + sd (phi(a) - phi(b)) / Z,
///     Var[x] = sd^2 [1 + (a phi(a) - b phi(b)) / Z - ((phi(a) - phi(b)) / Z)^2],
///
/// with a, b the standardised bounds and Z = Phi(b) - Phi(a). Z is formed out of
/// the upper tail, and an interval below zero is mirrored onto one above it
/// before any of it is evaluated, so the expected values themselves do not lose
/// their digits where the draw is hardest.

#include "core/algorithms/truncated_normal.h"

#include "bayests/arma.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

using bayests::core::truncated_normal;

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-52s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_close(const std::string &what, const double got, const double want, const double tol)
{
    const double difference = std::abs(got - want);
    const bool ok = difference < tol;
    std::printf("  %-52s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    std::printf("      got %.8f, want %.8f, difference %.8f, tolerance %.8f\n", got, want,
                difference, tol);
    if (!ok)
    {
        failures++;
    }
}

void check_rejects(const char *what, const double mu, const double sd, const double lower,
                   const double upper)
{
    bool threw = false;
    try
    {
        truncated_normal(mu, sd, lower, upper);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    check(what, threw);
}

double phi(const double x)
{
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * arma::datum::pi);
}

/// The upper tail of the standard normal, which keeps its digits where 1 - Phi
/// loses them.
double upper_tail(const double x)
{
    return 0.5 * std::erfc(x / std::sqrt(2.0));
}

/// The first two moments of the standard normal restricted to [a, b].
void standard_moments(const double a, const double b, double &mean, double &variance)
{
    // Mirror an interval below zero, where the two tails the mass is formed out
    // of would cancel to nothing.
    if (b <= 0.0)
    {
        standard_moments(-b, -a, mean, variance);
        mean = -mean;
        return;
    }

    const double mass = upper_tail(a) - upper_tail(b);
    const double ratio = (phi(a) - phi(b)) / mass;

    mean = ratio;
    variance = 1.0 + (a * phi(a) - b * phi(b)) / mass - ratio * ratio;
}

struct Interval
{
    const char *name;
    double mu;
    double sd;
    double lower;
    double upper;
};

void the_draws_have_the_moments_of_the_truncated_normal()
{
    std::printf("the moments\n");

    // One per branch: an interval wide around the mode, a narrow one around it,
    // a wide tail, a narrow tail far out, and a tail below the mode, which is
    // the mirror.
    const Interval cases[5] = {
        {"wide around the mode", 0.5, 2.0, -2.5, 4.5},
        {"narrow around the mode", 0.5, 2.0, 0.2, 0.9},
        {"a wide tail", 0.0, 1.0, 2.0, 12.0},
        {"a narrow tail far out", 0.95, 0.01, 0.999, 1.0},
        {"a tail below the mode", 0.0, 1.0, -12.0, -3.0},
    };

    const arma::uword n = 200000;

    for (const Interval &c : cases)
    {
        arma::arma_rng::set_seed(20260911);

        arma::vec draws(n);
        for (arma::uword i = 0; i < n; i++)
        {
            draws[i] = truncated_normal(c.mu, c.sd, c.lower, c.upper);
        }

        double standard_mean = 0.0;
        double standard_variance = 0.0;
        standard_moments((c.lower - c.mu) / c.sd, (c.upper - c.mu) / c.sd, standard_mean,
                         standard_variance);

        const double mean = c.mu + c.sd * standard_mean;
        const double variance = c.sd * c.sd * standard_variance;
        const double sd = std::sqrt(variance);

        check(std::string("inside the interval, ") + c.name,
              draws.min() >= c.lower && draws.max() <= c.upper);

        // Five standard errors of the Monte Carlo mean, and six of the sample
        // variance, whose own standard error is variance * sqrt(2 / n) for a
        // distribution no heavier tailed than this one.
        check_close(std::string("mean, ") + c.name, arma::mean(draws), mean,
                    5.0 * sd / std::sqrt(static_cast<double>(n)));
        check_close(std::string("variance, ") + c.name, arma::var(draws), variance,
                    6.0 * variance * std::sqrt(2.0 / static_cast<double>(n)));
    }
}

void the_standardisation_is_the_only_difference()
{
    std::printf("the standardisation\n");

    // The interval is standardised before a single variate is drawn, so the same
    // seed gives the same standard draw and the location and the scale are put
    // back afterwards. That identity is what carries the moment checks above
    // from the standard normal to every (mu, sd) a caller passes.
    const double mu = -3.0;
    const double sd = 0.25;
    const double lower = -3.4;
    const double upper = -2.9;

    const arma::uword n = 200;
    arma::vec direct(n);
    arma::vec rebuilt(n);

    arma::arma_rng::set_seed(4);
    for (arma::uword i = 0; i < n; i++)
    {
        direct[i] = truncated_normal(mu, sd, lower, upper);
    }

    arma::arma_rng::set_seed(4);
    for (arma::uword i = 0; i < n; i++)
    {
        rebuilt[i] = mu + sd * truncated_normal(0.0, 1.0, (lower - mu) / sd, (upper - mu) / sd);
    }

    check("a shifted draw is the standard one shifted",
          arma::approx_equal(direct, rebuilt, "absdiff", 1e-12));
}

void a_degenerate_interval_still_terminates()
{
    std::printf("the short intervals\n");
    arma::arma_rng::set_seed(99);

    // As short as a double can express, once around the mode and once a hundred
    // million standard deviations out. Neither is a sensible prior; both are
    // what a badly scaled file produces, and a rejection loop that cannot accept
    // would hang rather than fail.
    check_close("an interval of no width around the mode",
                truncated_normal(0.0, 1.0, 1e-300, 2e-300), 0.0, 1e-12);

    const double upper = std::nextafter(1.0, 2.0);
    const double draw = truncated_normal(0.0, 1e-8, 1.0, upper);
    check("an interval one ulp wide far out is inside it", draw >= 1.0 && draw <= upper);
}

void the_impossible_arguments_are_refused()
{
    std::printf("the refusals\n");

    check_rejects("a standard deviation of zero", 0.0, 0.0, -1.0, 1.0);
    check_rejects("a negative standard deviation", 0.0, -1.0, -1.0, 1.0);
    check_rejects("an empty interval", 0.0, 1.0, 1.0, 1.0);
    check_rejects("a reversed interval", 0.0, 1.0, 1.0, -1.0);
    check_rejects("a mean that is not a number", std::nan(""), 1.0, -1.0, 1.0);
    check_rejects("an infinite bound", 0.0, 1.0, -std::numeric_limits<double>::infinity(), 1.0);
}

} // namespace

int main()
{
    std::printf("unit_truncated_normal\n");

    the_draws_have_the_moments_of_the_truncated_normal();
    the_standardisation_is_the_only_difference();
    a_degenerate_interval_still_terminates();
    the_impossible_arguments_are_refused();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
