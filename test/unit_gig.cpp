// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_gig.cpp
/// @brief Checks the generalised inverse Gaussian draw against its moments.
///
/// Built the way unit_inverse_gaussian.cpp is: moments that hold whatever the
/// generator does, over enough draws that the Monte Carlo error is far inside
/// the bound. The sampler has three branches, so the grid visits each of them
/// on both sides of lambda = 0 and at the extremes a caller reaches -- the
/// normal-gamma prior hands it chi as small as a squared coefficient and lambda
/// = theta - 1/2 just below zero. E[X] and E[1/X] together pin both tails: a
/// hat that does not cover the density loses mass in one of them.
///
/// The expected moments are ratios of modified Bessel functions of the second
/// kind, E[X^r] = (chi/psi)^(r/2) K_(lambda+r)(w) / K_lambda(w) with
/// w = sqrt(chi psi); dividing X by sqrt(chi / psi) leaves the ratio, so the
/// draws are checked standardised, which keeps chi = 1e-200 from overflowing
/// the second moment of 1/X. Where w is that small and lambda < 0 the mean of
/// X rests on draws too rare to see, and its standard error says so; E[1/X]
/// is what pins that case. The ratios are computed here by the trapezoid rule on
/// 2 K_nu(w) = int exp(nu t - w cosh t) dt over the real line, whose integrand
/// is analytic and decays double exponentially, so the rule converges to the
/// last digit -- rather than by std::cyl_bessel_k, which not every standard
/// library has.

#include "core/algorithms/gig_hormann_leydold_2014.h"

#include "bayests/arma.h"

#include <algorithm>
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

/// log of int exp(nu t - w cosh t) dt, which is log(2 K_nu(w)).
double log_bessel_integral(const double nu, const double w)
{
    const double span = std::log(2.0 * (std::fabs(nu) + 60.0) / w) + 5.0;
    const double h = 1e-3;
    const long n = static_cast<long>(2.0 * span / h);
    double top = -INFINITY;
    for (long i = 0; i <= n; i++)
    {
        const double t = -span + h * static_cast<double>(i);
        top = std::max(top, nu * t - w * std::cosh(t));
    }
    double sum = 0.0;
    for (long i = 0; i <= n; i++)
    {
        const double t = -span + h * static_cast<double>(i);
        sum += std::exp(nu * t - w * std::cosh(t) - top);
    }
    return top + std::log(sum * h);
}

/// E[Y^r] for Y = X / sqrt(chi / psi), X ~ GIG(lambda, chi, psi): the
/// moments of the standardised variable, which depend on w alone. Checked in
/// place of those of X so that chi = 1e-200 does not overflow the second
/// moment of 1/X.
double moment(const double r, const double lambda, const double w)
{
    return std::exp(log_bessel_integral(lambda + r, w) - log_bessel_integral(lambda, w));
}

void check_moments(const double lambda, const double chi, const double psi, const char *branch)
{
    const arma::uword n = 200000;
    const double w = std::sqrt(chi) * std::sqrt(psi);
    const double alpha = std::sqrt(chi) / std::sqrt(psi);
    arma::vec x(n);
    for (arma::uword i = 0; i < n; i++)
    {
        x(i) = bayests::core::gig_hormann_leydold_2014(lambda, chi, psi) / alpha;
    }
    char label[160];
    std::snprintf(label, sizeof(label), "GIG(%g, %g, %g), %s", lambda, chi, psi, branch);
    std::printf("%s\n", label);

    check("every draw is positive and finite", x.is_finite() && x.min() > 0.0);

    for (const double r : {1.0, -1.0})
    {
        const double m1 = moment(r, lambda, w);
        const double m2 = moment(2.0 * r, lambda, w);
        const double se = std::sqrt((m2 - m1 * m1) / static_cast<double>(n));
        const double got = arma::mean(r > 0 ? x : 1.0 / x);
        const double z = (got - m1) / se;
        char what[160];
        std::snprintf(what, sizeof(what), "E[Y^%+.0f] = %.6g, drawn %.6g, %+.2f standard errors", r,
                      m1, got, z);
        check(what, std::fabs(z) < 5.0);
    }
}

template <typename F>
bool refuses(F &&f)
{
    try
    {
        f();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    arma::arma_rng::set_seed(20260928);
    try
    {
        // The rejection method for a density that is not T-concave: |lambda| < 1
        // and w <= 0.2 -- the normal-gamma prior's home ground.
        check_moments(-0.4, 1e-4, 0.2, "rejection");
        check_moments(0.0, 0.01, 0.01, "rejection, lambda = 0");
        check_moments(0.45, 1e-8, 10.0, "rejection");
        check_moments(-0.9, 0.04, 0.04, "rejection");
        check_moments(-0.4, 1e-200, 0.1, "rejection, w = 3e-101");
        // Ratio of uniforms without a shift.
        check_moments(0.5, 1.0, 1.0, "ratio of uniforms");
        check_moments(1.5, 0.01, 0.01, "ratio of uniforms");
        check_moments(-1.0, 0.3, 0.3, "ratio of uniforms");
        // Ratio of uniforms around the mode.
        check_moments(3.0, 2.0, 5.0, "shifted ratio of uniforms");
        check_moments(0.1, 50.0, 50.0, "shifted ratio of uniforms");
        check_moments(-25.0, 3.0, 0.5, "shifted ratio of uniforms");
        check_moments(0.3, 1e4, 1e-2, "shifted ratio of uniforms");
        check_moments(-0.4, 1e6, 1e2, "shifted ratio of uniforms, w = 1e4");

        std::printf("limits and refusals\n");
        const arma::uword n = 200000;
        arma::vec g(n);
        arma::vec ig(n);
        for (arma::uword i = 0; i < n; i++)
        {
            g(i) = bayests::core::gig_hormann_leydold_2014(2.0, 0.0, 4.0);
            ig(i) = bayests::core::gig_hormann_leydold_2014(-3.0, 2.0, 0.0);
        }
        // Gamma(2, rate 2): mean 1, sd 1/sqrt(2); inverse gamma(3, rate 1):
        // mean 1/2, sd 1/2.
        check("chi = 0 is the gamma limit", std::fabs(arma::mean(g) - 1.0) < 5.0 * 0.7071 / std::sqrt(n));
        check("psi = 0 is the inverse gamma limit",
              std::fabs(arma::mean(ig) - 0.5) < 5.0 * 0.5 / std::sqrt(n));
        check("chi = 0 with lambda <= 0 is refused",
              refuses([] { bayests::core::gig_hormann_leydold_2014(-0.5, 0.0, 1.0); }));
        check("psi = 0 with lambda >= 0 is refused",
              refuses([] { bayests::core::gig_hormann_leydold_2014(0.5, 1.0, 0.0); }));
        check("a negative chi is refused",
              refuses([] { bayests::core::gig_hormann_leydold_2014(0.5, -1.0, 1.0); }));
        check("a NaN is refused",
              refuses([] { bayests::core::gig_hormann_leydold_2014(std::nan(""), 1.0, 1.0); }));
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
