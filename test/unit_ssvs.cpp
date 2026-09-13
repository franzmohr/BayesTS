// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_ssvs.cpp
/// @brief Checks that an SSVS sweep draws each indicator from its posterior, and
///        that it still does so where the mixture densities underflow.
///
/// Given its coefficient an indicator's posterior is a ratio of two weighted
/// normal densities, so the probability a chain of sweeps includes it can be
/// written down. The second check is the one this was written for: a
/// coefficient many slab widths from zero sends both densities to zero, and the
/// sweep used to divide one by the other, draw against a NaN and exclude the
/// coefficient the data most want in.

#include "core/algorithms/ssvs.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{

using bayests::core::SsvsBlock;
using bayests::core::ssvs_sweep;

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

bayests::VarSelPrior spike_and_slab(const double inprior, const double tau0, const double tau1)
{
    bayests::VarSelPrior prior;
    prior.inprior = arma::vec{inprior};
    prior.include = arma::uvec{0};
    prior.ssvs.tau0 = arma::vec{tau0};
    prior.ssvs.tau1 = arma::vec{tau1};
    return prior;
}

void the_indicator_has_its_posterior()
{
    std::printf("the posterior of one indicator\n");

    const double inprior = 0.4, tau0 = 0.05, tau1 = 1.0, coef = 0.12;
    const double slab = inprior / tau1 * std::exp(-coef * coef / (2 * tau1 * tau1));
    const double spike = (1 - inprior) / tau0 * std::exp(-coef * coef / (2 * tau0 * tau0));
    const double want = slab / (slab + spike);

    SsvsBlock blk(arma::vec{1.0}, spike_and_slab(inprior, tau0, tau1));
    arma::mat prior_v_inv(1, 1, arma::fill::ones);
    const arma::vec drawn{coef};

    const int sweeps = 200000;
    double included = 0.0;
    bool precision_follows = true;
    for (int s = 0; s < sweeps; s++)
    {
        ssvs_sweep(blk, drawn, prior_v_inv);
        included += blk.lambda(0);
        const double expected = 1 / (blk.lambda(0) == 1.0 ? tau1 * tau1 : tau0 * tau0);
        precision_follows = precision_follows && prior_v_inv(0, 0) == expected;
    }

    std::printf("    inclusion %.4f against %.4f\n", included / sweeps, want);
    check("the inclusion frequency is the posterior", std::abs(included / sweeps - want) < 0.005);
    check("the prior precision follows the indicator", precision_follows);
}

void a_coefficient_far_from_zero_is_included()
{
    std::printf("both densities underflowing\n");

    // exp(-50^2 / 2) and exp(-50^2 / (2 * 0.01^2)) are both zero in double
    // precision; their log odds are about 1.25e7.
    SsvsBlock blk(arma::vec{0.0}, spike_and_slab(0.5, 0.01, 1.0));
    arma::mat prior_v_inv(1, 1, arma::fill::ones);
    const arma::vec drawn{50.0};

    bool always = true;
    for (int s = 0; s < 1000; s++)
    {
        ssvs_sweep(blk, drawn, prior_v_inv);
        always = always && blk.lambda(0) == 1.0;
    }
    check("it is included in every sweep", always);
    check("its prior precision is the slab's", prior_v_inv(0, 0) == 1.0);
}

} // namespace

int main()
{
    std::printf("unit_ssvs\n");
    arma::arma_rng::set_seed(20260913);

    the_indicator_has_its_posterior();
    a_coefficient_far_from_zero_is_included();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
