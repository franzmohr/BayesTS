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
///
/// The last part is the file's side of the same contract. The sweep scores
/// N(0, tau0^2) against N(0, tau1^2) one position at a time, which is George,
/// Sun and Ni's (2008) prior with R = I -- and validate() refuses the three
/// files that would draw the coefficients under a different prior from the one
/// the indicators are scored against: a non-zero prior mean at a selected
/// position, a prior precision coupling one to anything else, and a spike that
/// is not narrower than its slab.

#include "bayests/var_normal_gamma.h"
#include "core/algorithms/ssvs.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
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

/// A VarNormalGamma input selecting by SSVS over both coefficients of a
/// one-variable AR(1) with an intercept, and a prior the paper would accept.
bayests::VarNormalGammaInput ssvs_input()
{
    bayests::VarNormalGammaInput input;
    input.spec.k = 1;
    input.spec.p = 1;
    input.spec.iterations = 10;
    input.spec.varsel = bayests::VarSelection::ssvs;

    const int tt = 30;
    input.train.y = arma::randn<arma::mat>(tt, 1);
    input.train.z = arma::join_rows(arma::ones<arma::mat>(tt, 1), arma::randn<arma::mat>(tt, 1));

    input.a_prior.mu = arma::vec(2, arma::fill::zeros);
    input.a_prior.v_inv = arma::eye<arma::mat>(2, 2);
    input.a_varsel_prior.inprior = arma::vec(2, arma::fill::value(0.5));
    input.a_varsel_prior.include = arma::uvec{1};
    input.a_varsel_prior.ssvs.tau0 = arma::vec(2, arma::fill::value(0.1));
    input.a_varsel_prior.ssvs.tau1 = arma::vec(2, arma::fill::value(10.0));

    input.u_sigma_prior.shape = arma::vec(1, arma::fill::value(3.0));
    input.u_sigma_prior.rate = arma::vec(1, arma::fill::value(2.0));
    input.initial.a = arma::vec(2, arma::fill::zeros);
    input.initial.a_lambda = arma::vec(2, arma::fill::ones);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(1, 1);
    return input;
}

/// Whether validate() refuses the input once `change` has been made to it, and
/// with a message mentioning `part`.
bool refused(const std::function<void(bayests::VarNormalGammaInput &)> &change,
             const std::string &part)
{
    bayests::VarNormalGammaInput input = ssvs_input();
    change(input);
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &e)
    {
        return std::string(e.what()).find(part) != std::string::npos;
    }
    return false;
}

void a_prior_ssvs_cannot_score_is_refused()
{
    std::printf("a prior the sweep would score against the wrong model\n");

    bool accepted = true;
    try
    {
        ssvs_input().validate();
    }
    catch (const std::exception &)
    {
        accepted = false;
    }
    check("the paper's prior is accepted", accepted);

    check("a non-zero mean at a selected position is refused",
          refused([](auto &in) { in.a_prior.mu(1) = 1.0; }, "must be zero at every selected"));
    check("a non-zero mean elsewhere is not",
          !refused([](auto &in) { in.a_prior.mu(0) = 1.0; }, ""));
    check("a precision coupling a selected position is refused",
          refused([](auto &in) { in.a_prior.v_inv(0, 1) = in.a_prior.v_inv(1, 0) = 0.2; },
                  "couples position 2 to position 1"));
    check("a spike as wide as its slab is refused",
          refused([](auto &in) { in.a_varsel_prior.ssvs.tau0(1) = 10.0; },
                  "tau0 must be smaller than tau1"));
    check("and one wider is refused too",
          refused([](auto &in) { in.a_varsel_prior.ssvs.tau0(1) = 20.0; },
                  "tau0 must be smaller than tau1"));
    check("a swapped pair outside `include` is not read, so not refused",
          !refused([](auto &in) { in.a_varsel_prior.ssvs.tau0(0) = 20.0; }, ""));
}

} // namespace

int main()
{
    std::printf("unit_ssvs\n");
    arma::arma_rng::set_seed(20260913);

    the_indicator_has_its_posterior();
    a_coefficient_far_from_zero_is_included();
    a_prior_ssvs_cannot_score_is_refused();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
