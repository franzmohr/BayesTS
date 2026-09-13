// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_var_normal_stochvol.cpp
/// @brief Checks that VarNormalStochvol survives an observation far out in the
///        tails of every mixture component.
///
/// The sampler used to draw its mixture indicators from component densities
/// formed directly. For log(u^2) far from every component mean all ten
/// densities underflow to zero, the normalised row is NaN, and the index drawn
/// from it was one past the end of the table: `bayests coefficients` on a fixture
/// with a single value of 1e30 in `y` exited with "Mat::elem(): index out of
/// bounds", and a host that defines ARMA_NO_DEBUG read past the table instead.
/// The sampler now uses the shared routine, which forms the probabilities in
/// logs and clamps the index.

#include "bayests/reporter.h"
#include "bayests/var_normal_stochvol.h"

#include <cstdio>
#include <exception>
#include <string>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void an_outlier_does_not_break_the_chain()
{
    std::printf("an observation of 1e30\n");

    const int tt = 40;

    bayests::VarNormalStochvolInput input;
    input.spec.k = 2;
    input.spec.n = 1;
    input.spec.iterations = 50;
    input.spec.burnin = 20;

    input.train.y = arma::randn<arma::mat>(tt, 2);
    input.train.y(17, 1) = 1e30;
    input.train.z = arma::repmat(arma::eye<arma::mat>(2, 2), tt, 1);

    input.a_prior.mu = arma::zeros<arma::vec>(2);
    input.a_prior.v_inv = arma::eye<arma::mat>(2, 2);
    input.u_sigma_prior.offset = arma::vec(2, arma::fill::value(1e-4));
    input.u_sigma_prior.state.sigma.shape = arma::vec(2, arma::fill::value(3.0));
    input.u_sigma_prior.state.sigma.rate = arma::vec(2, arma::fill::value(0.2));
    input.u_sigma_prior.state.initial_state.mu = arma::zeros<arma::vec>(2);
    input.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(2, 2);

    input.initial.a = arma::zeros<arma::vec>(2);
    input.initial.h = arma::zeros<arma::mat>(tt, 2);
    input.initial.h_init = arma::zeros<arma::vec>(2);
    input.initial.h_sigma = arma::vec(2, arma::fill::value(0.05));

    bayests::NullReporter reporter;
    bool ran = false;
    bayests::VarNormalStochvolDraws draws;
    try
    {
        draws = bayests::VarNormalStochvolSampler{}.draw_coefficients(input, reporter);
        ran = true;
    }
    catch (const std::exception &e)
    {
        std::printf("    threw: %s\n", e.what());
    }

    check("the chain runs to the end", ran);
    check("every kept precision is finite", ran && draws.u_omega_inv.is_finite());
}

} // namespace

int main()
{
    std::printf("unit_var_normal_stochvol\n");
    arma::arma_rng::set_seed(20260913);

    an_outlier_does_not_break_the_chain();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
