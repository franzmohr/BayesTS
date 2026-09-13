// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_bvs.cpp
/// @brief Checks that a BVS sweep draws its indicators from their posterior, and
///        that the covariance block's selection is informed by the data.
///
/// The first half puts the sweep on its own against a log likelihood that is a
/// known function of which positions are switched on, so the posterior over the
/// indicators is a four-cell table that can be written down. A Gibbs scan visits
/// those cells in proportion to it; the rule this replaced did not, and at a
/// prior inclusion probability of one half it kept a flat-likelihood position
/// included for ever. Frequencies over a long chain rather than an identity, so
/// the tolerance is statistical -- but the chain is seeded, and the tolerance is
/// several standard errors wide.
///
/// The second half runs the two constant-coefficient VARs with a covariance
/// block on errors that are strongly correlated and on errors that are not.
/// Their selection step used to score its candidates against a regressor matrix
/// that was still zero, which made the likelihood flat and the indicators a
/// function of the prior alone: a prior inclusion probability of 0.1 against a
/// correlation the data leave no doubt about must now come out near one. The two
/// VECs share the block line for line with the VAR beside them.

#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_stochvol.h"
#include "core/algorithms/bvs.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{

using bayests::core::BvsBlock;
using bayests::core::BvsScope;
using bayests::core::bvs_sweep;

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

bayests::VarSelPrior selection_prior(const arma::vec &inprior)
{
    bayests::VarSelPrior prior;
    prior.inprior = inprior;
    prior.include = arma::regspace<arma::uvec>(0, inprior.n_elem - 1);
    return prior;
}

/// One or zero: whether row `i` of a candidate is switched on.
template <class Coefficients>
double switched_on(const Coefficients &candidate, const arma::uword i)
{
    return arma::accu(arma::abs(candidate.row(i))) > 0.0 ? 1.0 : 0.0;
}

/// How often a chain of sweeps over two positions visits each of the states
/// (g1, g2) = (0,0), (1,0), (0,1), (1,1), when the log likelihood of a mask is
/// c1 g1 + c2 g2 + c12 g1 g2.
template <class Coefficients>
arma::vec visit_frequencies(const Coefficients &drawn, const BvsScope scope,
                            const arma::vec &inprior, const double c1, const double c2,
                            const double c12, const int sweeps)
{
    BvsBlock blk(arma::vec{1.0, 1.0}, selection_prior(inprior));
    arma::vec counts(4, arma::fill::zeros);

    for (int s = 0; s < sweeps; s++)
    {
        // The sweep masks its argument on the way out, so each one is handed the
        // coefficients afresh -- as a sampler does, drawing them anew in between.
        Coefficients coef = drawn;
        bvs_sweep(blk, coef, scope, [&](const Coefficients &candidate) {
            const double g1 = switched_on(candidate, 0);
            const double g2 = switched_on(candidate, 1);
            return c1 * g1 + c2 * g2 + c12 * g1 * g2;
        });
        counts(static_cast<arma::uword>(blk.lambda(0) + 2.0 * blk.lambda(1))) += 1.0;
    }

    return counts / sweeps;
}

/// The posterior those frequencies should converge to.
arma::vec posterior_frequencies(const arma::vec &inprior, const double c1, const double c2,
                                const double c12)
{
    arma::vec weight(4);
    for (int g2 = 0; g2 <= 1; g2++)
    {
        for (int g1 = 0; g1 <= 1; g1++)
        {
            const double prior = (g1 == 1 ? inprior(0) : 1.0 - inprior(0)) *
                                 (g2 == 1 ? inprior(1) : 1.0 - inprior(1));
            weight(g1 + 2 * g2) = prior * std::exp(c1 * g1 + c2 * g2 + c12 * g1 * g2);
        }
    }
    return weight / arma::accu(weight);
}

void a_flat_likelihood_returns_the_prior()
{
    std::printf("a flat likelihood\n");

    const arma::vec inprior{0.1, 0.5};
    const arma::vec freq =
        visit_frequencies(arma::vec{1.0, -2.0}, BvsScope::element, inprior, 0.0, 0.0, 0.0, 100000);

    // Marginal inclusion of each position. The rule this replaced gave 0.53 and
    // 1.00.
    const double first = freq(1) + freq(3);
    const double second = freq(2) + freq(3);
    std::printf("    inclusion %.4f and %.4f against priors 0.1 and 0.5\n", first, second);
    check("a prior of 0.1 is returned", std::abs(first - 0.1) < 0.01);
    check("a prior of 0.5 is returned", std::abs(second - 0.5) < 0.01);
}

void the_indicators_have_their_joint_posterior()
{
    std::printf("the joint posterior of two dependent indicators\n");

    // An interaction strong enough that scoring each position against a stale
    // mask would visibly miss it.
    const arma::vec inprior{0.3, 0.6};
    const double c1 = 0.5, c2 = -1.0, c12 = 2.0;
    const arma::vec want = posterior_frequencies(inprior, c1, c2, c12);

    const arma::vec element =
        visit_frequencies(arma::vec{0.7, 1.3}, BvsScope::element, inprior, c1, c2, c12, 100000);
    const arma::mat path{{0.7, -0.2, 0.4}, {1.3, 0.9, -1.1}};
    const arma::vec path_row =
        visit_frequencies(path, BvsScope::path_row, inprior, c1, c2, c12, 100000);

    std::printf("    posterior  %.4f %.4f %.4f %.4f\n", want(0), want(1), want(2), want(3));
    std::printf("    element    %.4f %.4f %.4f %.4f\n", element(0), element(1), element(2),
                element(3));
    std::printf("    path_row   %.4f %.4f %.4f %.4f\n", path_row(0), path_row(1), path_row(2),
                path_row(3));
    check("element scope visits the posterior", arma::abs(element - want).max() < 0.01);
    check("path_row scope visits the posterior", arma::abs(path_row - want).max() < 0.01);
}

/// Two variables whose errors load on each other by `loading`, over tt periods.
arma::mat errors(const int tt, const double loading)
{
    arma::mat y(tt, 2);
    y.col(0) = arma::randn<arma::vec>(tt);
    y.col(1) = loading * y.col(0) + 0.5 * arma::randn<arma::vec>(tt);
    return y;
}

template <class Input>
void fill_common(Input &input, const arma::mat &y, const double inprior)
{
    input.spec.k = 2;
    input.spec.iterations = 1500;
    input.spec.burnin = 500;
    input.spec.varsel = bayests::VarSelection::bvs;
    input.spec.covar = true;

    input.train.y = y;

    input.psi_prior.mu = arma::vec(1, arma::fill::zeros);
    input.psi_prior.v_inv = arma::mat(1, 1, arma::fill::ones);
    input.psi_varsel_prior = selection_prior(arma::vec{inprior});

    input.initial.psi = arma::vec(1, arma::fill::zeros);
    input.initial.psi_lambda = arma::vec(1, arma::fill::ones);
}

/// Posterior mean inclusion of the one free element of Psi.
double gamma_inclusion(const arma::mat &y, const double inprior)
{
    bayests::VarNormalGammaInput input;
    fill_common(input, y, inprior);
    input.u_sigma_prior.shape = arma::vec(2, arma::fill::value(3.0));
    input.u_sigma_prior.rate = arma::vec(2, arma::fill::value(2.0));
    input.initial.u_sigma_inv = arma::eye<arma::mat>(2, 2);

    bayests::NullReporter reporter;
    const bayests::VarNormalGammaDraws draws =
        bayests::VarNormalGammaSampler{}.draw_coefficients(input, reporter);
    return arma::mean(draws.psi_lambda.row(1)); // element (1, 0) of vec(Psi_lambda)
}

double stochvol_inclusion(const arma::mat &y, const double inprior)
{
    const arma::uword tt = y.n_rows;

    bayests::VarNormalStochvolInput input;
    fill_common(input, y, inprior);
    input.u_sigma_prior.offset = arma::vec(2, arma::fill::value(1e-6));
    input.u_sigma_prior.state.sigma.shape = arma::vec(2, arma::fill::value(5.0));
    input.u_sigma_prior.state.sigma.rate = arma::vec(2, arma::fill::value(0.05));
    input.u_sigma_prior.state.initial_state.mu = arma::vec(2, arma::fill::zeros);
    input.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(2, 2);
    input.initial.h = arma::mat(tt, 2, arma::fill::zeros);
    input.initial.h_init = arma::vec(2, arma::fill::zeros);
    input.initial.h_sigma = arma::vec(2, arma::fill::value(0.01));

    bayests::NullReporter reporter;
    const bayests::VarNormalStochvolDraws draws =
        bayests::VarNormalStochvolSampler{}.draw_coefficients(input, reporter);
    return arma::mean(draws.psi_lambda.row(1));
}

void covariance_block_selection_sees_the_data()
{
    std::printf("selection over the covariance block\n");

    const arma::mat correlated = errors(200, 0.9);
    const arma::mat independent = errors(200, 0.0);

    const double gamma_in = gamma_inclusion(correlated, 0.1);
    const double gamma_out = gamma_inclusion(independent, 0.5);
    const double sv_in = stochvol_inclusion(correlated, 0.1);
    const double sv_out = stochvol_inclusion(independent, 0.5);

    std::printf("    VarNormalGamma     correlated %.3f (prior 0.1), independent %.3f (prior 0.5)\n",
                gamma_in, gamma_out);
    std::printf("    VarNormalStochvol  correlated %.3f (prior 0.1), independent %.3f (prior 0.5)\n",
                sv_in, sv_out);

    check("VarNormalGamma includes a correlation the data show", gamma_in > 0.9);
    check("VarNormalGamma excludes one they do not", gamma_out < 0.8);
    check("VarNormalStochvol includes a correlation the data show", sv_in > 0.9);
    check("VarNormalStochvol excludes one they do not", sv_out < 0.8);
}

} // namespace

int main()
{
    std::printf("unit_bvs\n");
    arma::arma_rng::set_seed(20260913);

    a_flat_likelihood_returns_the_prior();
    the_indicators_have_their_joint_posterior();
    covariance_block_selection_sees_the_data();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
