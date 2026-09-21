// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_coint_klgs_prior.cpp
/// @brief The cointegration space prior of Koop, Leon-Gonzalez and Strachan
///        (2010) as the four constant VECs hold it: what validate() refuses,
///        the pseudo-observations the prior owes the error precision, and that
///        VecNormalGamma now pays them.
///
/// Three parts.
///
/// The refusals. The collapsed sampler assumes alpha | beta is centred at zero
/// and independent of every other coefficient, v >= 0, and a positive definite
/// P_tau whenever v > 0; a file that says otherwise used to run, and each of
/// the three Gibbs blocks then conditioned on a different prior. G^-1 is read
/// by VecNormalStochvol alone and refused by the others. Every constant VEC is
/// checked, so a validate() that stops going through the shared code fails here.
///
/// The pseudo-observations. coint_prior_pseudo_errors() returns L with
/// L L' = v alpha (beta' P^-1 beta) alpha' -- the identity checked directly --
/// and zero at v = 0.
///
/// The fix to VecNormalGamma. With one variable the Wishart on the error
/// precision is a gamma, W(nu, S^-1) = Gamma(nu / 2, rate S / 2), so a
/// VecNormalWishart and a VecNormalGamma given matching priors are the same
/// model, and both samplers have to find the same posterior. VecNormalWishart
/// has always added the prior's term (eq. 8); VecNormalGamma left it out, and
/// on the data below that put the posterior mean of the precision 2.6% off the
/// Wishart's, against 0.07% with it -- measured by switching the term off and
/// watching this fail. Long seeded chains, compared on the mean of the
/// precision and of Pi = alpha beta', the only identified product.

#include "bayests/reporter.h"
#include "bayests/vec_klgs_2010.h"
#include "bayests/vec_normal_gamma.h"
#include "bayests/vec_normal_stochvol.h"
#include "bayests/vec_normal_wishart.h"
#include "core/models/vec_support.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok, const std::string &detail = "")
{
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
        if (!detail.empty())
        {
            std::printf("      %s\n", detail.c_str());
        }
    }
}

// --- The refusals ------------------------------------------------------------

constexpr int kK = 2;
constexpr int kP = 2;                  // level lags: one lagged difference
constexpr int kRank = 1;
constexpr int kKBeta = kK;
constexpr int kTT = 12;
constexpr int kNAlpha = kK * kRank;
constexpr int kNX = kK * (kP - 1);
constexpr int kNA = kNAlpha + kK * kNX;

bayests::VarSpec small_spec()
{
    bayests::VarSpec spec;
    spec.k = kK;
    spec.p = kP;
    spec.rank = kRank;
    spec.k_beta = kKBeta;
    spec.iterations = 10;
    spec.burnin = 5;
    return spec;
}

template <class Input>
void fill_small(Input &in, const bool compact)
{
    in.spec = small_spec();
    in.train.y = arma::mat(kTT, kK, arma::fill::ones);
    in.train.w = arma::mat(kTT, kKBeta, arma::fill::ones);
    if (compact)
    {
        in.train.x = arma::mat(kTT, kNX, arma::fill::ones);
    }
    else
    {
        in.train.z = arma::mat(kTT * kK, kNA, arma::fill::ones);
    }
    in.a_prior.mu = arma::vec(kNA, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(kNA, kNA);
    in.beta_prior.v_inv = 0.1;
    in.beta_prior.p_tau_inv = arma::eye<arma::mat>(kKBeta, kKBeta);
    in.initial.a = arma::vec(kNA, arma::fill::zeros);
    in.initial.beta = arma::vec{1.0, -1.0} / std::sqrt(2.0);
}

template <class Input>
void fill_wishart(Input &in)
{
    in.u_sigma_prior.df = kK;
    in.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
}

/// What validate() says about `in`: empty when it accepts it.
template <class Input>
std::string refusal(const Input &in)
{
    try
    {
        in.validate();
    }
    catch (const std::exception &e)
    {
        return e.what();
    }
    return "";
}

template <class Input>
void expect_refusals(const std::string &model, const Input &base, const bool reads_g)
{
    std::printf("%s\n", model.c_str());
    check("the paper's prior is accepted", refusal(base).empty(), refusal(base));

    Input in = base;
    in.a_prior.mu(1) = 0.5;
    check("a non-zero prior mean on alpha is refused",
          refusal(in).find("must be zero") != std::string::npos, refusal(in));

    in = base;
    in.a_prior.mu(kNAlpha) = 0.5;
    check("a non-zero prior mean past alpha is not", refusal(in).empty(), refusal(in));

    in = base;
    in.a_prior.v_inv(0, kNAlpha) = in.a_prior.v_inv(kNAlpha, 0) = 0.2;
    check("prior precision coupling alpha to the rest is refused",
          refusal(in).find("couples position 1") != std::string::npos, refusal(in));

    in = base;
    in.a_prior.v_inv(0, 1) = in.a_prior.v_inv(1, 0) = 0.2;
    check("off-diagonals inside alpha's own block are not (never read)", refusal(in).empty(),
          refusal(in));

    in = base;
    in.beta_prior.v_inv = -0.1;
    check("a negative shrinkage v is refused",
          refusal(in).find("at least zero") != std::string::npos, refusal(in));

    in = base;
    in.beta_prior.p_tau_inv = arma::diagmat(arma::vec{1.0, 0.0});
    check("a singular P_tau^-1 is refused when v > 0",
          refusal(in).find("positive definite") != std::string::npos, refusal(in));
    in.beta_prior.v_inv = 0.0;
    check("and accepted at v = 0, where it is never read", refusal(in).empty(), refusal(in));

    in = base;
    in.beta_prior.g_inv = arma::eye<arma::mat>(kK, kK);
    if (reads_g)
    {
        check("a positive definite G^-1 is accepted", refusal(in).empty(), refusal(in));
        in.beta_prior.g_inv(1, 1) = -1.0;
        check("an indefinite one is refused",
              refusal(in).find("positive definite") != std::string::npos, refusal(in));
    }
    else
    {
        check("a G^-1 is refused, since only VecNormalStochvol reads one",
              refusal(in).find("VecNormalStochvol alone") != std::string::npos, refusal(in));
    }
}

// --- The pseudo-observations -------------------------------------------------

void pseudo_errors_carry_the_prior()
{
    std::printf("coint_prior_pseudo_errors\n");

    const arma::mat alpha = {{0.3, -0.1}, {0.2, 0.4}, {-0.5, 0.05}};   // k = 3, r = 2
    arma::mat beta = arma::orth(arma::mat{{1.0, 0.2}, {-0.4, 1.0}, {0.3, -0.7}, {0.1, 0.1}});
    arma::mat p_inv = {{2.0, 0.3, 0.0, 0.1},
                       {0.3, 1.5, 0.2, 0.0},
                       {0.0, 0.2, 1.0, 0.0},
                       {0.1, 0.0, 0.0, 3.0}};
    const double v = 0.7;

    const arma::mat l = bayests::core::coint_prior_pseudo_errors(alpha, beta, v, p_inv);
    const arma::mat want = v * alpha * (beta.t() * p_inv * beta) * alpha.t();
    check("L is k x rank", l.n_rows == 3 && l.n_cols == 2);
    check("L L' = v alpha (beta' P^-1 beta) alpha'",
          arma::approx_equal(l * l.t(), want, "absdiff", 1e-12));

    const arma::mat identity = bayests::core::coint_prior_pseudo_errors(alpha, beta, v, arma::mat());
    check("an empty P^-1 is the identity",
          arma::approx_equal(identity * identity.t(), v * alpha * alpha.t(), "absdiff", 1e-12));

    const arma::mat flat = bayests::core::coint_prior_pseudo_errors(alpha, beta, 0.0, p_inv);
    check("v = 0 gives zero columns, still rank of them",
          flat.n_rows == 3 && flat.n_cols == 2 && arma::all(arma::vectorise(flat) == 0.0));
}

// --- VecNormalGamma against VecNormalWishart with one variable ---------------

struct Univariate
{
    arma::mat dy;   // tt x 1
    arma::mat lag;  // tt x 1, the level y_{t-1}
};

Univariate univariate_data(const int tt)
{
    Univariate d;
    d.dy.set_size(tt, 1);
    d.lag.set_size(tt, 1);
    double level = 2.0;
    for (int t = 0; t < tt; t++)
    {
        d.lag(t, 0) = level;
        d.dy(t, 0) = -0.5 * level + arma::randn();
        level += d.dy(t, 0);
    }
    return d;
}

template <class Input>
void fill_univariate(Input &in, const Univariate &d, const int iterations)
{
    in.spec.k = 1;
    in.spec.p = 1;
    in.spec.rank = 1;
    in.spec.k_beta = 1;
    in.spec.iterations = iterations;
    in.spec.burnin = 2000;
    in.train.y = d.dy;
    in.train.w = d.lag;
    in.train.z = d.lag;           // beta' w with beta = 1; rebuilt every draw
    in.a_prior.mu = arma::vec(1, arma::fill::zeros);
    in.a_prior.v_inv = arma::mat(1, 1, arma::fill::ones);
    // Strong shrinkage against data that want alpha well away from zero. The two
    // parts of the prior's term in the error precision -- r/2 on the shape and
    // v alpha (beta' P^-1 beta) alpha' / 2 on the rate -- pull in opposite
    // directions and very nearly cancel when alpha sits where its prior puts it,
    // which is why a test on weak data cannot see them. Here leaving both out
    // moves the posterior mean of the precision by 2.6%.
    in.beta_prior.v_inv = 50.0;
    in.beta_prior.p_tau_inv = arma::mat(1, 1, arma::fill::ones);
    in.initial.a = arma::vec(1, arma::fill::zeros);
    in.initial.beta = arma::vec(1, arma::fill::ones);
    in.initial.u_sigma_inv = arma::mat(1, 1, arma::fill::ones);
}

void gamma_and_wishart_agree_with_one_variable()
{
    std::printf("VecNormalGamma and VecNormalWishart, one variable, matching priors\n");

    const int tt = 200;
    const int iterations = 40000;
    const Univariate d = univariate_data(tt);

    // W(nu, S^-1) on a 1 x 1 precision is Gamma(nu / 2, rate S / 2).
    const double nu = 4.0;
    const double scale = 2.0;

    bayests::VecNormalWishartInput wishart;
    fill_univariate(wishart, d, iterations);
    wishart.u_sigma_prior.df = static_cast<int>(nu);
    wishart.u_sigma_prior.scale = arma::mat(1, 1, arma::fill::value(scale));

    bayests::VecNormalGammaInput gamma;
    fill_univariate(gamma, d, iterations);
    gamma.u_sigma_prior.shape = arma::vec(1, arma::fill::value(nu / 2));
    gamma.u_sigma_prior.rate = arma::vec(1, arma::fill::value(scale / 2));

    bayests::NullReporter reporter;
    const bayests::VecNormalWishartDraws w =
        bayests::VecNormalWishartSampler{}.draw_coefficients(wishart, reporter);
    const bayests::VecNormalGammaDraws g =
        bayests::VecNormalGammaSampler{}.draw_coefficients(gamma, reporter);

    // Pi = alpha beta', the product that is identified; alpha and beta alone
    // are only up to a sign here.
    const arma::rowvec pi_w = w.a.row(0) % w.beta.row(0);
    const arma::rowvec pi_g = g.a.row(0) % g.beta.row(0);
    const double prec_w = arma::mean(w.u_sigma_inv.row(0));
    const double prec_g = arma::mean(g.u_omega_inv.row(0));

    std::printf("    precision  wishart %.4f  gamma %.4f\n", prec_w, prec_g);
    std::printf("    Pi         wishart %.4f  gamma %.4f\n", arma::mean(pi_w), arma::mean(pi_g));

    // The Monte Carlo error of each mean is about 0.15% here; leaving the
    // prior's term out moved the gamma sampler's by 2.6%.
    check("the posterior mean of the precision agrees within 1%",
          std::abs(prec_g / prec_w - 1.0) < 0.01);
    check("the posterior mean of Pi agrees within 0.01",
          std::abs(arma::mean(pi_g) - arma::mean(pi_w)) < 0.01);
    check("and so does its posterior standard deviation, within 5%",
          std::abs(arma::stddev(pi_g) / arma::stddev(pi_w) - 1.0) < 0.05);
}

} // namespace

int main()
{
    std::printf("unit_coint_klgs_prior\n");
    arma::arma_rng::set_seed(20260921);

    bayests::VecNormalWishartInput normal_wishart;
    fill_small(normal_wishart, false);
    fill_wishart(normal_wishart);
    expect_refusals("VecNormalWishart", normal_wishart, false);

    bayests::VecKlgs2010Input klgs;
    fill_small(klgs, true);
    fill_wishart(klgs);
    expect_refusals("VecKlgs2010", klgs, false);

    bayests::VecNormalGammaInput normal_gamma;
    fill_small(normal_gamma, false);
    normal_gamma.u_sigma_prior.shape = arma::vec(kK, arma::fill::value(3.0));
    normal_gamma.u_sigma_prior.rate = arma::vec(kK, arma::fill::value(2.0));
    normal_gamma.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    expect_refusals("VecNormalGamma", normal_gamma, false);

    bayests::VecNormalStochvolInput normal_stochvol;
    fill_small(normal_stochvol, false);
    normal_stochvol.u_sigma_prior.offset = arma::vec(kK, arma::fill::value(1e-4));
    normal_stochvol.u_sigma_prior.state.sigma.shape = arma::vec(kK, arma::fill::value(3.0));
    normal_stochvol.u_sigma_prior.state.sigma.rate = arma::vec(kK, arma::fill::value(0.1));
    normal_stochvol.u_sigma_prior.state.initial_state.mu = arma::vec(kK, arma::fill::zeros);
    normal_stochvol.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(kK, kK);
    normal_stochvol.initial.h = arma::mat(kTT, kK, arma::fill::zeros);
    normal_stochvol.initial.h_init = arma::vec(kK, arma::fill::zeros);
    normal_stochvol.initial.h_sigma = arma::vec(kK, arma::fill::value(0.1));
    expect_refusals("VecNormalStochvol", normal_stochvol, true);

    pseudo_errors_carry_the_prior();
    gamma_and_wishart_agree_with_one_variable();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
