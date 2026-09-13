// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// The informative marginal prior on a time-varying cointegration space.
//
// Koop, Leon-Gonzalez and Strachan (2011, working paper version, eq. 12) centre
// the marginal prior of the space on sp(H) by putting P_tau = H H' + tau H_perp
// H_perp' into the transition of the state equation, leaving the innovation
// variance at the identity. What is asserted here, on VecTvpWishart -- the three
// time-varying VECs share the block, and the fixtures run the other two:
//
//   1. P_tau = I is the prior every file without one has always meant, draw for
//      draw. Same seed, same chain, compared exactly, with rho drawn so that its
//      block is on the path too. This is the identity the change rests on.
//   2. A P_tau other than the identity reaches the sampler: same seed, different
//      chain.
//   3. validate() refuses a P_tau of the wrong size, one that is not symmetric,
//      and one with an eigenvalue outside [0, 1], and accepts the tau = 0 end.
//
// Like unit_vec_klgs_2010, it draws and so depends on the RNG, but what it
// asserts is a relation between two chains from the same seed rather than the
// value of either.

#include "bayests/vec_tvp_wishart.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kK = 3;           // endogenous variables
constexpr int kP = 2;           // level lags, so one lagged difference
constexpr int kRank = 1;        // cointegration rank
constexpr int kNRestricted = 1; // a constant inside the cointegration space
constexpr int kTT = 20;         // periods
constexpr int kKBeta = kK + kNRestricted;
constexpr int kNAlpha = kK * kRank;
constexpr int kNBeta = kKBeta * kRank;
constexpr int kNX = kK * (kP - 1);
constexpr int kNA = kNAlpha + kK * kNX;

// Above the hundred draws the progress reporting divides by.
constexpr int kIterations = 80;
constexpr int kBurnin = 40;

constexpr double kRho = 0.99;
constexpr double kTau = 0.5;

constexpr unsigned long long kSeed = 20260913ULL;

int failures = 0;

void check(const std::string &label, bool condition, const std::string &detail = "")
{
    failures += condition ? 0 : 1;
    std::cout << (condition ? "  ok   " : "  FAIL ") << label << "\n";
    if (!detail.empty())
    {
        std::cout << "         " << detail << "\n";
    }
}

struct Sample
{
    arma::mat y; ///< tt x k differences
    arma::mat w; ///< tt x k_beta error correction term
    arma::mat z; ///< (tt k) x n_a regressors in SUR form
};

/// A stable VAR(1) in levels, deterministic and not drawn from Armadillo's
/// generator, which belongs to the chains under comparison.
Sample make_sample()
{
    arma::mat levels(kK, kTT + kP, arma::fill::zeros);
    unsigned long long state = 12345ULL;
    const auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const unsigned long long mantissa = (state >> 11) & ((1ULL << 53) - 1);
        return static_cast<double>(mantissa) / 9007199254740992.0 - 0.5;
    };

    for (int t = 1; t < kTT + kP; ++t)
    {
        for (int i = 0; i < kK; ++i)
        {
            levels(i, t) = 0.5 * levels(i, t - 1) + next();
        }
    }

    Sample s;
    s.y = arma::mat(kTT, kK);
    s.w = arma::mat(kTT, kKBeta, arma::fill::ones);
    arma::mat x(kTT, kNX);

    for (int t = 0; t < kTT; ++t)
    {
        s.y.row(t) = arma::trans(levels.col(kP + t) - levels.col(kP + t - 1));
        s.w.submat(t, 0, t, kK - 1) = arma::trans(levels.col(kP + t - 1));
        x.row(t) = arma::trans(levels.col(kP + t - 1) - levels.col(kP + t - 2));
    }

    // The loading columns are rebuilt from beta before they are read; they are
    // filled here only so that the input is well formed.
    const arma::mat diag_k = arma::eye<arma::mat>(kK, kK);
    s.z = arma::mat(kTT * kK, kNA, arma::fill::zeros);
    for (int t = 0; t < kTT; ++t)
    {
        s.z.submat(t * kK, kNAlpha, (t + 1) * kK - 1, kNA - 1) = arma::kron(x.row(t), diag_k);
    }

    return s;
}

arma::vec initial_beta()
{
    arma::vec beta(kNBeta, arma::fill::zeros);
    beta(0) = 1.0;
    beta(1) = -0.5;
    return beta;
}

/// P_tau = H H' + tau H_perp H_perp' with H the direction the chain starts in.
arma::mat informative_p_tau(double tau)
{
    const arma::vec h = arma::normalise(initial_beta());
    const arma::mat along = h * arma::trans(h);
    return along + tau * (arma::eye<arma::mat>(kKBeta, kKBeta) - along);
}

bayests::VecTvpWishartInput make_input(const Sample &s)
{
    bayests::VecTvpWishartInput in;
    in.spec.k = kK;
    in.spec.p = kP;
    in.spec.n = 0;
    in.spec.n_restricted = kNRestricted;
    in.spec.rank = kRank;
    in.spec.k_beta = kKBeta;
    in.spec.iterations = kIterations;
    in.spec.burnin = kBurnin;

    in.train.y = s.y;
    in.train.w = s.w;
    in.train.z = s.z;

    in.a_prior.sigma.shape = arma::vec(kNA, arma::fill::value(3.0));
    in.a_prior.sigma.rate = arma::vec(kNA, arma::fill::value(1e-4));
    in.a_prior.initial_state.mu = arma::vec(kNA, arma::fill::zeros);
    in.a_prior.initial_state.v_inv = arma::eye<arma::mat>(kNA, kNA);

    in.beta_prior.rho = kRho;
    in.beta_prior.rho_prior.draw = true;
    in.beta_prior.rho_prior.min = 0.9;
    in.beta_prior.rho_prior.max = 0.999;
    in.beta_prior.initial_state.mu = arma::vec(kNBeta, arma::fill::zeros);
    in.beta_prior.initial_state.v_inv = (1.0 - kRho * kRho) * arma::eye<arma::mat>(kNBeta, kNBeta);

    in.u_sigma_prior.df = kK;
    in.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);

    in.initial.a = arma::mat(kNA, kTT, arma::fill::zeros);
    in.initial.a_sigma_inv = 1e4 * arma::eye<arma::mat>(kNA, kNA);
    in.initial.a_init = arma::vec(kNA, arma::fill::zeros);
    in.initial.beta = arma::repmat(initial_beta(), 1, kTT);
    in.initial.beta_init = initial_beta();
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return in;
}

bayests::VecTvpWishartDraws run(bayests::VecTvpWishartInput in, const arma::mat &p_tau)
{
    bayests::NullReporter reporter;
    in.beta_prior.p_tau = p_tau;
    arma::arma_rng::set_seed(kSeed);
    return bayests::VecTvpWishartSampler{}.draw_coefficients(in, reporter);
}

/// Largest absolute difference, or infinity when the shapes disagree.
double max_diff(const arma::mat &lhs, const arma::mat &rhs)
{
    if (lhs.n_rows != rhs.n_rows || lhs.n_cols != rhs.n_cols)
    {
        return arma::datum::inf;
    }
    return lhs.n_elem == 0 ? 0.0 : arma::abs(lhs - rhs).max();
}

void expect_identical(const std::string &label, const arma::mat &lhs, const arma::mat &rhs)
{
    const double diff = max_diff(lhs, rhs);
    check(label, diff == 0.0, "max |difference| = " + std::to_string(diff));
}

void expect_refused(const std::string &label, const Sample &s, const arma::mat &p_tau,
                    const std::string &mentions)
{
    bayests::VecTvpWishartInput in = make_input(s);
    in.beta_prior.p_tau = p_tau;

    std::string message;
    try
    {
        in.validate();
    }
    catch (const std::invalid_argument &e)
    {
        message = e.what();
    }
    check(label, message.find(mentions) != std::string::npos,
          message.empty() ? "validate() accepted it" : message);
}

} // namespace

int main()
{
    const Sample sample = make_sample();
    const bayests::VecTvpWishartInput input = make_input(sample);

    std::cout << "P_tau = I is the prior without one, draw for draw:\n";
    const bayests::VecTvpWishartDraws plain = run(input, arma::mat());
    const bayests::VecTvpWishartDraws identity = run(input, arma::eye<arma::mat>(kKBeta, kKBeta));

    expect_identical("a", plain.a, identity.a);
    expect_identical("a_sigma", plain.a_sigma, identity.a_sigma);
    expect_identical("beta", plain.beta, identity.beta);
    expect_identical("rho", plain.rho, identity.rho);
    expect_identical("u_sigma_inv", plain.u_sigma_inv, identity.u_sigma_inv);

    std::cout << "an informative P_tau reaches the sampler:\n";
    const bayests::VecTvpWishartDraws informative = run(input, informative_p_tau(kTau));
    check("beta differs", max_diff(plain.beta, informative.beta) > 0.0);
    check("rho differs", max_diff(plain.rho, informative.rho) > 0.0);
    check("rho stays in its support",
          informative.rho.min() >= 0.9 && informative.rho.max() <= 0.999);

    std::cout << "validate():\n";
    expect_refused("refuses the wrong size", sample, arma::eye<arma::mat>(kKBeta - 1, kKBeta - 1),
                   "P_tau");

    arma::mat asymmetric = informative_p_tau(kTau);
    asymmetric(0, 1) += 0.1;
    expect_refused("refuses an asymmetric P_tau", sample, asymmetric, "symmetric");

    expect_refused("refuses an eigenvalue above one", sample,
                   1.5 * arma::eye<arma::mat>(kKBeta, kKBeta), "[0, 1]");

    arma::mat negative = arma::eye<arma::mat>(kKBeta, kKBeta);
    negative(kKBeta - 1, kKBeta - 1) = -0.1;
    expect_refused("refuses a negative eigenvalue", sample, negative, "[0, 1]");

    bayests::VecTvpWishartInput edge = make_input(sample);
    edge.beta_prior.p_tau = informative_p_tau(0.0);
    bool accepted = true;
    try
    {
        edge.validate();
    }
    catch (const std::exception &)
    {
        accepted = false;
    }
    check("accepts tau = 0", accepted);

    std::cout << (failures == 0 ? "all checks passed\n" : "some checks FAILED\n");
    return failures == 0 ? 0 : 1;
}
