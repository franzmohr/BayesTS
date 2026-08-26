// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That VecKlgs2010 and VecNormalWishart are the same sampler.
//
// They implement one algorithm -- Koop, Leon-Gonzalez and Strachan (2010) --
// written two ways: VecNormalWishart against the SUR design matrix
// z = kron(W_x, I_k), VecKlgs2010 against the compact W_x it is built from. The
// second exists only because the first spends O(tt k^3 n_x^2) per draw where
// O(tt n_x^2) will do, so the whole claim it rests on is that the posterior does
// not change. This asserts it.
//
// How the comparison is made exact. Both samplers consume the RNG in the same
// order and in the same amounts -- n_a standard normals for the coefficient
// block, n_beta for the cointegration block, then the Wishart -- so seeding both
// with the same value makes the two chains the same chain, and the only
// differences left are the last bits of a differently associated matrix product.
// A single draw is compared rather than a long chain for that reason: a Gibbs
// sampler is a feedback loop, and rounding that starts at 1e-16 does not stay
// there over eighty iterations. What a one-draw comparison establishes is the
// arithmetic; that the chains then agree in distribution follows from it.
//
// Also here, because it is the same property from the other side: that the two
// models' log likelihoods agree on one posterior, and that validate() refuses
// the variable selection this sampler cannot do.

#include "bayests/vec_klgs_2010.h"
#include "bayests/vec_normal_wishart.h"

#include <iostream>
#include <string>

namespace
{

constexpr int kK = 3;             // endogenous variables
constexpr int kP = 2;             // level lags, so one lagged difference
constexpr int kRank = 1;          // cointegration rank
constexpr int kNRestricted = 1;   // a constant inside the cointegration space
constexpr int kTT = 20;           // periods
constexpr int kKBeta = kK + kNRestricted;
constexpr int kNAlpha = kK * kRank;
constexpr int kNBeta = kKBeta * kRank;
constexpr int kNX = kK * (kP - 1);      // compact regressors besides the ect
constexpr int kNA = kNAlpha + kK * kNX; // coefficients in `a`

constexpr unsigned long long kSeed = 20260826ULL;

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

/// A sample the samplers can actually run on: a stable VAR(1) in levels, from
/// which the response, the error correction term and the lagged difference are
/// all derived. Deterministic, and not drawn from Armadillo's generator -- that
/// one belongs to the samplers under comparison.
struct Sample
{
    arma::mat y;   ///< tt x k differences
    arma::mat w;   ///< tt x k_beta error correction term
    arma::mat x;   ///< tt x n_x compact regressors
    arma::mat z;   ///< (tt k) x n_a the same, in SUR form
};

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
    s.x = arma::mat(kTT, kNX);

    for (int t = 0; t < kTT; ++t)
    {
        s.y.row(t) = arma::trans(levels.col(kP + t) - levels.col(kP + t - 1));
        s.w.submat(t, 0, t, kK - 1) = arma::trans(levels.col(kP + t - 1));
        s.x.row(t) = arma::trans(levels.col(kP + t - 1) - levels.col(kP + t - 2));
    }

    // The SUR reading of the same regressors. The loading columns hold
    // kron((beta' w_t)', I_k) for the beta the chain starts from; every VEC
    // rebuilds them from its own draw before reading them, so they are here to
    // make the input well formed rather than because anything depends on them.
    const arma::mat diag_k = arma::eye<arma::mat>(kK, kK);
    arma::vec beta0(kNBeta, arma::fill::zeros);
    beta0(0) = 1.0;
    const arma::mat beta_mat = arma::reshape(beta0, kKBeta, kRank);

    s.z = arma::mat(kTT * kK, kNA, arma::fill::zeros);
    for (int t = 0; t < kTT; ++t)
    {
        s.z.submat(t * kK, 0, (t + 1) * kK - 1, kNAlpha - 1) =
            arma::kron(arma::trans(arma::trans(beta_mat) * arma::trans(s.w.row(t))), diag_k);
        s.z.submat(t * kK, kNAlpha, (t + 1) * kK - 1, kNA - 1) =
            arma::kron(s.x.row(t), diag_k);
    }

    return s;
}

arma::vec initial_beta()
{
    arma::vec beta(kNBeta, arma::fill::zeros);
    beta(0) = 1.0;
    return beta;
}

bayests::VarSpec make_spec(int iterations, int burnin)
{
    bayests::VarSpec spec;
    spec.k = kK;
    spec.p = kP;
    spec.n = 0;
    spec.n_restricted = kNRestricted;
    spec.rank = kRank;
    spec.k_beta = kKBeta;
    spec.iterations = iterations;
    spec.burnin = burnin;
    return spec;
}

bayests::VecNormalWishartInput make_sur(const Sample &s, int iterations, int burnin)
{
    bayests::VecNormalWishartInput in;
    in.spec = make_spec(iterations, burnin);
    in.train.y = s.y;
    in.train.w = s.w;
    in.train.z = s.z;

    in.a_prior.mu = arma::vec(kNA, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(kNA, kNA);
    in.beta_prior.v_inv = 0.1;
    in.beta_prior.p_tau_inv = arma::eye<arma::mat>(kKBeta, kKBeta);
    in.u_sigma_prior.df = kK;
    in.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);

    in.initial.a = arma::vec(kNA, arma::fill::zeros);
    in.initial.beta = initial_beta();
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return in;
}

bayests::VecKlgs2010Input make_compact(const Sample &s, int iterations, int burnin)
{
    bayests::VecKlgs2010Input in;
    in.spec = make_spec(iterations, burnin);
    in.train.y = s.y;
    in.train.w = s.w;
    in.train.x = s.x;

    in.a_prior.mu = arma::vec(kNA, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(kNA, kNA);
    in.beta_prior.v_inv = 0.1;
    in.beta_prior.p_tau_inv = arma::eye<arma::mat>(kKBeta, kKBeta);
    in.u_sigma_prior.df = kK;
    in.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);

    in.initial.a = arma::vec(kNA, arma::fill::zeros);
    in.initial.beta = initial_beta();
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return in;
}

/// Largest absolute difference, or infinity when the shapes disagree -- which is
/// a failure the caller should see as one rather than as an exception.
double max_diff(const arma::mat &lhs, const arma::mat &rhs)
{
    if (lhs.n_rows != rhs.n_rows || lhs.n_cols != rhs.n_cols)
    {
        return arma::datum::inf;
    }
    return lhs.n_elem == 0 ? 0.0 : arma::abs(lhs - rhs).max();
}

void expect_close(const std::string &label, const arma::mat &lhs, const arma::mat &rhs,
                  double tolerance)
{
    const double diff = max_diff(lhs, rhs);
    check(label, diff <= tolerance, "max |difference| = " + std::to_string(diff));
}

} // namespace

int main()
{
    const Sample sample = make_sample();
    bayests::NullReporter reporter;

    // One draw, no burn-in: the two chains have to agree on the arithmetic
    // before they can be said to agree on anything.
    std::cout << "one draw from the same seed:\n";

    arma::arma_rng::set_seed(kSeed);
    const bayests::VecNormalWishartDraws sur =
        bayests::VecNormalWishartSampler{}.draw_coefficients(make_sur(sample, 1, 0), reporter);

    arma::arma_rng::set_seed(kSeed);
    const bayests::VecKlgs2010Draws compact =
        bayests::VecKlgs2010Sampler{}.draw_coefficients(make_compact(sample, 1, 0), reporter);

    // Loose enough for a differently associated matrix product, tight enough
    // that a genuine disagreement -- a transposed Kronecker factor, a
    // coefficient block in the wrong order -- cannot hide under it.
    constexpr double kTolerance = 1e-9;
    expect_close("a", sur.a, compact.a, kTolerance);
    expect_close("beta", sur.beta, compact.beta, kTolerance);
    expect_close("u_sigma_inv", sur.u_sigma_inv, compact.u_sigma_inv, kTolerance);

    // A chain long enough that every branch has run repeatedly. Compared for
    // scale rather than element by element: the loop amplifies the rounding
    // above, so what is checked is that the two posteriors sit in the same
    // place, not that they are the same numbers.
    std::cout << "eighty draws, posterior means:\n";

    arma::arma_rng::set_seed(kSeed);
    const bayests::VecNormalWishartDraws sur_long =
        bayests::VecNormalWishartSampler{}.draw_coefficients(make_sur(sample, 80, 40), reporter);

    arma::arma_rng::set_seed(kSeed);
    const bayests::VecKlgs2010Draws compact_long =
        bayests::VecKlgs2010Sampler{}.draw_coefficients(make_compact(sample, 80, 40), reporter);

    expect_close("mean a", arma::mean(sur_long.a, 1), arma::mean(compact_long.a, 1), 0.05);
    expect_close("mean u_sigma_inv", arma::mean(sur_long.u_sigma_inv, 1),
                 arma::mean(compact_long.u_sigma_inv, 1), 0.05);

    // The same posterior through both log likelihoods. No RNG in either, so
    // this one is an identity up to rounding however long the chain was.
    std::cout << "log likelihood of one posterior:\n";

    bayests::VecKlgs2010Draws shared;
    shared.a = sur_long.a;
    shared.beta = sur_long.beta;
    shared.u_sigma_inv = sur_long.u_sigma_inv;

    const arma::mat loglik_sur =
        bayests::VecNormalWishartSampler{}.log_likelihood(make_sur(sample, 80, 40), sur_long);
    const arma::mat loglik_compact =
        bayests::VecKlgs2010Sampler{}.log_likelihood(make_compact(sample, 80, 40), shared);

    expect_close("loglik", loglik_sur, loglik_compact, 1e-9);

    // Variable selection is the one thing the compact form gives up, and it has
    // to be refused rather than ignored.
    std::cout << "variable selection must be rejected:\n";

    bayests::VecKlgs2010Input selected = make_compact(sample, 10, 5);
    selected.spec.varsel = bayests::VarSelection::bvs;
    bool threw = false;
    std::string what;
    try
    {
        selected.validate();
    }
    catch (const std::exception &e)
    {
        threw = true;
        what = e.what();
    }
    check("bvs -> rejected", threw, what);

    std::cout << (failures == 0 ? "all as expected\n" : "SOMETHING IS WRONG\n");
    return failures == 0 ? 0 : 1;
}
