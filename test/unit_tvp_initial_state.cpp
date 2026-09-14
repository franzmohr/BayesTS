// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_tvp_initial_state.cpp
/// @brief Checks that a time-varying sampler finds the posterior of a slowly
///        moving coefficient path wherever its chain starts.
///
/// The time-varying blocks used to draw the path against the previous draw of
/// the state before the sample, with the random walk's own innovation variance
/// as the prior covariance of the first period, and then draw that state given
/// the path. With the innovation variance as small as a prior rate of 1e-12
/// makes it, the two steps held each other in place: a chain returned its
/// starting path as the posterior, and two chains started apart never met.
/// initial_state_variance() in src/core/models/model_support.h has the rest.
///
/// So two chains are run from the same seed on the same data, one started at
/// zero and one started at 5, and their posterior means have to agree. On the
/// old draw they differed by the distance between the starts; the tolerance of
/// 0.25 is a small fraction of that, and several posterior standard deviations
/// of a regression on 200 periods. It is a statistical statement rather than an
/// identity, made with that margin, like unit_coint_jacobian's.
///
/// Two samplers cover the two ways a block reaches the smoother: VarTvpGamma
/// with its covariance block, so that the coefficient path and psi are both on
/// the path, and VecTvpWishart, whose coefficient block carries the loadings
/// ahead of the coefficients started far away. The other time-varying samplers
/// make the same draw with the same few lines, and the factor models reach it
/// through draw_random_walk_state(); the fixtures run all of them.
///
/// Around it: validate() refusing a prior precision on the state before the
/// sample that has no inverse, which the draw now needs.

#include "bayests/reporter.h"
#include "bayests/var_tvp_gamma.h"
#include "bayests/vec_tvp_wishart.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok, const std::string &detail = "")
{
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
        if (!detail.empty())
        {
            std::printf("    %s\n", detail.c_str());
        }
    }
}

constexpr int kTT = 200;
constexpr int kIterations = 300;
constexpr int kBurnin = 200;

// A random walk meant to barely move, which is what froze the old draw.
constexpr double kRate = 1e-12;

constexpr double kFar = 5.0;
constexpr double kTolerance = 0.25;

constexpr unsigned long long kSeed = 20260914ULL;

/// Uniform on (-0.5, 0.5), deterministic and not drawn from Armadillo's
/// generator, which belongs to the chains under comparison.
class Uniform
{
  public:
    double next()
    {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const unsigned long long mantissa = (state_ >> 11) & ((1ULL << 53) - 1);
        return static_cast<double>(mantissa) / 9007199254740992.0 - 0.5;
    }

  private:
    unsigned long long state_ = 12345ULL;
};

/// Posterior mean of every element of a path, averaged over the periods too:
/// `draws` is (n tt) x iterations, periods stacked within a column.
arma::vec path_mean(const arma::mat &draws, const arma::uword n)
{
    const arma::vec over_draws = arma::mean(draws, 1);
    return arma::mean(arma::reshape(over_draws, n, over_draws.n_elem / n), 1);
}

std::string largest_gap(const arma::vec &lhs, const arma::vec &rhs)
{
    return "largest |difference| = " + std::to_string(arma::abs(lhs - rhs).max());
}

bool refused(const std::function<void()> &validate)
{
    try
    {
        validate();
    }
    catch (const std::invalid_argument &e)
    {
        return std::string(e.what()).find("positive definite") != std::string::npos;
    }
    return false;
}

// VarTvpGamma --------------------------------------------------------------

constexpr int kVarK = 2;
constexpr int kVarNParams = kVarK * kVarK; // one lag, no deterministic terms
constexpr int kVarNPsi = 1;

/// A stable VAR(1) with independent errors.
bayests::TrainData var_sample()
{
    const arma::mat a_true = {{0.5, 0.2}, {0.0, 0.3}};
    Uniform uniform;

    arma::mat levels(kVarK, kTT + 1, arma::fill::zeros);
    for (int t = 1; t <= kTT; ++t)
    {
        arma::vec e(kVarK);
        for (int i = 0; i < kVarK; ++i)
        {
            e(i) = uniform.next();
        }
        levels.col(t) = a_true * levels.col(t - 1) + e;
    }

    bayests::TrainData train;
    train.y = arma::trans(levels.cols(1, kTT));
    train.z = arma::zeros<arma::mat>(kTT * kVarK, kVarNParams);
    const arma::mat diag_k = arma::eye<arma::mat>(kVarK, kVarK);
    for (int t = 0; t < kTT; ++t)
    {
        train.z.rows(t * kVarK, (t + 1) * kVarK - 1) = arma::kron(arma::trans(levels.col(t)), diag_k);
    }
    return train;
}

bayests::VarTvpGammaInput var_input(const double start)
{
    bayests::VarTvpGammaInput in;
    in.spec.k = kVarK;
    in.spec.p = 1;
    in.spec.n = 0;
    in.spec.covar = true;
    in.spec.iterations = kIterations;
    in.spec.burnin = kBurnin;
    in.train = var_sample();

    in.a_prior.sigma.shape = arma::vec(kVarNParams, arma::fill::value(3.0));
    in.a_prior.sigma.rate = arma::vec(kVarNParams, arma::fill::value(kRate));
    in.a_prior.initial_state.mu = arma::vec(kVarNParams, arma::fill::zeros);
    in.a_prior.initial_state.v_inv = arma::eye<arma::mat>(kVarNParams, kVarNParams);

    in.psi_prior.sigma.shape = arma::vec(kVarNPsi, arma::fill::value(3.0));
    in.psi_prior.sigma.rate = arma::vec(kVarNPsi, arma::fill::value(kRate));
    in.psi_prior.initial_state.mu = arma::vec(kVarNPsi, arma::fill::zeros);
    in.psi_prior.initial_state.v_inv = arma::eye<arma::mat>(kVarNPsi, kVarNPsi);

    in.u_sigma_prior.shape = arma::vec(kVarK, arma::fill::value(3.0));
    in.u_sigma_prior.rate = arma::vec(kVarK, arma::fill::value(0.01));

    in.initial.a = arma::mat(kVarNParams, kTT, arma::fill::value(start));
    in.initial.a_sigma_inv = arma::eye<arma::mat>(kVarNParams, kVarNParams) / kRate;
    in.initial.a_init = arma::vec(kVarNParams, arma::fill::value(start));
    in.initial.psi = arma::mat(kVarNPsi, kTT, arma::fill::value(start));
    in.initial.psi_sigma_inv = arma::eye<arma::mat>(kVarNPsi, kVarNPsi) / kRate;
    in.initial.psi_init = arma::vec(kVarNPsi, arma::fill::value(start));
    in.initial.u_omega_inv = arma::eye<arma::mat>(kVarK, kVarK) * 12.0;
    return in;
}

bayests::VarTvpGammaDraws run(const bayests::VarTvpGammaInput &in)
{
    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(kSeed);
    return bayests::VarTvpGammaSampler{}.draw_coefficients(in, reporter);
}

void var_chains_meet()
{
    std::printf("VarTvpGamma, chains started at 0 and at %g:\n", kFar);

    const bayests::VarTvpGammaDraws near = run(var_input(0.0));
    const bayests::VarTvpGammaDraws far = run(var_input(kFar));

    const arma::vec a_near = path_mean(near.a, kVarNParams);
    const arma::vec a_far = path_mean(far.a, kVarNParams);
    check("the coefficient paths agree", arma::abs(a_near - a_far).max() < kTolerance,
          largest_gap(a_near, a_far));

    // vec of the transition matrix used to simulate the data, column by column.
    const arma::vec a_true = {0.5, 0.0, 0.2, 0.3};
    check("and are where the data put them", arma::abs(a_near - a_true).max() < kTolerance,
          largest_gap(a_near, a_true));

    // The one free element of Psi, (2, 1), at position 1 of each k x k block.
    const arma::uword kk = static_cast<arma::uword>(kVarK * kVarK);
    const double psi_near = path_mean(near.psi, kk)(1);
    const double psi_far = path_mean(far.psi, kk)(1);
    check("the covariance paths agree", std::abs(psi_near - psi_far) < kTolerance,
          "|difference| = " + std::to_string(std::abs(psi_near - psi_far)));
}

// VecTvpWishart ------------------------------------------------------------

constexpr int kVecK = 2;
constexpr int kVecP = 2;    // level lags, so one lagged difference
constexpr int kVecRank = 1;
constexpr int kVecKBeta = kVecK;
constexpr int kVecNAlpha = kVecK * kVecRank;
constexpr int kVecNX = kVecK * (kVecP - 1);
constexpr int kVecNA = kVecNAlpha + kVecK * kVecNX;
constexpr int kVecNBeta = kVecKBeta * kVecRank;
constexpr double kRho = 0.999;

/// A common stochastic trend and a stationary gap, so y1 - y2 cointegrates.
bayests::TrainData vec_sample()
{
    Uniform uniform;
    arma::mat levels(kVecK, kTT + kVecP, arma::fill::zeros);
    double trend = 0.0;
    double gap = 0.0;
    for (int t = 1; t < kTT + kVecP; ++t)
    {
        trend += uniform.next();
        gap = 0.5 * gap + uniform.next();
        levels(0, t) = trend + gap;
        levels(1, t) = trend - gap;
    }

    bayests::TrainData train;
    train.y = arma::mat(kTT, kVecK);
    train.w = arma::mat(kTT, kVecKBeta);
    train.z = arma::zeros<arma::mat>(kTT * kVecK, kVecNA);
    const arma::mat diag_k = arma::eye<arma::mat>(kVecK, kVecK);
    for (int t = 0; t < kTT; ++t)
    {
        const int now = kVecP + t;
        train.y.row(t) = arma::trans(levels.col(now) - levels.col(now - 1));
        train.w.row(t) = arma::trans(levels.col(now - 1));

        // The loading columns are rebuilt from beta before they are read.
        const arma::rowvec lagged_difference = arma::trans(levels.col(now - 1) - levels.col(now - 2));
        train.z.submat(t * kVecK, kVecNAlpha, (t + 1) * kVecK - 1, kVecNA - 1) =
            arma::kron(lagged_difference, diag_k);
    }
    return train;
}

bayests::VecTvpWishartInput vec_input(const double start)
{
    bayests::VecTvpWishartInput in;
    in.spec.k = kVecK;
    in.spec.p = kVecP;
    in.spec.n = 0;
    in.spec.rank = kVecRank;
    in.spec.k_beta = kVecKBeta;
    in.spec.iterations = kIterations;
    in.spec.burnin = kBurnin;
    in.train = vec_sample();

    in.a_prior.sigma.shape = arma::vec(kVecNA, arma::fill::value(3.0));
    in.a_prior.sigma.rate = arma::vec(kVecNA, arma::fill::value(kRate));
    in.a_prior.initial_state.mu = arma::vec(kVecNA, arma::fill::zeros);
    in.a_prior.initial_state.v_inv = arma::eye<arma::mat>(kVecNA, kVecNA);

    in.beta_prior.rho = kRho;
    in.beta_prior.initial_state.mu = arma::vec(kVecNBeta, arma::fill::zeros);
    in.beta_prior.initial_state.v_inv =
        (1.0 - kRho * kRho) * arma::eye<arma::mat>(kVecNBeta, kVecNBeta);

    in.u_sigma_prior.df = kVecK;
    in.u_sigma_prior.scale = arma::eye<arma::mat>(kVecK, kVecK) * 0.1;

    // Only the coefficients on the lagged differences start far away: the
    // loadings and the cointegration vector start where both chains do, so
    // what differs between the chains is the block the fix is about.
    arma::vec a_start(kVecNA, arma::fill::zeros);
    a_start.tail(kVecNA - kVecNAlpha).fill(start);
    const arma::vec beta = {1.0, -1.0};

    in.initial.a = arma::repmat(a_start, 1, kTT);
    in.initial.a_sigma_inv = arma::eye<arma::mat>(kVecNA, kVecNA) / kRate;
    in.initial.a_init = a_start;
    in.initial.beta = arma::repmat(beta, 1, kTT);
    in.initial.beta_init = beta;
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kVecK, kVecK) * 6.0;
    return in;
}

bayests::VecTvpWishartDraws run(const bayests::VecTvpWishartInput &in)
{
    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(kSeed);
    return bayests::VecTvpWishartSampler{}.draw_coefficients(in, reporter);
}

void vec_chains_meet()
{
    std::printf("VecTvpWishart, short-run coefficients started at 0 and at %g:\n", kFar);

    const bayests::VecTvpWishartDraws near = run(vec_input(0.0));
    const bayests::VecTvpWishartDraws far = run(vec_input(kFar));

    const arma::vec gamma_near = path_mean(near.a, kVecNA).tail(kVecNA - kVecNAlpha);
    const arma::vec gamma_far = path_mean(far.a, kVecNA).tail(kVecNA - kVecNAlpha);
    check("the short-run coefficient paths agree",
          arma::abs(gamma_near - gamma_far).max() < kTolerance, largest_gap(gamma_near, gamma_far));
}

// Refusals -----------------------------------------------------------------

void singular_initial_state_precision_is_refused()
{
    std::printf("A prior precision on the state before the sample without an inverse:\n");

    check("the VAR input as built is accepted",
          !refused([] { var_input(0.0).validate(); }));
    check("a zero precision on a is refused", refused([] {
              bayests::VarTvpGammaInput in = var_input(0.0);
              in.a_prior.initial_state.v_inv.zeros();
              in.validate();
          }));
    check("a singular precision on psi is refused", refused([] {
              bayests::VarTvpGammaInput in = var_input(0.0);
              in.psi_prior.initial_state.v_inv.zeros();
              in.validate();
          }));
    check("an indefinite precision on a is refused", refused([] {
              bayests::VarTvpGammaInput in = var_input(0.0);
              in.a_prior.initial_state.v_inv(0, 0) = -1.0;
              in.validate();
          }));
    check("the VEC input as built is accepted",
          !refused([] { vec_input(0.0).validate(); }));
}

} // namespace

int main()
{
    var_chains_meet();
    vec_chains_meet();
    singular_initial_state_precision_is_refused();

    if (failures > 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
