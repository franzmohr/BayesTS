// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That the constant VECs sample the posterior of their cointegration space prior
// when the cointegration term has more rows than the model has equations.
//
// The samplers change from (alpha, beta) to A = alpha (alpha' alpha)^(-1/2) and
// B = beta (alpha' alpha)^(1/2) to draw the cointegration matrix. In those
// coordinates the prior carries |B' P_tau^-1 B|^(-(k_beta - k)/2) besides the
// normal kernel B is drawn from, a factor of one only when k_beta = k. Without it
// a model with a restricted constant, trend or unmodelled variable overstates
// |Pi|; accept_coint_draw() in src/core/models/vec_support.h puts it back.
//
// The smallest model the factor matters in has one endogenous variable and a
// constant restricted to the cointegration space: k = 1, k_beta = 2, rank 1, so
// Pi = alpha beta' has two elements and its exact posterior can be integrated
// on a grid. With P_tau = I, beta = (cos t, sin t) uniform, alpha | h ~
// N(0, 1 / (v h)) and a Wishart prior with `df` and scale S on the precision h,
// integrating h out and changing from (alpha, t) to Pi, whose Jacobian is |Pi|,
// leaves
//
//     p(Pi | y)  proportional to  |Pi|^-1 (S + v |Pi|^2 + SSR(Pi))^(-(df + T + 1) / 2).
//
// The posterior means of both elements and of |Pi| from VecNormalWishart and
// VecKlgs2010 are compared with the grid's, within five Monte Carlo standard
// errors from batch means. The same posterior without the |Pi|^-1 -- what the
// samplers drew before -- is checked to be more than ten away, so that the test
// would have failed then. The sample is short and only weakly mean reverting,
// which puts the origin, where the factor matters most, well inside the
// posterior.

#include "bayests/vec_klgs_2010.h"
#include "bayests/vec_normal_wishart.h"

#include <cmath>
#include <iostream>
#include <string>

namespace
{

constexpr int kTT = 30;          // periods
constexpr int kKBeta = 2;        // the level and the restricted constant
constexpr double kV = 1.0;       // cointegration space prior shrinkage v
constexpr double kDf = 3.0;      // Wishart degrees of freedom
constexpr double kScale = 0.1;   // Wishart scale S
constexpr int kIterations = 60000;
constexpr int kBurnin = 2000;
constexpr int kBatches = 60;
constexpr unsigned long long kSeed = 20260914ULL;

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

/// Differences, and the lagged level with a constant, of a weakly mean reverting
/// series. Deterministic, and not drawn from Armadillo's generator, which belongs
/// to the samplers.
struct Sample
{
    arma::mat y; ///< tt x 1
    arma::mat w; ///< tt x 2
};

Sample make_sample()
{
    unsigned long long state = 424242ULL;
    const auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const unsigned long long mantissa = (state >> 11) & ((1ULL << 53) - 1);
        return static_cast<double>(mantissa) / 9007199254740992.0 - 0.5;
    };

    Sample s;
    s.y = arma::mat(kTT, 1);
    s.w = arma::mat(kTT, kKBeta, arma::fill::ones);
    double level = 0.3;
    for (int t = 0; t < kTT; ++t)
    {
        const double previous = level;
        level = 0.95 * previous + 0.05 + next();
        s.w(t, 0) = previous;
        s.y(t, 0) = level - previous;
    }
    return s;
}

bayests::VarSpec make_spec()
{
    bayests::VarSpec spec;
    spec.k = 1;
    spec.p = 1;
    spec.n = 0;
    spec.n_restricted = 1;
    spec.rank = 1;
    spec.k_beta = kKBeta;
    spec.iterations = kIterations;
    spec.burnin = kBurnin;
    return spec;
}

template <typename Input> void fill_common(Input &in, const Sample &s)
{
    in.spec = make_spec();
    in.train.y = s.y;
    in.train.w = s.w;
    in.a_prior.mu = arma::vec(1, arma::fill::zeros);
    in.a_prior.v_inv = arma::mat(1, 1, arma::fill::zeros);
    in.beta_prior.v_inv = kV;
    in.beta_prior.p_tau_inv = arma::eye<arma::mat>(kKBeta, kKBeta);
    in.u_sigma_prior.df = kDf;
    in.u_sigma_prior.scale = arma::mat(1, 1, arma::fill::value(kScale));
    in.initial.a = arma::vec(1, arma::fill::value(0.1));
    in.initial.beta = arma::vec({1.0, 0.0});
    in.initial.u_sigma_inv = arma::mat(1, 1, arma::fill::ones);
}

bayests::VecNormalWishartInput make_sur(const Sample &s)
{
    bayests::VecNormalWishartInput in;
    fill_common(in, s);
    // The loading column holds (beta' w_t) for the starting beta, the lagged level.
    in.train.z = s.w.col(0);
    return in;
}

bayests::VecKlgs2010Input make_compact(const Sample &s)
{
    bayests::VecKlgs2010Input in;
    fill_common(in, s);
    return in;
}

/// Posterior means of pi_1, pi_2 and |Pi| on a grid, with or without the
/// Jacobian factor |Pi|^-1.
arma::vec exact_means(const Sample &s, bool jacobian)
{
    const arma::mat ww = arma::trans(s.w) * s.w;
    const arma::vec b_hat = arma::solve(ww, arma::trans(s.w) * s.y);
    const double ssr0 = arma::accu(arma::square(s.y - s.w * b_hat));
    // SSR(Pi) = ssr0 + u'u with u = L' (Pi - b_hat) and ww = L L'.
    const arma::mat l_inv_t = arma::inv(arma::trans(arma::mat(arma::chol(ww, "lower"))));
    const double exponent = (kDf + kTT + 1.0) / 2.0;
    const double spread = std::sqrt((kScale + ssr0) / (2.0 * exponent - 2.0));

    constexpr int kPoints = 2001;
    const double half = 30.0 * spread;
    const double step = 2.0 * half / (kPoints - 1);

    // Weights relative to the largest, accumulated in two passes to stay in range.
    double max_log = -arma::datum::inf;
    for (int pass = 0; pass < 2; ++pass)
    {
        double total = 0.0, sum1 = 0.0, sum2 = 0.0, sum_norm = 0.0;
        for (int i = 0; i < kPoints; ++i)
        {
            for (int j = 0; j < kPoints; ++j)
            {
                // Offset by half a step, so that no point sits on |Pi| = 0.
                const arma::vec u = {-half + (i + 0.5) * step, -half + (j + 0.5) * step};
                const arma::vec pi = b_hat + l_inv_t * u;
                const double norm = arma::norm(pi);
                double log_w = -exponent * std::log(kScale + kV * norm * norm + ssr0 + arma::dot(u, u));
                if (jacobian)
                {
                    log_w -= std::log(norm);
                }
                if (pass == 0)
                {
                    max_log = std::max(max_log, log_w);
                    continue;
                }
                const double w = std::exp(log_w - max_log);
                total += w;
                sum1 += w * pi(0);
                sum2 += w * pi(1);
                sum_norm += w * norm;
            }
        }
        if (pass == 1)
        {
            return arma::vec({sum1 / total, sum2 / total, sum_norm / total});
        }
    }
    return arma::vec();
}

/// The same three means from a chain, and their Monte Carlo standard errors from
/// non-overlapping batch means.
void chain_means(const arma::mat &a, const arma::mat &beta, arma::vec &means, arma::vec &se)
{
    const arma::uword draws = a.n_cols;
    arma::mat stats(3, draws);
    stats.row(0) = a.row(0) % beta.row(0);
    stats.row(1) = a.row(0) % beta.row(1);
    stats.row(2) = arma::sqrt(arma::square(stats.row(0)) + arma::square(stats.row(1)));
    means = arma::mean(stats, 1);

    const arma::uword size = draws / kBatches;
    arma::mat batch(3, kBatches);
    for (int b = 0; b < kBatches; ++b)
    {
        batch.col(b) = arma::mean(stats.cols(b * size, (b + 1) * size - 1), 1);
    }
    se = arma::stddev(batch, 0, 1) / std::sqrt(static_cast<double>(kBatches));
}

void compare(const std::string &model, const arma::mat &a, const arma::mat &beta,
             const arma::vec &exact, const arma::vec &without)
{
    arma::vec means, se;
    chain_means(a, beta, means, se);
    const char *names[] = {"pi_1", "pi_2", "|Pi|"};
    for (int i = 0; i < 3; ++i)
    {
        const double z_exact = (means(i) - exact(i)) / se(i);
        const double z_without = (means(i) - without(i)) / se(i);
        check(model + ": posterior mean of " + names[i] + " is the exact one",
              std::abs(z_exact) < 5.0,
              "chain " + std::to_string(means(i)) + ", exact " + std::to_string(exact(i)) +
                  ", z " + std::to_string(z_exact));
        check(model + ": and not the one without the Jacobian", std::abs(z_without) > 10.0,
              "without the Jacobian " + std::to_string(without(i)) + ", z " + std::to_string(z_without));
    }
}

} // namespace

int main()
{
    const Sample sample = make_sample();
    bayests::NullReporter reporter;

    const arma::vec exact = exact_means(sample, true);
    const arma::vec without = exact_means(sample, false);

    std::cout << "VecNormalWishart, k = 1, k_beta = 2:\n";
    arma::arma_rng::set_seed(kSeed);
    const bayests::VecNormalWishartDraws sur =
        bayests::VecNormalWishartSampler{}.draw_coefficients(make_sur(sample), reporter);
    compare("VecNormalWishart", sur.a, sur.beta, exact, without);

    std::cout << "VecKlgs2010, k = 1, k_beta = 2:\n";
    arma::arma_rng::set_seed(kSeed + 1);
    const bayests::VecKlgs2010Draws compact =
        bayests::VecKlgs2010Sampler{}.draw_coefficients(make_compact(sample), reporter);
    compare("VecKlgs2010", compact.a, compact.beta, exact, without);

    std::cout << (failures == 0 ? "all as expected\n" : "SOMETHING IS WRONG\n");
    return failures == 0 ? 0 : 1;
}
