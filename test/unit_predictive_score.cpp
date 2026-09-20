// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_predictive_score.cpp
/// @brief Checks what a forecast is scored as, against the likelihood itself.
///
/// The score of a horizon is the log density of the realised observation under
/// each draw, conditional on the realised observations before it. Written that
/// way it has an identity to be pinned against: hand a model back a stretch of
/// its own sample as the values its horizon realised, with the regressors those
/// periods had, and the score must be the pointwise log likelihood of exactly
/// those periods. Nothing about the expression is then taken on trust -- it is
/// the same numbers the likelihood produces or it is wrong.
///
/// That identity is also what says the lag substitution happened. The
/// regressors of the second scored period carry the first one's realised
/// values, so a scorer that left the lag blocks as the forecast input had them,
/// or filled them from somewhere else, would miss the sample's own regressors
/// and the two would part company.

#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_wishart.h"

#include "core/models/predictive_score.h"

#include <iostream>
#include <string>

using bayests::VarNormalWishartDraws;
using bayests::VarNormalWishartInput;
using bayests::VarNormalWishartSampler;
using bayests::core::realised_regressors;

namespace
{

int failures = 0;

void check(bool condition, const std::string &what)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
        return;
    }
    std::cout << "ok: " << what << '\n';
}

constexpr int kK = 2;
constexpr int kP = 1;
constexpr arma::uword kPeriods = 8;
constexpr arma::uword kDraws = 3;

/// A VAR(1) with a constant, laid out the way a model file's would be: `y` one
/// row per period, `x` the lag of it beside a column of ones, `z` the SUR
/// spelling of `x`.
VarNormalWishartInput sample_input()
{
    VarNormalWishartInput input;
    input.spec.k = kK;
    input.spec.p = kP;
    input.spec.n = 1;

    // Nine periods of two series, distinct and not a straight line, so that a
    // lag written into the wrong row cannot pass unnoticed.
    arma::mat levels(kPeriods + 1, kK);
    for (arma::uword t = 0; t < levels.n_rows; t++)
    {
        levels(t, 0) = 1.0 + 0.5 * static_cast<double>(t) + 0.25 * static_cast<double>(t * t % 5);
        levels(t, 1) = -0.75 + 0.3 * static_cast<double>(t % 4) - 0.1 * static_cast<double>(t);
    }

    input.train.y = levels.tail_rows(kPeriods);

    arma::mat x(kPeriods, kK + 1);
    x.cols(0, kK - 1) = levels.head_rows(kPeriods);
    x.col(kK).ones();
    input.train.x = x;
    input.train.z = arma::kron(x, arma::eye<arma::mat>(kK, kK));

    return input;
}

VarNormalWishartDraws sample_draws(const arma::uword n_params)
{
    VarNormalWishartDraws draws;
    draws.a.set_size(n_params, kDraws);
    for (arma::uword j = 0; j < kDraws; j++)
    {
        for (arma::uword i = 0; i < n_params; i++)
        {
            draws.a(i, j) = 0.1 * static_cast<double>(i + 1) - 0.05 * static_cast<double>(j);
        }
    }

    // One vectorised precision per draw, each symmetric positive definite.
    draws.u_sigma_inv.set_size(kK * kK, kDraws);
    for (arma::uword j = 0; j < kDraws; j++)
    {
        arma::mat precision = {{2.0 + 0.5 * static_cast<double>(j), 0.4}, {0.4, 1.5}};
        draws.u_sigma_inv.col(j) = arma::vectorise(precision);
    }
    return draws;
}

/// The last `periods` of the sample, handed back as what the horizon realised.
void score_the_sample(arma::uword periods)
{
    VarNormalWishartInput input = sample_input();
    const VarNormalWishartDraws draws = sample_draws(input.train.z.n_cols);
    const VarNormalWishartSampler sampler;

    const arma::mat full = sampler.log_likelihood(input, draws);

    input.spec.h = static_cast<int>(periods);
    input.test.y = input.train.y.tail_rows(periods);
    input.forecast.x = input.train.x.tail_rows(periods);

    const arma::mat score = sampler.predictive_log_density(input, draws);
    const arma::mat expected = full.tail_cols(periods);

    const std::string label = std::to_string(periods) + " scored period(s)";
    check(score.n_rows == kDraws && score.n_cols == periods,
          label + ": the score is draws by scored periods");
    check(arma::approx_equal(score, expected, "absdiff", 1e-12),
          label + ": and is the pointwise log likelihood of those periods");
}

/// The regressors on their own, where the substitution is visible rather than
/// inferred from an identity.
void test_realised_regressors()
{
    const int k = 2;
    const int p = 2;

    // What a forecast would be driven by: the lag blocks of the first row are
    // the end of the sample and the later ones are placeholders a forecast
    // overwrites as it goes.
    arma::mat forecast_x = {{10.0, 11.0, 20.0, 21.0, 1.0},
                            {-1.0, -1.0, 10.0, 11.0, 1.0},
                            {-2.0, -2.0, -1.0, -1.0, 1.0}};
    const arma::mat realised = {{100.0, 101.0}, {200.0, 201.0}, {300.0, 301.0}};

    const arma::mat x = realised_regressors(forecast_x, realised, k, p);

    check(arma::approx_equal(x.row(0), forecast_x.row(0), "absdiff", 0.0),
          "the first scored period keeps the regressors it was given");
    check(x(1, 0) == 100.0 && x(1, 1) == 101.0,
          "the second period's first lag is what the first period realised");
    // Period T+2's second lag is y_T, which is what the caller put in that row
    // and what the first row carries as its first lag. Not row 0's second lag,
    // y_{T-1}, which belongs a period further back.
    check(x(1, 2) == 10.0 && x(1, 3) == 11.0,
          "and its second lag, which reaches into the sample, is left alone");
    check(x(2, 0) == 200.0 && x(2, 1) == 201.0 && x(2, 2) == 100.0 && x(2, 3) == 101.0,
          "the third period's two lags are the two periods before it");
    check(x(0, 4) == 1.0 && x(1, 4) == 1.0 && x(2, 4) == 1.0,
          "and the deterministic column is untouched throughout");
}

/// A file may forecast further than it realised. The horizons it did not
/// realise are dropped rather than scored against nothing.
void test_short_horizon()
{
    VarNormalWishartInput input = sample_input();
    const VarNormalWishartDraws draws = sample_draws(input.train.z.n_cols);

    input.spec.h = 4;
    input.test.y = input.train.y.tail_rows(2);
    input.forecast.x = input.train.x.tail_rows(4);

    const arma::mat score = VarNormalWishartSampler{}.predictive_log_density(input, draws);
    check(score.n_cols == 2, "only the realised horizons are scored");
}

bool throws(const VarNormalWishartInput &input, const VarNormalWishartDraws &draws)
{
    try
    {
        VarNormalWishartSampler{}.predictive_log_density(input, draws);
    }
    catch (const std::exception &)
    {
        return true;
    }
    return false;
}

void test_refusals()
{
    VarNormalWishartInput input = sample_input();
    const VarNormalWishartDraws draws = sample_draws(input.train.z.n_cols);

    check(throws(input, draws), "a model with no realised values is refused");

    input.spec.h = 2;
    input.test.y = input.train.y.tail_rows(2);
    input.forecast.x = input.train.x.tail_rows(2);
    check(!throws(input, draws), "and is scored once it has them");

    VarNormalWishartInput structural = input;
    structural.spec.structural = true;
    check(throws(structural, draws), "a structural model is refused");

    VarNormalWishartInput unfilled = input;
    unfilled.forecast.x = input.train.x.tail_rows(1);
    check(throws(unfilled, draws),
          "so is a file that realised more periods than it holds regressors for");
}

/// The second model that can be scored today, on the same identity. Its
/// precision is a diagonal rather than a full matrix, so the two expressions
/// differ and the identity has to be checked against each.
void test_gamma_matches_its_own_likelihood()
{
    bayests::VarNormalGammaInput input;
    input.spec.k = kK;
    input.spec.p = kP;
    input.spec.n = 1;

    const VarNormalWishartInput shape = sample_input();
    input.train = shape.train;

    bayests::VarNormalGammaDraws draws;
    draws.a = sample_draws(input.train.z.n_cols).a;
    draws.u_sigma_inv.set_size(kK * kK, kDraws);
    draws.u_omega_inv.set_size(kK, kDraws);
    for (arma::uword j = 0; j < kDraws; j++)
    {
        const arma::vec diagonal = {1.5 + 0.25 * static_cast<double>(j), 2.25};
        draws.u_omega_inv.col(j) = diagonal;
        draws.u_sigma_inv.col(j) = arma::vectorise(arma::diagmat(diagonal));
    }

    const bayests::VarNormalGammaSampler sampler;
    const arma::mat full = sampler.log_likelihood(input, draws);

    input.spec.h = 3;
    input.test.y = input.train.y.tail_rows(3);
    input.forecast.x = input.train.x.tail_rows(3);

    const arma::mat score = sampler.predictive_log_density(input, draws);
    check(arma::approx_equal(score, full.tail_cols(3), "absdiff", 1e-12),
          "VarNormalGamma is scored as its own likelihood too");
}

} // namespace

int main()
{
    try
    {
        test_realised_regressors();
        score_the_sample(1);
        score_the_sample(3);
        test_short_horizon();
        test_refusals();
        test_gamma_matches_its_own_likelihood();
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: threw: " << e.what() << '\n';
        ++failures;
    }

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "all checks passed\n";
    return 0;
}
