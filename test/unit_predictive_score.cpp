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
#include "bayests/var_normal_stochvol.h"
#include "bayests/var_normal_wishart.h"
#include "bayests/var_tvp_wishart.h"

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


// --- the models whose states move --------------------------------------------
//
// A drifting model is scored under the state each draw reaches by stepping
// forward, so the identity above cannot be used as it stands: the scored
// periods are not under the states the sample had. It comes back on a posterior
// whose path happens to stand still. Then the sample's last state is the state
// of every period, holding it reproduces them, and stepping it with a variance
// of zero reproduces them too -- which is what says the stepping is wired to
// the right block and arrives at the right period.

/// The VAR(1) above as a time-varying model whose coefficients do not in fact
/// vary: one period of `a` repeated over the sample.
bayests::VarTvpWishartInput tvp_input()
{
    bayests::VarTvpWishartInput input;
    const VarNormalWishartInput shape = sample_input();
    input.spec = shape.spec;
    input.train = shape.train;
    return input;
}

bayests::VarTvpWishartDraws tvp_draws(arma::uword n_params, arma::uword periods,
                                      double state_variance)
{
    const VarNormalWishartDraws flat = sample_draws(n_params);
    bayests::VarTvpWishartDraws draws;
    draws.u_sigma_inv = flat.u_sigma_inv;
    draws.a = arma::repmat(flat.a, periods, 1);
    draws.a_sigma = arma::mat(n_params, kDraws, arma::fill::value(state_variance));
    return draws;
}

void test_tvp_standing_still(bool hold)
{
    bayests::VarTvpWishartInput input = tvp_input();
    const arma::uword n_params = input.train.z.n_cols;
    const bayests::VarTvpWishartSampler sampler;

    // The sample under a path that repeats one set of coefficients.
    const arma::mat full = sampler.log_likelihood(input, tvp_draws(n_params, kPeriods, 0.0));

    const arma::uword periods = 3;
    input.spec.h = static_cast<int>(periods);
    input.spec.forecast_states =
        hold ? bayests::ForecastStates::hold : bayests::ForecastStates::simulate;
    input.test.y = input.train.y.tail_rows(periods);
    input.forecast.x = input.train.x.tail_rows(periods);

    // One period of those coefficients is what a forecast reader hands over.
    const bayests::VarTvpWishartDraws forecast_draws = tvp_draws(n_params, 1, 0.0);
    const arma::mat score = sampler.predictive_log_density(input, forecast_draws);

    const std::string label = hold ? "held" : "stepped by nothing";
    check(arma::approx_equal(score, full.tail_cols(periods), "absdiff", 1e-12),
          "a path that stands still is scored as the likelihood of those periods (" + label + ")");
}

/// And a path that does move is scored differently, which says the steps are
/// taken rather than merely allowed for.
void test_tvp_moves()
{
    bayests::VarTvpWishartInput input = tvp_input();
    const arma::uword n_params = input.train.z.n_cols;
    const arma::uword periods = 3;

    input.spec.h = static_cast<int>(periods);
    input.test.y = input.train.y.tail_rows(periods);
    input.forecast.x = input.train.x.tail_rows(periods);

    const bayests::VarTvpWishartSampler sampler;

    input.spec.forecast_states = bayests::ForecastStates::hold;
    const arma::mat held = sampler.predictive_log_density(input, tvp_draws(n_params, 1, 0.0));

    arma::arma_rng::set_seed(20260920);
    input.spec.forecast_states = bayests::ForecastStates::simulate;
    const arma::mat moved = sampler.predictive_log_density(input, tvp_draws(n_params, 1, 0.05));

    check(moved.n_rows == held.n_rows && moved.n_cols == held.n_cols,
          "a drifting score has the shape of a held one");
    check(!arma::approx_equal(moved, held, "absdiff", 1e-8),
          "and differs from it, the coefficients having moved");
}

/// How far the state gets, checked on the walk itself rather than through a
/// density, where three draws of a log likelihood are too few to say anything
/// about a direction.
void test_carry_state_forward()
{
    const arma::uword n_state = 4;
    const arma::uword draws = 2;
    const arma::uword periods = 3;
    const arma::mat last(n_state, draws, arma::fill::randu);
    const arma::mat none(n_state, draws, arma::fill::zeros);
    const arma::mat some(n_state, draws, arma::fill::value(0.25));

    const auto block = [&](const arma::mat &path, arma::uword i) {
        return arma::mat(path.rows(i * n_state, (i + 1) * n_state - 1));
    };

    const arma::mat held = bayests::core::carry_state_forward(last, none, arma::mat(), periods,
                                                              false, "the state");
    check(held.n_rows == n_state * periods && held.n_cols == draws,
          "the path is one block of the state per period");
    check(arma::approx_equal(block(held, 0), last, "absdiff", 0.0) &&
              arma::approx_equal(block(held, periods - 1), last, "absdiff", 0.0),
          "holding repeats the sample's last state at every period");

    const arma::mat still = bayests::core::carry_state_forward(last, none, arma::mat(), periods,
                                                               true, "the state");
    check(arma::approx_equal(still, held, "absdiff", 0.0),
          "and a step of variance zero is the same as holding");

    // Everything switched off by selection stays where it is, whatever the
    // variance says: BVS stores an excluded coefficient as zero in every period
    // and a walk away from zero would put the regressor back.
    const arma::mat off(n_state, draws, arma::fill::zeros);
    const arma::mat masked = bayests::core::carry_state_forward(last, some, off, periods, true,
                                                                "the state");
    check(arma::approx_equal(masked, held, "absdiff", 0.0),
          "and a state selection left out does not move either");

    arma::arma_rng::set_seed(20260920);
    const arma::mat walked = bayests::core::carry_state_forward(last, some, arma::mat(), periods,
                                                                true, "the state");
    check(!arma::approx_equal(block(walked, 0), last, "absdiff", 1e-10),
          "the first period is already one step from the sample");
    check(!arma::approx_equal(block(walked, 1), block(walked, 0), "absdiff", 1e-10) &&
              !arma::approx_equal(block(walked, 2), block(walked, 1), "absdiff", 1e-10),
          "and every period after it takes a step of its own");
}

/// The stochastic volatility model, whose log variances are the random walk.
void test_stochvol_standing_still()
{
    bayests::VarNormalStochvolInput input;
    const VarNormalWishartInput shape = sample_input();
    input.spec = shape.spec;
    input.train = shape.train;

    const arma::uword n_params = input.train.z.n_cols;
    const arma::uword periods = 3;

    bayests::VarNormalStochvolDraws draws;
    draws.a = sample_draws(n_params).a;
    draws.u_omega_inv.set_size(kK, kDraws);
    for (arma::uword j = 0; j < kDraws; j++)
    {
        draws.u_omega_inv.col(j) = arma::vec({1.5 + 0.25 * static_cast<double>(j), 2.25});
    }
    draws.h_sigma = arma::mat(kK, kDraws, arma::fill::zeros);

    // The sample under that volatility, held at every period.
    bayests::VarNormalStochvolDraws over_sample = draws;
    over_sample.u_omega_inv = arma::repmat(draws.u_omega_inv, kPeriods, 1);
    over_sample.u_sigma_inv = bayests::core::precision_path(over_sample.u_omega_inv, arma::mat(),
                                                           kK, kPeriods);
    const bayests::VarNormalStochvolSampler sampler;
    const arma::mat full = sampler.log_likelihood(input, over_sample);

    input.spec.h = static_cast<int>(periods);
    input.test.y = input.train.y.tail_rows(periods);
    input.forecast.x = input.train.x.tail_rows(periods);
    draws.u_sigma_inv = bayests::core::precision_path(draws.u_omega_inv, arma::mat(), kK, 1);

    const arma::mat score = sampler.predictive_log_density(input, draws);
    check(arma::approx_equal(score, full.tail_cols(periods), "absdiff", 1e-12),
          "a volatility that stands still is scored as the likelihood of those periods");
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
        test_tvp_standing_still(true);
        test_tvp_standing_still(false);
        test_tvp_moves();
        test_carry_state_forward();
        test_stochvol_standing_still();
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
