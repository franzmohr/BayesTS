// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_var_mixed_frequency.cpp
/// @brief VarNormalWishart on a panel not observed whole: an identity, a
///        recovery and the forecast's starting point.
///
/// Each sweep of the sampler now starts by completing the panel -- drawing what
/// was not observed given the current coefficients and precision -- and then
/// runs the sweep of a complete panel. Four statements pin that down.
///
/// The identity. A panel pinned whole leaves nothing to draw, so the completion
/// consumes no random numbers and rebuilds the regressors the file already has:
/// the chain has to agree bit for bit with the same panel run without
/// constraints, and the log likelihood, now the density of what was observed
/// computed by a filter rather than term by term, has to agree to rounding.
/// That is what says the completion step adds nothing to a model it has
/// nothing to complete.
///
/// The recovery. A monthly VAR is simulated and one of its three series kept
/// only as quarterly averages, with the ragged edge a second series has in its
/// last two months. The chain has to satisfy every quarterly average exactly
/// in every draw, recover the coefficients, and put the posterior mean of the
/// hidden monthly series closer to the truth than the step function a
/// quarterly figure repeated three times is. The margins are read off the run,
/// several times what a seed moves.
///
/// The forecast starts from each draw's own completed panel, not from what the
/// host wrote into /data/forecast/x, which for an entry nothing observed is a
/// placeholder.
///
/// A volatility break and Student t errors read a panel not observed whole too:
/// pinned whole it is the complete panel's chain bit for bit, its likelihood
/// the complete panel's under a break and the normal given the mixing scales
/// under t errors, its score the recursion's; and a quarterly series is
/// recovered beside a break, every average holding.
///
/// Beyond those: a score from a completed panel or against a horizon realised
/// in part agrees with the recursion where the two describe the same thing; a
/// scenario holds in every draw of a conditional forecast, whose mean is the
/// conditioned one; a series read with an error has that error's variance
/// recovered and is smoothed by the model. And what is refused is refused.

#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_wishart.h"
#include "core/algorithms/constrained_var_path.h"
#include "core/models/completion_support.h"
#include "core/models/volatility_break_support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-68s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_below(const std::string &what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-68s %s   %.3e < %.3e\n", what.c_str(), ok ? "ok" : "FAILED", got, bound);
    if (!ok)
    {
        failures++;
    }
}

constexpr arma::uword kK = 3;
constexpr arma::uword kNx = kK + 1; // one lag and an intercept

/// A monthly VAR(1) with an intercept, and the regressors it implies.
struct Sample
{
    arma::mat y;     // tt x k, the truth
    arma::vec lag0;  // the period before the sample
    arma::mat a;     // k x n_x, the truth: [A_1 c]
    arma::mat sigma; // the truth
};

Sample simulate(const arma::uword tt)
{
    Sample s;
    // The third series -- the one kept quarterly in the recovery -- moves with
    // the first, through its lag and through the error, so the monthly series
    // carry information about its months: otherwise the best monthly path would
    // be the quarterly figure repeated, and no sampler could beat it.
    s.a = {{0.5, 0.1, 0.0, 0.1}, {0.0, 0.4, 0.2, -0.1}, {0.4, 0.1, 0.4, 0.2}};
    s.sigma = 0.5 * arma::mat{{1.0, 0.3, 0.7}, {0.3, 1.0, 0.2}, {0.7, 0.2, 1.0}};
    const arma::mat root = arma::chol(s.sigma, "lower");

    arma::arma_rng::set_seed(4242);
    arma::vec previous(kK, arma::fill::zeros);
    for (int burn = 0; burn < 50; burn++)
    {
        previous = s.a.cols(0, kK - 1) * previous + s.a.col(kK) + root * arma::randn<arma::vec>(kK);
    }
    s.lag0 = previous;
    s.y.set_size(tt, kK);
    for (arma::uword t = 0; t < tt; t++)
    {
        previous = s.a.cols(0, kK - 1) * previous + s.a.col(kK) + root * arma::randn<arma::vec>(kK);
        s.y.row(t) = previous.t();
    }
    return s;
}

/// kron(x, I_k), x a row per period of [y_{t-1}' 1], from the panel as given.
arma::mat regressors(const arma::mat &y, const arma::vec &lag0)
{
    arma::mat x(y.n_rows, kNx);
    x.col(kK).ones();
    for (arma::uword t = 0; t < y.n_rows; t++)
    {
        x.submat(t, 0, t, kK - 1) = t == 0 ? arma::rowvec(lag0.t()) : arma::rowvec(y.row(t - 1));
    }
    return arma::kron(x, arma::eye<arma::mat>(kK, kK));
}

struct RowBuilder
{
    std::vector<double> value, weight;
    std::vector<arma::uword> group, row, period, variable;

    void add(const double v, const std::vector<std::pair<arma::uword, arma::uword>> &entries,
             const double w, const arma::uword g = 0)
    {
        const arma::uword r = value.size();
        value.push_back(v);
        group.push_back(g);
        for (const auto &e : entries)
        {
            row.push_back(r);
            period.push_back(e.first);
            variable.push_back(e.second);
            weight.push_back(w);
        }
    }

    bayests::Constraints build() const
    {
        bayests::Constraints c;
        c.value = arma::vec(value);
        c.weight = arma::vec(weight);
        c.group = arma::conv_to<arma::uvec>::from(group);
        c.row = arma::conv_to<arma::uvec>::from(row);
        c.period = arma::conv_to<arma::uvec>::from(period);
        c.variable = arma::conv_to<arma::uvec>::from(variable);
        return c;
    }
};

bayests::VarNormalWishartInput input_for(const arma::mat &y_file, const arma::vec &lag0,
                                         const int iterations, const int burnin)
{
    bayests::VarNormalWishartInput input;
    input.spec.k = static_cast<int>(kK);
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.train.y = y_file;
    input.train.z = regressors(y_file, lag0);
    input.a_prior.mu = arma::vec(kK * kNx, arma::fill::zeros);
    input.a_prior.v_inv = 0.1 * arma::eye<arma::mat>(kK * kNx, kK * kNx);
    input.u_sigma_prior.df = static_cast<int>(kK) + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    input.initial.a = arma::vec(kK * kNx, arma::fill::zeros);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return input;
}

void test_pinned_whole_is_the_complete_panel()
{
    std::printf("a panel pinned whole is the complete panel\n");
    const Sample s = simulate(60);
    bayests::NullReporter reporter;
    const bayests::VarNormalWishartSampler sampler;

    bayests::VarNormalWishartInput plain = input_for(s.y, s.lag0, 200, 50);
    bayests::VarNormalWishartInput pinned = plain;
    RowBuilder rows;
    for (arma::uword t = 0; t < s.y.n_rows; t++)
    {
        for (arma::uword i = 0; i < kK; i++)
        {
            rows.add(s.y(t, i), {{t, i}}, 1.0);
        }
    }
    pinned.train.constraints = rows.build();

    arma::arma_rng::set_seed(99);
    const auto a = sampler.draw_coefficients(plain, reporter);
    arma::arma_rng::set_seed(99);
    const auto b = sampler.draw_coefficients(pinned, reporter);

    check("coefficient draws agree bit for bit", arma::approx_equal(a.a, b.a, "absdiff", 0.0));
    check("precision draws agree bit for bit",
          arma::approx_equal(a.u_sigma_inv, b.u_sigma_inv, "absdiff", 0.0));
    check("no completed panel without constraints", !a.has_y());
    check("and the completed panel with them is the data in every draw",
          b.has_y() && arma::approx_equal(b.y, arma::repmat(arma::vectorise(s.y.t()), 1, b.y.n_cols),
                                          "absdiff", 0.0));

    const arma::mat ll_plain = sampler.log_likelihood(plain, a);
    const arma::mat ll_pinned = sampler.log_likelihood(pinned, b);
    check_below("the log likelihood agrees to rounding",
                arma::abs(ll_plain - ll_pinned).max() / arma::abs(ll_plain).max(), 1e-12);
}

void test_quarterly_series_is_recovered()
{
    std::printf("a quarterly series inside a monthly VAR\n");
    const arma::uword tt = 240;
    const Sample s = simulate(tt);

    // What the host would write: the monthly values where they were observed,
    // zeros where they were not -- the third series throughout, the second in
    // its last two months.
    arma::mat y_file = s.y;
    y_file.col(2).zeros();
    y_file(tt - 1, 1) = 0.0;
    y_file(tt - 2, 1) = 0.0;

    RowBuilder rows;
    for (arma::uword t = 0; t < tt; t++)
    {
        rows.add(s.y(t, 0), {{t, 0}}, 1.0);
        if (t < tt - 2)
        {
            rows.add(s.y(t, 1), {{t, 1}}, 1.0);
        }
    }
    arma::vec quarterly(tt / 3);
    for (arma::uword q = 0; q < tt / 3; q++)
    {
        quarterly(q) = arma::mean(s.y.col(2).subvec(3 * q, 3 * q + 2));
        rows.add(quarterly(q), {{3 * q, 2}, {3 * q + 1, 2}, {3 * q + 2, 2}}, 1.0 / 3.0);
    }

    bayests::VarNormalWishartInput input = input_for(y_file, s.lag0, 3000, 1000);
    input.train.constraints = rows.build();

    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(2026);
    const auto draws = bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);

    // Every quarterly average holds, and every pinned value, in every draw.
    double worst = 0.0;
    for (arma::uword d = 0; d < draws.y.n_cols; d++)
    {
        const arma::mat path = arma::reshape(draws.y.col(d), kK, tt);
        for (arma::uword q = 0; q < tt / 3; q++)
        {
            worst = std::max(worst, std::abs(arma::mean(path.row(2).subvec(3 * q, 3 * q + 2)) -
                                             quarterly(q)));
        }
        worst = std::max(worst, arma::abs(path.row(0).t() - s.y.col(0)).max());
    }
    check_below("every quarterly average and every pinned value holds in every draw", worst, 1e-10);

    // Against the posterior's own spread: a coefficient on a series seen only
    // quarterly is less well determined than one on a monthly series, and a
    // single bound would be loose for the one or tight for the other.
    const arma::mat a_mean = arma::reshape(arma::mean(draws.a, 1), kK, kNx);
    const arma::mat a_sd = arma::reshape(arma::stddev(draws.a, 0, 1), kK, kNx);
    const arma::mat z_scores = arma::abs(a_mean - s.a) / a_sd;
    std::printf("    posterior mean of [A_1 c] against the truth, in posterior sds:\n");
    for (arma::uword i = 0; i < kK; i++)
    {
        std::printf("     ");
        for (arma::uword j = 0; j < kNx; j++)
        {
            std::printf(" %6.3f (%.2f sd)", a_mean(i, j), z_scores(i, j));
        }
        std::printf("\n");
    }
    check_below("every coefficient within four posterior sds of the truth", z_scores.max(), 4.0);

    // The same sample observed whole. Where the two posteriors land on the same
    // side of the truth, it is the sample that put them there, not the
    // completion; and the posterior from quarterly figures has to be centred
    // near the one from the monthly values they summarise, within its own,
    // wider, spread.
    bayests::VarNormalWishartInput whole = input_for(s.y, s.lag0, 3000, 1000);
    arma::arma_rng::set_seed(2026);
    const auto whole_draws = bayests::VarNormalWishartSampler{}.draw_coefficients(whole, reporter);
    const arma::mat whole_mean = arma::reshape(arma::mean(whole_draws.a, 1), kK, kNx);
    const arma::mat whole_z =
        (whole_mean - s.a) / arma::reshape(arma::stddev(whole_draws.a, 0, 1), kK, kNx);
    std::printf("    the same sample observed whole, against the truth in its posterior sds:\n");
    for (arma::uword i = 0; i < kK; i++)
    {
        std::printf("     ");
        for (arma::uword j = 0; j < kNx; j++)
        {
            std::printf(" %6.3f (%+.2f sd)", whole_mean(i, j), whole_z(i, j));
        }
        std::printf("\n");
    }
    check_below("the quarterly posterior is centred within three of its sds of the monthly one",
                (arma::abs(a_mean - whole_mean) / a_sd).max(), 3.0);

    const arma::mat path_mean = arma::reshape(arma::mean(draws.y, 1), kK, tt);
    const arma::vec hidden = s.y.col(2);
    arma::vec step(tt);
    for (arma::uword t = 0; t < tt; t++)
    {
        step(t) = quarterly(t / 3);
    }
    const double rmse_model = std::sqrt(arma::mean(arma::square(path_mean.row(2).t() - hidden)));
    const double rmse_step = std::sqrt(arma::mean(arma::square(step - hidden)));
    std::printf("    hidden monthly series, RMSE: posterior mean %.3f, quarterly step %.3f\n",
                rmse_model, rmse_step);
    check_below("the monthly path beats the quarterly figure repeated",
                rmse_model / rmse_step, 0.9);

    const double edge = std::abs(arma::mean(path_mean.row(1).tail(2).t() - s.y.col(1).tail(2)));
    std::printf("    ragged edge, mean absolute error of the last two months: %.3f\n", edge);
}

void test_forecast_starts_from_the_completed_panel()
{
    std::printf("the forecast starts from each draw's completed panel\n");
    const arma::uword tt = 12;
    const Sample s = simulate(tt);
    arma::mat y_file = s.y;
    y_file(tt - 1, 2) = 0.0;

    RowBuilder rows;
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < kK; i++)
        {
            if (!(t == tt - 1 && i == 2))
            {
                rows.add(s.y(t, i), {{t, i}}, 1.0);
            }
        }
    }
    bayests::VarNormalWishartInput input = input_for(y_file, s.lag0, 1, 0);
    input.train.constraints = rows.build();
    input.spec.h = 1;
    input.forecast.x = arma::mat(1, kNx, arma::fill::zeros); // a placeholder lag block
    input.forecast.x(0, kK) = 1.0;

    // One draw: the true coefficients, a precision so large the error vanishes,
    // and a completed panel whose last hidden value is not the placeholder.
    bayests::VarNormalWishartDraws draws;
    draws.a = arma::vectorise(s.a);
    draws.u_sigma_inv = arma::vectorise(1e14 * arma::eye<arma::mat>(kK, kK));
    arma::mat path = s.y.t();
    path(2, tt - 1) = 3.25;
    draws.y = arma::vectorise(path);

    bayests::NullReporter reporter;
    const auto fcst = bayests::VarNormalWishartSampler{}.forecast(input, draws, reporter);
    const arma::vec want = s.a.cols(0, kK - 1) * path.col(tt - 1) + s.a.col(kK);
    check_below("one step ahead is A_1 times the completed last period plus c",
                arma::abs(fcst.values.col(0) - want).max(), 1e-5);

    draws.y.reset();
    bool refused = false;
    try
    {
        bayests::VarNormalWishartSampler{}.forecast(input, draws, reporter);
    }
    catch (const std::invalid_argument &)
    {
        refused = true;
    }
    check("and without the completed panel it refuses rather than use the placeholder", refused);
}

bool validate_refuses(const bayests::VarNormalWishartInput &input, const std::string &contains)
{
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &e)
    {
        return std::string(e.what()).find(contains) != std::string::npos;
    }
    return false;
}

/// A forecast input over a horizon of `h`: the intercept filled, the lag block
/// of the first period the sample's last period, as a host writes one.
void set_horizon(bayests::VarNormalWishartInput &input, const arma::mat &sample, const int h)
{
    input.spec.h = h;
    input.forecast.x = arma::mat(static_cast<arma::uword>(h), kNx, arma::fill::zeros);
    input.forecast.x.col(kK).ones();
    input.forecast.x.submat(0, 0, 0, kK - 1) = sample.row(sample.n_rows - 1);
}

RowBuilder pin_all(const arma::mat &y)
{
    RowBuilder rows;
    for (arma::uword t = 0; t < y.n_rows; t++)
    {
        for (arma::uword i = 0; i < kK; i++)
        {
            rows.add(y(t, i), {{t, i}}, 1.0);
        }
    }
    return rows;
}

void test_scores()
{
    std::printf("scores\n");
    const Sample s = simulate(64);
    const arma::mat sample = s.y.rows(0, 59);
    const arma::mat realised = s.y.rows(60, 63);
    bayests::NullReporter reporter;
    const bayests::VarNormalWishartSampler sampler;

    bayests::VarNormalWishartInput plain = input_for(sample, s.lag0, 100, 20);
    set_horizon(plain, sample, 4);
    plain.test.y = realised;
    arma::arma_rng::set_seed(5);
    const auto draws = sampler.draw_coefficients(input_for(sample, s.lag0, 100, 20), reporter);
    const arma::mat old_score = sampler.predictive_log_density(plain, draws);

    // A horizon realised whole, written as constraints pinning every entry,
    // goes through the completion step's filter instead of the recursion.
    bayests::VarNormalWishartInput pinned_horizon = plain;
    pinned_horizon.test.constraints = pin_all(realised).build();
    const arma::mat new_score = sampler.predictive_log_density(pinned_horizon, draws);
    check_below("a horizon pinned whole scores as the recursion does",
                arma::abs(new_score - old_score).max() / arma::abs(old_score).max(), 1e-12);

    // A sample pinned whole scores from each draw's completed panel, which is
    // the sample: the same score again.
    bayests::VarNormalWishartInput pinned_sample = plain;
    pinned_sample.train.constraints = pin_all(sample).build();
    bayests::VarNormalWishartInput estimate = input_for(sample, s.lag0, 100, 20);
    estimate.train.constraints = pinned_sample.train.constraints;
    arma::arma_rng::set_seed(5);
    const auto completed = sampler.draw_coefficients(estimate, reporter);
    const arma::mat from_panel = sampler.predictive_log_density(pinned_sample, completed);
    check_below("a sample pinned whole scores from its completed panel as from the data",
                arma::abs(from_panel - old_score).max() / arma::abs(old_score).max(), 1e-12);

    // A horizon realised in part: the first series throughout, the third as the
    // average of the first three periods, the second not at all. What the
    // horizon holds where nothing was realised is never read.
    RowBuilder part;
    for (arma::uword t = 0; t < 4; t++)
    {
        part.add(realised(t, 0), {{t, 0}}, 1.0);
    }
    part.add(arma::mean(realised.col(2).head(3)), {{0, 2}, {1, 2}, {2, 2}}, 1.0 / 3.0);
    bayests::VarNormalWishartInput partial = plain;
    partial.test.constraints = part.build();
    const arma::mat partial_score = sampler.predictive_log_density(partial, draws);
    partial.test.y.col(1).fill(123.0);
    partial.test.y.col(2).fill(-45.0);
    check("a horizon realised in part never reads what was not realised",
          arma::approx_equal(sampler.predictive_log_density(partial, draws), partial_score,
                             "absdiff", 0.0));
}

void test_conditional_forecast()
{
    std::printf("a forecast conditioned on a scenario\n");
    const Sample s = simulate(40);
    bayests::VarNormalWishartInput input = input_for(s.y, s.lag0, 1, 0);
    set_horizon(input, s.y, 3);

    // The first series at its assumed path for two periods, and the third
    // averaging 0.3 over the three.
    RowBuilder scenario;
    scenario.add(1.0, {{0, 0}}, 1.0);
    scenario.add(-0.5, {{1, 0}}, 1.0);
    scenario.add(0.3, {{0, 2}, {1, 2}, {2, 2}}, 1.0 / 3.0);
    input.forecast.constraints = scenario.build();

    // One set of parameters, many times: the forecast's distribution is then
    // the conditioned path of those parameters, whose moments are known.
    const arma::uword n = 4000;
    bayests::VarNormalWishartDraws draws;
    draws.a = arma::repmat(arma::vectorise(s.a), 1, n);
    draws.u_sigma_inv = arma::repmat(arma::vectorise(arma::inv_sympd(s.sigma)), 1, n);

    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(31);
    const arma::mat fcst =
        bayests::VarNormalWishartSampler{}.forecast(input, draws, reporter).values;

    double worst = 0.0;
    for (arma::uword d = 0; d < n; d++)
    {
        const arma::mat path = arma::reshape(fcst.col(d), kK, 3);
        worst = std::max({worst, std::abs(path(0, 0) - 1.0), std::abs(path(0, 1) + 0.5),
                          std::abs(arma::mean(path.row(2)) - 0.3)});
    }
    check_below("the scenario holds in every draw", worst, 1e-10);

    const auto moments = bayests::core::constrained_var_path_moments(
        bayests::core::var_path_prior(s.a, input.forecast.x,
                                      bayests::core::horizon_presample(s.y.t(), 1), s.sigma, 1),
        input.forecast.constraints, arma::vec());
    const arma::vec mean = arma::vectorise(moments.mean);
    const arma::vec sd = arma::sqrt(moments.covariance.diag());
    const arma::vec sample_mean = arma::mean(fcst, 1);
    double worst_z = 0.0;
    for (arma::uword j = 0; j < mean.n_elem; j++)
    {
        if (sd(j) > 1e-8)
        {
            worst_z = std::max(worst_z, std::abs(sample_mean(j) - mean(j)) /
                                            (sd(j) / std::sqrt(static_cast<double>(n))));
        }
    }
    std::printf("    largest |z| of the forecast mean against the conditioned one: %.6f\n", worst_z);
    check_below("the forecast mean is the conditioned mean, within five standard errors",
                worst_z, 5.0);
}

/// A series observed every month with a measurement error: soft rows, one per
/// month, of a common group whose variance is drawn. The chain has to recover
/// that variance and put the posterior mean of the series closer to the truth
/// than the noisy readings are -- the smoothing a measurement error model
/// exists for.
void test_soft_rows()
{
    std::printf("a series read with an error\n");
    const arma::uword tt = 240;
    const Sample s = simulate(tt);
    const double noise_variance = 0.2;

    arma::arma_rng::set_seed(77);
    const arma::vec readings = s.y.col(2) + std::sqrt(noise_variance) * arma::randn<arma::vec>(tt);

    arma::mat y_file = s.y;
    y_file.col(2).zeros();
    RowBuilder rows;
    for (arma::uword t = 0; t < tt; t++)
    {
        rows.add(s.y(t, 0), {{t, 0}}, 1.0);
        rows.add(s.y(t, 1), {{t, 1}}, 1.0);
        rows.add(readings(t), {{t, 2}}, 1.0, 1);
    }

    bayests::VarNormalWishartInput input = input_for(y_file, s.lag0, 3000, 1000);
    input.train.constraints = rows.build();
    input.constraints_prior.shape = {1.0};
    input.constraints_prior.rate = {0.1};
    input.initial.constraints_inv = {1.0};

    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(2027);
    const auto draws = bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);

    check("a precision is drawn for the one soft group in every kept draw",
          draws.constraints_inv.n_rows == 1 && draws.constraints_inv.n_cols == 3000);
    const arma::rowvec variance = 1.0 / draws.constraints_inv.row(0);
    const double mean = arma::mean(variance);
    const double sd = arma::stddev(variance);
    std::printf("    measurement error variance: posterior mean %.3f (sd %.3f), truth %.3f\n", mean,
                sd, noise_variance);
    check_below("the variance is recovered, within four posterior sds",
                std::abs(mean - noise_variance) / sd, 4.0);

    const arma::mat path_mean = arma::reshape(arma::mean(draws.y, 1), kK, tt);
    const double rmse_model = std::sqrt(arma::mean(arma::square(path_mean.row(2).t() - s.y.col(2))));
    const double rmse_readings = std::sqrt(arma::mean(arma::square(readings - s.y.col(2))));
    std::printf("    series read with an error, RMSE: posterior mean %.3f, readings %.3f\n",
                rmse_model, rmse_readings);
    check_below("the posterior mean of the series beats the readings", rmse_model / rmse_readings,
                0.9);

    const arma::mat loglik = bayests::VarNormalWishartSampler{}.log_likelihood(input, draws);
    check("the log likelihood, which needs the drawn precisions, is finite",
          loglik.is_finite() && loglik.n_rows == 3000 && loglik.n_cols == tt);

    bayests::VarNormalWishartDraws without = draws;
    without.constraints_inv.reset();
    bool refused = false;
    try
    {
        bayests::VarNormalWishartSampler{}.log_likelihood(input, without);
    }
    catch (const std::invalid_argument &)
    {
        refused = true;
    }
    check("and is refused without them", refused);
}

/// VarNormalGamma with a covariance block, through the same completion step:
/// a panel pinned whole is the complete panel bit for bit, and a quarterly
/// series has its averages hold in every draw. The completion step is handed
/// the whole error precision, Psi' Omega Psi, so the block reaches it.
bayests::VarNormalGammaInput gamma_input_for(const arma::mat &y_file, const arma::vec &lag0,
                                             const int iterations, const int burnin)
{
    bayests::VarNormalGammaInput input;
    input.spec.k = static_cast<int>(kK);
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.covar = true;
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.train.y = y_file;
    input.train.z = regressors(y_file, lag0);
    input.a_prior.mu = arma::vec(kK * kNx, arma::fill::zeros);
    input.a_prior.v_inv = 0.1 * arma::eye<arma::mat>(kK * kNx, kK * kNx);
    const arma::uword n_psi = kK * (kK - 1) / 2;
    input.psi_prior.mu = arma::vec(n_psi, arma::fill::zeros);
    input.psi_prior.v_inv = arma::eye<arma::mat>(n_psi, n_psi);
    input.u_sigma_prior.shape = arma::vec(kK, arma::fill::value(3.0));
    input.u_sigma_prior.rate = arma::vec(kK, arma::fill::value(1.0));
    input.initial.a = arma::vec(kK * kNx, arma::fill::zeros);
    input.initial.psi = arma::vec(n_psi, arma::fill::zeros);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return input;
}

/// An i.i.d. variable not observed at the start of the sample: the
/// high-frequency surprise of Jarocinski and Karadi (2020), missing before its
/// series begins. The first variable is white noise whose error moves with the
/// second's at a correlation of 0.9, and the other two a VAR(1) that reads it
/// as a regressor. Pinned whole, the chain is the complete panel's bit for bit;
/// with its first 30 periods missing, its equation stays exactly zero, every
/// observation holds, and what was missing is recovered from the second
/// variable's errors far better than by its unconditional mean.
void test_iid_variable_not_observed()
{
    std::printf("an i.i.d. variable not observed at the start of the sample\n");
    bayests::NullReporter reporter;
    const bayests::VarNormalWishartSampler sampler;

    const arma::uword tt = 150;
    arma::mat a_true = {{0.0, 0.0, 0.0, 0.0}, {0.3, 0.5, 0.1, 0.1}, {0.1, 0.2, 0.4, -0.1}};
    const arma::mat sigma = {{1.0, 0.9, 0.0}, {0.9, 1.0, 0.2}, {0.0, 0.2, 1.0}};
    const arma::mat root = arma::chol(sigma, "lower");
    arma::arma_rng::set_seed(515);
    arma::vec previous(kK, arma::fill::zeros);
    for (int burn = 0; burn < 50; burn++)
    {
        previous = a_true.cols(0, kK - 1) * previous + a_true.col(kK) + root * arma::randn<arma::vec>(kK);
    }
    const arma::vec lag0 = previous;
    arma::mat y(tt, kK);
    for (arma::uword t = 0; t < tt; t++)
    {
        previous = a_true.cols(0, kK - 1) * previous + a_true.col(kK) + root * arma::randn<arma::vec>(kK);
        y.row(t) = previous.t();
    }

    bayests::VarNormalWishartInput plain = input_for(y, lag0, 400, 100);
    plain.spec.n_iid = 1;
    bayests::VarNormalWishartInput pinned = plain;
    RowBuilder all;
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < kK; i++)
        {
            all.add(y(t, i), {{t, i}}, 1.0);
        }
    }
    pinned.train.constraints = all.build();
    arma::arma_rng::set_seed(21);
    const auto a = sampler.draw_coefficients(plain, reporter);
    arma::arma_rng::set_seed(21);
    const auto b = sampler.draw_coefficients(pinned, reporter);
    check("pinned whole with n_iid: coefficient draws agree bit for bit",
          arma::approx_equal(a.a, b.a, "absdiff", 0.0) &&
              arma::approx_equal(a.u_sigma_inv, b.u_sigma_inv, "absdiff", 0.0));

    const arma::uword missing = 30;
    arma::mat y_file = y;
    y_file.col(0).head(missing).zeros();
    RowBuilder rows;
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < kK; i++)
        {
            if (i > 0 || t >= missing)
            {
                rows.add(y(t, i), {{t, i}}, 1.0);
            }
        }
    }
    bayests::VarNormalWishartInput gappy = input_for(y_file, lag0, 2000, 500);
    gappy.spec.n_iid = 1;
    gappy.train.constraints = rows.build();
    arma::arma_rng::set_seed(22);
    const auto d = sampler.draw_coefficients(gappy, reporter);

    // The restricted equation is every k-th coefficient from the first.
    double restricted = 0.0;
    for (arma::uword j = 0; j < d.a.n_rows; j += kK)
    {
        restricted = std::max(restricted, arma::abs(d.a.row(j)).max());
    }
    check("the i.i.d. equation carries exact zeros in every draw", restricted == 0.0);

    double worst = 0.0;
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < kK; i++)
        {
            if (i > 0 || t >= missing)
            {
                worst = std::max(worst, arma::abs(d.y.row(t * kK + i) - y(t, i)).max());
            }
        }
    }
    check_below("every observation holds in every draw", worst, 1e-10);

    arma::vec completed(missing);
    for (arma::uword t = 0; t < missing; t++)
    {
        completed(t) = arma::mean(d.y.row(t * kK));
    }
    const arma::vec truth = y.col(0).head(missing);
    const double rmse = std::sqrt(arma::mean(arma::square(completed - truth)));
    const double rms = std::sqrt(arma::mean(arma::square(truth)));
    std::printf("    missing surprises: error of the posterior mean %.3f, of zero %.3f\n", rmse, rms);
    check_below("the missing surprises are recovered from the correlated errors", rmse / rms, 0.7);

    const arma::mat ll = sampler.log_likelihood(gappy, d);
    check("and the log likelihood of what was observed is finite", ll.is_finite());
}

void test_var_normal_gamma()
{
    std::printf("VarNormalGamma with a covariance block\n");
    bayests::NullReporter reporter;
    const bayests::VarNormalGammaSampler sampler;

    const Sample s = simulate(60);
    bayests::VarNormalGammaInput plain = gamma_input_for(s.y, s.lag0, 200, 50);
    bayests::VarNormalGammaInput pinned = plain;
    RowBuilder all;
    for (arma::uword t = 0; t < s.y.n_rows; t++)
    {
        for (arma::uword i = 0; i < kK; i++)
        {
            all.add(s.y(t, i), {{t, i}}, 1.0);
        }
    }
    pinned.train.constraints = all.build();
    arma::arma_rng::set_seed(11);
    const auto a = sampler.draw_coefficients(plain, reporter);
    arma::arma_rng::set_seed(11);
    const auto b = sampler.draw_coefficients(pinned, reporter);
    check("a panel pinned whole: coefficient draws agree bit for bit",
          arma::approx_equal(a.a, b.a, "absdiff", 0.0) &&
              arma::approx_equal(a.psi, b.psi, "absdiff", 0.0) &&
              arma::approx_equal(a.u_sigma_inv, b.u_sigma_inv, "absdiff", 0.0));
    const arma::mat ll_a = sampler.log_likelihood(plain, a);
    check_below("and the log likelihood agrees to rounding",
                arma::abs(sampler.log_likelihood(pinned, b) - ll_a).max() / arma::abs(ll_a).max(),
                1e-12);

    const arma::uword tt = 120;
    const Sample m = simulate(tt);
    arma::mat y_file = m.y;
    y_file.col(2).zeros();
    RowBuilder rows;
    arma::vec quarterly(tt / 3);
    for (arma::uword t = 0; t < tt; t++)
    {
        rows.add(m.y(t, 0), {{t, 0}}, 1.0);
        rows.add(m.y(t, 1), {{t, 1}}, 1.0);
    }
    for (arma::uword q = 0; q < tt / 3; q++)
    {
        quarterly(q) = arma::mean(m.y.col(2).subvec(3 * q, 3 * q + 2));
        rows.add(quarterly(q), {{3 * q, 2}, {3 * q + 1, 2}, {3 * q + 2, 2}}, 1.0 / 3.0);
    }
    bayests::VarNormalGammaInput mixed = gamma_input_for(y_file, m.lag0, 500, 200);
    mixed.train.constraints = rows.build();
    arma::arma_rng::set_seed(12);
    const auto draws = sampler.draw_coefficients(mixed, reporter);
    double worst = 0.0;
    for (arma::uword d = 0; d < draws.y.n_cols; d++)
    {
        const arma::mat path = arma::reshape(draws.y.col(d), kK, tt);
        for (arma::uword q = 0; q < tt / 3; q++)
        {
            worst = std::max(worst, std::abs(arma::mean(path.row(2).subvec(3 * q, 3 * q + 2)) -
                                             quarterly(q)));
        }
    }
    check_below("a quarterly series: every average holds in every draw", worst, 1e-10);
    check("and the observed-data log likelihood is finite",
          sampler.log_likelihood(mixed, draws).is_finite());
}

void test_refusals()
{
    std::printf("what is refused\n");
    const Sample s = simulate(12);
    RowBuilder base;
    base.add(s.y(0, 0), {{0, 0}}, 1.0);

    const auto passes = [](const bayests::VarNormalWishartInput &input) {
        try
        {
            input.validate();
        }
        catch (const std::exception &e)
        {
            std::printf("    %s\n", e.what());
            return false;
        }
        return true;
    };

    bayests::VarNormalWishartInput input = input_for(s.y, s.lag0, 10, 0);
    input.train.constraints = base.build();
    check("hard constraints are read", passes(input));

    set_horizon(input, s.y, 2);
    input.test.y = s.y.rows(0, 1);
    check("and a score beside them", passes(input));

    input.test.constraints = pin_all(s.y.rows(0, 1)).build();
    check("and constraints on the horizon realised", passes(input));

    RowBuilder soft = base;
    soft.add(0.1, {{3, 2}}, 1.0, 1);
    input = input_for(s.y, s.lag0, 10, 0);
    input.train.constraints = soft.build();
    check("a soft row without a prior on its precision is refused",
          validate_refuses(input, "one value per group"));
    input.constraints_prior.shape = {2.0};
    input.constraints_prior.rate = {0.5};
    input.initial.constraints_inv = {1.0};
    check("and with one it is read", passes(input));
    input.constraints_prior.rate = {0.0};
    check("a rate of zero is refused", validate_refuses(input, "positive"));

    input = input_for(s.y, s.lag0, 10, 0);
    set_horizon(input, s.y, 2);
    input.test.y = s.y.rows(0, 1);
    RowBuilder soft_test;
    soft_test.add(0.1, {{0, 0}}, 1.0, 1);
    input.test.constraints = soft_test.build();
    check("a soft row in what the horizon realised is refused",
          validate_refuses(input, "hard constraints only in /data/test"));

    RowBuilder soft_scenario;
    soft_scenario.add(0.1, {{0, 0}}, 1.0, 1);
    input = input_for(s.y, s.lag0, 10, 0);
    set_horizon(input, s.y, 2);
    input.forecast.constraints = soft_scenario.build();
    check("so is a soft row in a scenario", validate_refuses(input, "hard constraints only"));

    RowBuilder scenario;
    scenario.add(0.5, {{1, 0}}, 1.0);
    input.forecast.constraints = scenario.build();
    check("a hard scenario is read", passes(input));

    input.test.y = s.y.rows(0, 1);
    check("a scenario beside a score is refused", validate_refuses(input, "conditional forecast"));

    input = input_for(s.y, s.lag0, 10, 0);
    input.forecast.constraints = scenario.build();
    check("a scenario without a horizon is refused", validate_refuses(input, "set h"));

    RowBuilder beyond;
    beyond.add(0.5, {{2, 0}}, 1.0);
    input = input_for(s.y, s.lag0, 10, 0);
    set_horizon(input, s.y, 2);
    input.forecast.constraints = beyond.build();
    check("a scenario past the horizon is refused", validate_refuses(input, "periods"));

    input = input_for(s.y, s.lag0, 10, 0);
    input.train.constraints = base.build();
    input.spec.n_iid = 1;
    bool iid_read = true;
    try
    {
        input.validate();
    }
    catch (const std::exception &e)
    {
        std::printf("    %s\n", e.what());
        iid_read = false;
    }
    check("an i.i.d. block beside constraints is read, not refused", iid_read);
}

} // namespace


/// The VAR of simulate(), its shocks in period t multiplied by scale(t).
Sample simulate_scaled(const arma::uword tt, const arma::vec &scale)
{
    Sample s = simulate(1);
    const arma::mat root = arma::chol(s.sigma, "lower");
    arma::arma_rng::set_seed(5151);
    arma::vec previous = s.lag0;
    s.y.set_size(tt, kK);
    for (arma::uword t = 0; t < tt; t++)
    {
        previous = s.a.cols(0, kK - 1) * previous + s.a.col(kK) +
                   scale(t) * (root * arma::randn<arma::vec>(kK));
        s.y.row(t) = previous.t();
    }
    return s;
}

/// The volatility break of Lenza and Primiceri (2022) at `break_at`, and
/// Student t errors where `t` says so.
void add_break_and_t(bayests::VarNormalWishartInput &input, const int break_at, const bool t)
{
    input.spec.volatility_break = break_at;
    input.spec.volatility_scales = 3;
    input.volatility_prior.shape = 1.0;
    input.volatility_prior.rho_shape1 = 3.0357;
    input.volatility_prior.rho_shape2 = 1.5089;
    if (t)
    {
        input.spec.student_t = true;
        input.student_t_prior.shape = 2.0;
        input.student_t_prior.rate = 0.1;
    }
}

void test_scaled_pinned_whole()
{
    std::printf("a volatility break and Student t errors on a panel pinned whole\n");
    const arma::uword tt = 60;
    // Ones, then a jump at period 52 (one-based) decaying through the three
    // periods a score reads after the sample.
    const arma::vec scale = arma::join_cols(
        arma::vec(51, arma::fill::ones),
        arma::vec({8.0, 20.0, 6.0, 3.5, 2.2, 1.6, 1.3, 1.2, 1.1, 1.05, 1.03, 1.02}));
    const Sample s = simulate_scaled(tt + 3, scale);
    const arma::mat y = s.y.head_rows(tt);
    bayests::NullReporter reporter;
    const bayests::VarNormalWishartSampler sampler;

    for (const bool t : {false, true})
    {
        const std::string tag = t ? ", with t errors" : "";
        bayests::VarNormalWishartInput plain = input_for(y, s.lag0, 200, 50);
        add_break_and_t(plain, 52, t);
        bayests::VarNormalWishartInput pinned = plain;
        RowBuilder rows;
        for (arma::uword p = 0; p < tt; p++)
        {
            for (arma::uword i = 0; i < kK; i++)
            {
                rows.add(y(p, i), {{p, i}}, 1.0);
            }
        }
        pinned.train.constraints = rows.build();

        arma::arma_rng::set_seed(77);
        const auto a = sampler.draw_coefficients(plain, reporter);
        arma::arma_rng::set_seed(77);
        const auto b = sampler.draw_coefficients(pinned, reporter);
        bool same = arma::approx_equal(a.a, b.a, "absdiff", 0.0) &&
                    arma::approx_equal(a.u_sigma_inv, b.u_sigma_inv, "absdiff", 0.0) &&
                    arma::approx_equal(a.volatility, b.volatility, "absdiff", 0.0);
        if (t)
        {
            same = same && arma::approx_equal(a.student_t_lambda, b.student_t_lambda, "absdiff", 0.0) &&
                   arma::approx_equal(a.student_t_df, b.student_t_df, "absdiff", 0.0);
        }
        check("the chain agrees bit for bit with the complete panel's" + tag, same);
        check("and its stored panel is the data, on the original scale" + tag,
              b.has_y() && arma::approx_equal(b.y, arma::repmat(arma::vectorise(y.t()), 1, b.y.n_cols),
                                              "absdiff", 0.0));

        const arma::mat ll_pinned = sampler.log_likelihood(pinned, b);
        if (!t)
        {
            const arma::mat ll_plain = sampler.log_likelihood(plain, a);
            check_below("the log likelihood is the complete panel's, to rounding",
                        arma::abs(ll_plain - ll_pinned).max() / arma::abs(ll_plain).max(), 1e-12);
        }
        else
        {
            // Given the mixing scales: the normal with lambda_t s_t^2 Sigma.
            double worst = 0.0;
            for (arma::uword d = 0; d < 5; d++)
            {
                const arma::mat a_d = arma::reshape(b.a.col(d), kK, kNx);
                const arma::mat sigma = arma::inv_sympd(arma::reshape(b.u_sigma_inv.col(d), kK, kK));
                const arma::vec path =
                    bayests::core::volatility_path(pinned.spec, b.volatility.col(d), 0, tt) %
                    arma::sqrt(b.student_t_lambda.col(d));
                for (arma::uword p = 0; p < tt; p++)
                {
                    const arma::vec lag = p == 0 ? s.lag0 : arma::vec(y.row(p - 1).t());
                    const arma::vec e = y.row(p).t() - a_d * arma::join_cols(lag, arma::vec({1.0}));
                    const arma::mat cov = path(p) * path(p) * sigma;
                    double logdet, sign;
                    arma::log_det(logdet, sign, cov);
                    const double want =
                        -0.5 * (kK * std::log(2.0 * arma::datum::pi) + logdet +
                                arma::as_scalar(e.t() * arma::solve(cov, e)));
                    worst = std::max(worst, std::abs(ll_pinned(d, p) - want));
                }
            }
            check_below("the log likelihood is the normal given the mixing scales", worst, 1e-8);
        }

        if (!t)
        {
            // The score of a horizon realised whole, given as constraints or not.
            bayests::VarNormalWishartInput scored = plain;
            scored.spec.h = 3;
            scored.test.y = s.y.rows(tt, tt + 2);
            scored.forecast.x = arma::mat(3, kNx, arma::fill::zeros);
            scored.forecast.x.col(kK).ones();
            scored.forecast.x.submat(0, 0, 0, kK - 1) = y.row(tt - 1);
            const arma::mat recursion = sampler.predictive_log_density(scored, a);
            scored.test.constraints = bayests::core::pin_every_entry(scored.test.y);
            const arma::mat from_panel = sampler.predictive_log_density(scored, a);
            check_below("the score from a panel is the recursion's, horizon scales and all",
                        arma::abs(recursion - from_panel).max() / arma::abs(recursion).max(), 1e-10);
        }
    }
}

void test_quarterly_series_beside_a_break()
{
    std::printf("a quarterly series beside a volatility break\n");
    const arma::uword tt = 240;
    const arma::vec theta = {10.0, 25.0, 6.0, 0.7};
    bayests::VarSpec spec;
    spec.volatility_break = 202;
    spec.volatility_scales = 3;
    const Sample s = simulate_scaled(tt, bayests::core::volatility_path(spec, theta, 0, tt));

    arma::mat y_file = s.y;
    y_file.col(2).zeros();
    RowBuilder rows;
    for (arma::uword t = 0; t < tt; t++)
    {
        rows.add(s.y(t, 0), {{t, 0}}, 1.0);
        rows.add(s.y(t, 1), {{t, 1}}, 1.0);
    }
    arma::vec quarterly(tt / 3);
    for (arma::uword q = 0; q < tt / 3; q++)
    {
        quarterly(q) = arma::mean(s.y.col(2).subvec(3 * q, 3 * q + 2));
        rows.add(quarterly(q), {{3 * q, 2}, {3 * q + 1, 2}, {3 * q + 2, 2}}, 1.0 / 3.0);
    }

    bayests::NullReporter reporter;
    for (const bool t : {false, true})
    {
        const std::string tag = t ? ", with t errors" : "";
        bayests::VarNormalWishartInput input = input_for(y_file, s.lag0, 3000, 1500);
        input.train.constraints = rows.build();
        add_break_and_t(input, 202, t);
        arma::arma_rng::set_seed(2027);
        const auto draws = bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);

        double worst = 0.0;
        for (arma::uword d = 0; d < draws.y.n_cols; d++)
        {
            const arma::mat path = arma::reshape(draws.y.col(d), kK, tt);
            for (arma::uword q = 0; q < tt / 3; q++)
            {
                worst = std::max(worst, std::abs(arma::mean(path.row(2).subvec(3 * q, 3 * q + 2)) -
                                                 quarterly(q)));
            }
            worst = std::max(worst, arma::abs(path.row(0).t() - s.y.col(0)).max());
        }
        check_below("every quarterly average and every pinned value holds in every draw" + tag, worst,
                    1e-10);

        const arma::vec lo = arma::quantile(draws.volatility, arma::vec({0.005}), 1);
        const arma::vec hi = arma::quantile(draws.volatility, arma::vec({0.995}), 1);
        const arma::vec med = arma::median(draws.volatility, 1);
        std::printf("    break medians %.2f %.2f %.2f %.3f, truth 10 25 6 0.7\n", med(0), med(1), med(2),
                    med(3));
        check("the break's 99% intervals cover the truth" + tag,
              arma::all(lo <= theta) && arma::all(hi >= theta));

        // The hidden monthly series against the quarterly figure repeated.
        const arma::mat mean_path = arma::reshape(arma::mean(draws.y, 1), kK, tt);
        arma::vec step(tt);
        for (arma::uword p = 0; p < tt; p++)
        {
            step(p) = quarterly(p / 3);
        }
        const double rmse_model = std::sqrt(arma::mean(arma::square(mean_path.row(2).t() - s.y.col(2))));
        const double rmse_step = std::sqrt(arma::mean(arma::square(step - s.y.col(2))));
        std::printf("    hidden monthly series, RMSE: posterior mean %.3f, quarterly step %.3f\n",
                    rmse_model, rmse_step);
        check("the completed months beat the quarterly step" + tag, rmse_model < rmse_step);

        const arma::mat ll = bayests::VarNormalWishartSampler{}.log_likelihood(input, draws);
        check("the log likelihood is finite" + tag, ll.is_finite());
    }
}

int main()
{
    try
    {
        test_pinned_whole_is_the_complete_panel();
        test_quarterly_series_is_recovered();
        test_forecast_starts_from_the_completed_panel();
        test_scores();
        test_conditional_forecast();
        test_soft_rows();
        test_iid_variable_not_observed();
        test_var_normal_gamma();
        test_scaled_pinned_whole();
        test_quarterly_series_beside_a_break();
        test_refusals();
    }
    catch (const std::exception &e)
    {
        std::printf("FAILED: threw: %s\n", e.what());
        failures++;
    }

    if (failures != 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
