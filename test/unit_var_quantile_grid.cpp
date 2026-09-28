// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_var_quantile_grid.cpp
/// @brief Checks the structural quantile VAR: VarNormalAld over a grid of
///        quantiles, forecast by simulation.
///
/// First the distribution a grid describes (core/models/quantile_grid.h): its
/// density integrates to one, its inverse passes through every grid point, and
/// the two are inverses of each other -- identities, to rounding.
///
/// Then a recursive system simulated with normal errors,
///
///     y1_t = 0.5 y1_{t-1} + e1_t,
///     y2_t = 0.3 y2_{t-1} + 0.8 y1_t + e2_t,
///
/// whose conditional quantiles are linear with known coefficients, so the grid
/// has a truth to recover: the contemporaneous coefficient at every level, and
/// the intercept moving with the normal quantile. Its forecast, simulated
/// variable by variable through the grid, has to reproduce the true predictive
/// distribution -- the one-step mean and spread of both variables and the
/// three-step spread of the first -- with margins read off the run. A fixed
/// level gives a path that does not depend on the random numbers; a pinned
/// value holds in every draw and moves what is ordered after it by the
/// coefficient; the log likelihood of the implied density averages close to the
/// normal one. And what is refused is refused.

#include "bayests/reporter.h"
#include "bayests/var_normal_ald.h"
#include "bayests/var_tvp_ald.h"
#include "core/models/quantile_grid.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_below(const std::string &what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-66s %s   %.3e < %.3e\n", what.c_str(), ok ? "ok" : "FAILED", got, bound);
    if (!ok)
    {
        failures++;
    }
}

double normal_quantile(const double p)
{
    // Acklam's rational approximation, ample for a test's reference values.
    const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                        1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                        6.680131188771972e+01,  -1.328068155288572e+01};
    const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                        -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                        3.754408661907416e+00};
    if (p < 0.02425)
    {
        const double q = std::sqrt(-2 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - 0.02425)
    {
        return -normal_quantile(1 - p);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

void test_grid_distribution()
{
    std::printf("the distribution a grid describes\n");
    const arma::vec tau = {0.1, 0.25, 0.5, 0.75, 0.9};
    // Out of order and with two equal: rearranged and pulled apart.
    const bayests::core::QuantileGrid g = bayests::core::quantile_grid(tau, arma::vec{-0.4, -1.0, 0.3, 1.7, 0.3});
    check("the quantiles are rearranged into order", g.q(0) == -1.0 && g.q(4) == 1.7);

    double worst_at_grid = 0.0;
    for (arma::uword j = 0; j < tau.n_elem; j++)
    {
        worst_at_grid = std::max(worst_at_grid, std::fabs(bayests::core::quantile_grid_value(g, tau(j)) - g.q(j)));
    }
    check_below("the inverse passes through every grid point", worst_at_grid, 1e-12);

    check("coinciding quantiles are pulled apart", g.q(3) > g.q(2));

    // On distinct quantiles: the density integrates to one -- trapezoid on a
    // fine grid far into both tails -- and d value / d u is one over the density
    // at the value, in both tails and inside.
    const bayests::core::QuantileGrid d =
        bayests::core::quantile_grid(tau, arma::vec{-1.3, -0.6, 0.1, 0.7, 1.4});
    double total = 0.0;
    const double step = 1e-4;
    double previous = std::exp(bayests::core::quantile_grid_log_density(d, -40.0));
    for (double x = -40.0 + step; x <= 40.0; x += step)
    {
        const double current = std::exp(bayests::core::quantile_grid_log_density(d, x));
        total += 0.5 * (previous + current) * step;
        previous = current;
    }
    check_below("the density integrates to one", std::fabs(total - 1.0), 1e-3);

    double worst_slope = 0.0;
    for (const double u : {0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99})
    {
        const double h = 1e-7;
        const double slope = (bayests::core::quantile_grid_value(d, u + h) -
                              bayests::core::quantile_grid_value(d, u - h)) / (2 * h);
        const double density =
            std::exp(bayests::core::quantile_grid_log_density(d, bayests::core::quantile_grid_value(d, u)));
        worst_slope = std::max(worst_slope, std::fabs(slope * density - 1.0));
    }
    check_below("the inverse and the density are inverses of each other", worst_slope, 1e-4);
}

struct Sample
{
    arma::mat y; // tt x 2
    arma::mat z; // (2 tt) x (2 * 3 + 1)
    arma::vec last;
};

/// The recursive system, and its regressors laid out as a structural model
/// lays them out: kron([y1_{t-1} y2_{t-1} 1], I_2), then -y1_t in the row of
/// the second equation.
Sample simulate(const arma::uword tt)
{
    Sample s;
    arma::arma_rng::set_seed(2019);
    arma::vec previous(2, arma::fill::zeros);
    arma::mat path(tt + 51, 2);
    path.row(0).zeros();
    for (arma::uword t = 1; t < path.n_rows; t++)
    {
        const double y1 = 0.5 * path(t - 1, 0) + arma::randn<double>();
        const double y2 = 0.3 * path(t - 1, 1) + 0.8 * y1 + arma::randn<double>();
        path(t, 0) = y1;
        path(t, 1) = y2;
    }
    path = path.rows(50, path.n_rows - 1); // tt + 1 rows: one lag, then the sample
    s.y = path.rows(1, tt);
    s.z = arma::zeros<arma::mat>(2 * tt, 7);
    for (arma::uword t = 0; t < tt; t++)
    {
        const arma::rowvec x = {path(t, 0), path(t, 1), 1.0};
        s.z.submat(2 * t, 0, 2 * t + 1, 5) = arma::kron(x, arma::eye<arma::mat>(2, 2));
        s.z(2 * t + 1, 6) = -s.y(t, 0);
    }
    s.last = path.row(tt).t();
    return s;
}

bayests::VarNormalAldInput make_input(const Sample &s, const std::vector<double> &grid,
                                      const int iterations, const int burnin)
{
    bayests::VarNormalAldInput input;
    input.spec.k = 2;
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.structural = true;
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.spec.quantiles = grid;
    input.train.y = s.y;
    input.train.z = s.z;
    input.a_prior.mu = arma::zeros<arma::vec>(7);
    input.a_prior.v_inv = 0.01 * arma::eye<arma::mat>(7, 7);
    input.u_scale_prior.shape = arma::vec(2, arma::fill::value(3.0));
    input.u_scale_prior.rate = arma::vec(2, arma::fill::value(0.2));
    input.initial.a = arma::zeros<arma::vec>(7);
    input.initial.w = arma::ones<arma::mat>(s.y.n_rows, 2);
    input.initial.u_scale = arma::ones<arma::vec>(2);
    return input;
}

void set_horizon(bayests::VarNormalAldInput &input, const Sample &s, const int h)
{
    input.spec.h = h;
    input.forecast.x = arma::zeros<arma::mat>(h, 3);
    input.forecast.x.col(2).ones();
    input.forecast.x(0, 0) = s.last(0);
    input.forecast.x(0, 1) = s.last(1);
}

std::vector<double> grid19()
{
    std::vector<double> grid;
    for (int j = 1; j <= 19; j++)
    {
        grid.push_back(0.05 * j);
    }
    return grid;
}

void test_recursive_system()
{
    std::printf("a recursive system with normal errors\n");
    const Sample s = simulate(600);
    const std::vector<double> grid = grid19();
    bayests::VarNormalAldInput input = make_input(s, grid, 500, 300);
    bayests::NullReporter reporter;
    const bayests::VarNormalAldSampler sampler;

    arma::arma_rng::set_seed(1);
    const bayests::VarNormalAldDraws draws = sampler.draw_coefficients(input, reporter);
    check("one block of coefficients and of scales per level",
          draws.a.n_rows == 7 * grid.size() && draws.u_scale.n_rows == 2 * grid.size() &&
              draws.iterations() == 500);

    // A_0 y = B x + u: the coefficient on -y1 in the second equation is A_0(2, 1),
    // which is -0.8 at every level.
    double worst_a0 = 0.0;
    for (std::size_t j = 0; j < grid.size(); j++)
    {
        worst_a0 = std::max(worst_a0, std::fabs(arma::mean(draws.a.row(7 * j + 6)) + 0.8));
    }
    check_below("the contemporaneous coefficient is -0.8 at every level", worst_a0, 0.15);

    // The first equation's intercept is the normal quantile at each level.
    double worst_intercept = 0.0;
    for (std::size_t j = 0; j < grid.size(); j++)
    {
        worst_intercept = std::max(
            worst_intercept, std::fabs(arma::mean(draws.a.row(7 * j + 4)) - normal_quantile(grid[j])));
    }
    check_below("the intercept moves with the normal quantile", worst_intercept, 0.3);

    // The one-step predictive distribution, simulated through the grid.
    set_horizon(input, s, 3);
    arma::arma_rng::set_seed(2);
    const arma::mat f = sampler.forecast(input, draws, reporter).values;
    check("the forecast holds every horizon of every draw", f.n_rows == 6 && f.n_cols == 500);
    // Pooled over draws: many draws of the path per posterior draw would be
    // closer, but the posterior draws already carry the parameter uncertainty.
    const double m1 = arma::mean(f.row(0));
    const double s1 = arma::stddev(f.row(0));
    const double s2 = arma::stddev(f.row(1));
    const double s1_3 = arma::stddev(f.row(4));
    std::printf("    y1 one step: mean %.3f (truth %.3f), sd %.3f (1.000); y2 one step sd %.3f (1.281)"
                "; y1 three steps sd %.3f (1.146)\n",
                m1, 0.5 * s.last(0), s1, s2, s1_3);
    check_below("the one-step mean of y1", std::fabs(m1 - 0.5 * s.last(0)), 0.15);
    check_below("the one-step spread of y1", std::fabs(s1 - 1.0), 0.12);
    check_below("the one-step spread of y2, through the contemporaneous term",
                std::fabs(s2 - std::sqrt(1.64)), 0.15);
    check_below("the three-step spread of y1, through the lags", std::fabs(s1_3 - std::sqrt(1.3125)), 0.15);

    // A fixed level: a path per draw that the random numbers do not touch.
    bayests::VarNormalAldInput fixed = input;
    fixed.spec.forecast_quantile = 0.5;
    arma::arma_rng::set_seed(3);
    const arma::mat a = sampler.forecast(fixed, draws, reporter).values;
    arma::arma_rng::set_seed(4);
    const arma::mat b = sampler.forecast(fixed, draws, reporter).values;
    check("a fixed level gives the same paths whatever the seed",
          arma::approx_equal(a, b, "absdiff", 0.0));

    // A pin: y1 held at 2 in the first period, y2 responding through A_0.
    bayests::VarNormalAldInput pinned = input;
    pinned.forecast.constraints.value = {2.0};
    pinned.forecast.constraints.group = {0};
    pinned.forecast.constraints.row = {0};
    pinned.forecast.constraints.period = {0};
    pinned.forecast.constraints.variable = {0};
    pinned.forecast.constraints.weight = {1.0};
    arma::arma_rng::set_seed(2);
    const arma::mat p = sampler.forecast(pinned, draws, reporter).values;
    check("the pinned value holds in every draw", arma::all(p.row(0) == 2.0));
    const double expected = 0.3 * s.last(1) + 0.8 * 2.0;
    std::printf("    y2 given y1 = 2: mean %.3f (truth %.3f)\n", arma::mean(p.row(1)), expected);
    check_below("and y2 moves by the contemporaneous coefficient",
                std::fabs(arma::mean(p.row(1)) - expected), 0.15);

    // The implied density against the true normal one, per period.
    input.spec.h = 0;
    input.forecast = bayests::ForecastData();
    const arma::mat ll = sampler.log_likelihood(input, draws);
    // The true per-period log density: both errors standard normal.
    const arma::mat res = arma::reshape(arma::vec(arma::vectorise(s.y.t())), 2, s.y.n_rows);
    arma::vec e1(s.y.n_rows), e2(s.y.n_rows);
    for (arma::uword t = 0; t < s.y.n_rows; t++)
    {
        const double l1 = s.z(2 * t, 0);
        const double l2 = s.z(2 * t, 2);
        e1(t) = res(0, t) - 0.5 * l1;
        e2(t) = res(1, t) - 0.3 * l2 - 0.8 * res(0, t);
    }
    const double truth = arma::mean(-std::log(2 * arma::datum::pi) - 0.5 * (arma::square(e1) + arma::square(e2)));
    // The density code at the true quantiles: a grid of normal quantiles at
    // these nineteen levels describes the normal closely, so this is the
    // approximation alone.
    arma::vec normal_levels(grid.size());
    for (std::size_t j = 0; j < grid.size(); j++)
    {
        normal_levels(j) = normal_quantile(grid[j]);
    }
    const bayests::core::QuantileGrid normal = bayests::core::quantile_grid(arma::vec(grid), normal_levels);
    double at_truth = 0.0;
    for (arma::uword t = 0; t < s.y.n_rows; t++)
    {
        at_truth += bayests::core::quantile_grid_log_density(normal, e1(t)) +
                    bayests::core::quantile_grid_log_density(normal, e2(t));
    }
    at_truth /= static_cast<double>(s.y.n_rows);

    // Estimated, the quantiles carry sampling noise, which roughens the density
    // they describe, and the average over draws of a log density sits below the
    // log density of the average: so a looser bound.
    const double implied = arma::mean(arma::mean(ll));
    std::printf("    mean log density per period: true %.3f, grid at the true quantiles %.3f, "
                "estimated %.3f\n", truth, at_truth, implied);
    check_below("at the true quantiles the grid's density is the normal one",
                std::fabs(at_truth - truth), 0.03);
    check("the implied log likelihood is finite", ll.is_finite());
    check_below("and estimated, it stays within the noise of the quantiles",
                std::fabs(implied - truth), 0.35);

    // The tails are bounded by the asymmetric Laplace ones: an observation far
    // beyond the outermost quantile costs what it costs under that distribution,
    // not a near-impossibility, however close the outermost quantiles lie.
    const arma::vec close = {-1.0, -0.999999, 0.0, 0.999999, 1.0};
    const arma::vec five = {0.05, 0.1, 0.5, 0.9, 0.95};
    const double bounded = bayests::core::quantile_grid_log_density(
        bayests::core::quantile_grid(five, close, 1.0, 1.0), 4.0);
    const double unbounded =
        bayests::core::quantile_grid_log_density(bayests::core::quantile_grid(five, close), 4.0);
    std::printf("    log density three units beyond a tight grid: %.1f bounded, %.1f by continuity\n",
                bounded, unbounded);
    check("the tails are bounded by the asymmetric Laplace ones", bounded > -10.0 && unbounded < -1000.0);
}

template <typename Input>
bool refuses(const Input &input, const std::string &contains)
{
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &e)
    {
        const bool ok = std::string(e.what()).find(contains) != std::string::npos;
        if (!ok)
        {
            std::printf("    refused, but: %s\n", e.what());
        }
        return ok;
    }
    return false;
}

void test_refusals()
{
    std::printf("refusals\n");
    const Sample s = simulate(60);
    const bayests::VarNormalAldInput base = make_input(s, {0.25, 0.5, 0.75}, 10, 0);

    auto input = base;
    input.spec.quantiles = {0.5, 0.25};
    check("a grid out of order is refused", refuses(input, "strictly increasing"));
    input.spec.quantiles = {0.5, 1.0};
    check("and one reaching 1", refuses(input, "(0, 1)"));
    input.spec.quantiles = {0.5};
    check("and a grid of one level", refuses(input, "at least two"));

    input = base;
    set_horizon(input, s, 2);
    input.spec.structural = false;
    input.train.z = s.z.cols(0, 5);
    input.a_prior.mu = arma::zeros<arma::vec>(6);
    input.a_prior.v_inv = 0.01 * arma::eye<arma::mat>(6, 6);
    input.initial.a = arma::zeros<arma::vec>(6);
    check("a grid forecast without a recursive system is refused", refuses(input, "structural"));

    input = base;
    input.spec.quantiles.clear();
    input.spec.forecast_quantile = 0.5;
    check("a fixed level without a grid is refused", refuses(input, "grid only"));

    input = base;
    set_horizon(input, s, 2);
    input.forecast.constraints.value = {1.0};
    input.forecast.constraints.group = {0};
    input.forecast.constraints.row = {0, 0};
    input.forecast.constraints.period = {0, 0};
    input.forecast.constraints.variable = {0, 1};
    input.forecast.constraints.weight = {0.5, 0.5};
    check("a scenario row of two entries is refused", refuses(input, "pins"));

    input = base;
    input.spec.quantiles.clear();
    input.spec.quantile = 0.5;
    set_horizon(input, s, 2);
    check("a single quantile still does not forecast", refuses(input, "does not forecast"));

    bayests::VarTvpAldInput tvp;
    tvp.spec.k = 2;
    tvp.spec.quantiles = {0.25, 0.75};
    check("VarTvpAld refuses a grid by name", refuses(tvp, "VarTvpAld does not read /model/quantiles"));
}

} // namespace

int main()
{
    try
    {
        test_grid_distribution();
        test_recursive_system();
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
