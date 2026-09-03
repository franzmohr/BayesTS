// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That DfmTvpStochvol draws the model it says it does.
//
// The three tests before this one cover everything it inherits.
// unit_dfm_normal_gamma.cpp has the identification of Lambda and the layout of
// the lagged factors; unit_dfm_normal_stochvol.cpp has the per-period covariance
// conventions; unit_dfm_tvp_gamma.cpp has the stacking helpers and the per-period
// coefficient conventions. What is left, and what nothing before this exercises,
// is the two of them at once:
//
//   1. The factor path with all four of the band sampler's per-period arguments
//      stacked -- a loading matrix, a measurement covariance, a transition and a
//      transition covariance, each its own per period. Two of the four are
//      shifted on the way in and two are not, and this is the only model in which
//      getting that wrong for one of them cannot be masked by the other being
//      constant. Checked against a dense posterior built with every one of them
//      indexed at period t, where there is no room for an off-by-one.
//   2. Recovery, and specifically that the two halves are told apart. A series
//      whose loading fell looks like a series whose idiosyncratic variance rose;
//      a model that carries both has to put each where it belongs rather than
//      splitting the difference.
//
// Plus what validate() has to refuse. No fixture and no file: everything is
// built in memory.

#include "bayests/dfm_tvp_stochvol.h"
#include "core/models/dfm_support.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{

using bayests::core::draw_factor_path;
using bayests::core::fill_lambda;
using bayests::core::fill_stacked_diagonal;
using bayests::core::fill_stacked_loadings;
using bayests::core::fill_stacked_transition;
using bayests::core::identified_loadings;
using bayests::core::stacked_identified_loadings;

int failures = 0;

void check(const std::string &what, const bool ok, const std::string &detail = "")
{
    failures += ok ? 0 : 1;
    std::printf("  %-52s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!detail.empty())
    {
        std::printf("      %s\n", detail.c_str());
    }
}

void check_below(const std::string &what, const double got, const double bound)
{
    check(what, got <= bound,
          "got " + std::to_string(got) + ", allowed " + std::to_string(bound));
}

double max_abs_diff(const arma::mat &lhs, const arma::mat &rhs)
{
    if (lhs.n_rows != rhs.n_rows || lhs.n_cols != rhs.n_cols)
    {
        return arma::datum::inf;
    }
    return lhs.n_elem == 0 ? 0.0 : arma::abs(lhs - rhs).max();
}

/// A loading path that moves: n_lambda x tt, every element on its own ramp.
arma::mat moving_loadings(const int n_lambda, const int tt)
{
    arma::mat path(n_lambda, tt);
    for (int i = 0; i < n_lambda; i++)
    {
        for (int t = 0; t < tt; t++)
        {
            const double share = static_cast<double>(t) / static_cast<double>(tt - 1);
            path(i, t) = 0.9 - 0.25 * static_cast<double>(i % 3) + (0.1 * (i % 2) - 0.3) * share;
        }
    }
    return path;
}

/// A transition path that moves: (N^2 p) x tt, decaying over the sample.
arma::mat moving_transition(const int n, const int p, const int tt)
{
    arma::mat path(n * n * p, tt);
    for (int t = 0; t < tt; t++)
    {
        const double share = static_cast<double>(t) / static_cast<double>(tt - 1);
        arma::mat a(n, n * p, arma::fill::zeros);
        a(0, 0) = 0.6 - 0.4 * share;
        if (n > 1)
        {
            a(1, 1) = 0.3 + 0.2 * share;
            a(1, 0) = 0.15;
        }
        if (p > 1)
        {
            a(0, n) = -0.2 + 0.1 * share;
        }
        path.col(t) = arma::vectorise(a);
    }
    return path;
}

//////////////////////////////////////////////////////////////////////////////
// 1. The factor block against the dense posterior, everything moving.

struct DensePosterior
{
    arma::vec mean;
    arma::mat cov;
};

/// The factor path's posterior with all four per-period objects, formed densely:
///
///   K = H' blkdiag(V_t^-1) H + blkdiag(Lambda_t' U_t^-1 Lambda_t),
///   b = blkdiag(Lambda_t' U_t^-1) vec(X),
///
/// with H unit block lower triangular carrying -A_{j,t} in row block t, and
/// truncated at the top because the factors before the sample are zero.
///
/// Row block t of H f is the transition residual of period t, so both A_t and
/// V_t sit at index t and there is no room for an off-by-one anywhere in this
/// construction. That is what makes it worth comparing against: the sampler
/// reaches the same distribution through a prior-plus-transitions decomposition
/// in which *both* of those are shifted by a period and the measurement pair is
/// not, and agreement says each shift is applied exactly once, in the right
/// direction, and to the right two of the four.
DensePosterior dense_factor_posterior(const arma::mat &x_t, const arma::mat &lambda_stack,
                                      const arma::mat &u_precision,
                                      const arma::mat &v_precision, const arma::mat &a_stack,
                                      const int n, const int p)
{
    const int k = static_cast<int>(x_t.n_rows);
    const int tt = static_cast<int>(x_t.n_cols);
    const bool a_is_stacked = a_stack.n_rows != static_cast<arma::uword>(n);

    arma::mat h = arma::eye<arma::mat>(tt * n, tt * n);
    for (int t = 0; t < tt; t++)
    {
        const arma::mat a_t =
            a_is_stacked ? arma::mat(a_stack.rows(t * n, (t + 1) * n - 1)) : a_stack;
        for (int j = 1; j <= p && j <= t; j++)
        {
            h.submat(t * n, (t - j) * n, (t + 1) * n - 1, (t - j + 1) * n - 1) =
                -a_t.cols((j - 1) * n, j * n - 1);
        }
    }

    arma::mat v_block(tt * n, tt * n, arma::fill::zeros);
    arma::mat measurement(tt * n, tt * n, arma::fill::zeros);
    arma::vec rhs(tt * n, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        v_block.submat(t * n, t * n, (t + 1) * n - 1, (t + 1) * n - 1) =
            arma::diagmat(arma::trans(v_precision.row(t)));

        const arma::mat u_prec_t = arma::diagmat(arma::trans(u_precision.row(t)));
        const arma::mat lambda_t = lambda_stack.rows(t * k, (t + 1) * k - 1);
        measurement.submat(t * n, t * n, (t + 1) * n - 1, (t + 1) * n - 1) =
            arma::trans(lambda_t) * u_prec_t * lambda_t;
        rhs.subvec(t * n, (t + 1) * n - 1) = arma::trans(lambda_t) * u_prec_t * x_t.col(t);
    }

    const arma::mat precision = arma::symmatu(h.t() * v_block * h + measurement);

    DensePosterior out;
    out.cov = arma::inv_sympd(precision);
    out.mean = out.cov * rhs;
    return out;
}

/// Draws the path many times against the dense posterior above, at three
/// transition orders, zero among them.
void the_factor_path_is_the_posterior_it_should_be()
{
    std::printf("the factor path matches the dense posterior\n");

    const int k = 4, n = 2, tt = 6;
    const int n_lambda = n * (2 * k - n - 1) / 2;
    const int reps = 30000;

    arma::arma_rng::set_seed(20260903);

    const arma::mat x_t = arma::randn<arma::mat>(k, tt);

    // All four move over the sample, and by enough that using the wrong period's
    // would show. A flat argument would let an off-by-one pass, which is why
    // none of them is flat here.
    arma::mat u_precision(tt, k);
    arma::mat v_precision(tt, n);
    for (int t = 0; t < tt; t++)
    {
        for (int i = 0; i < k; i++)
        {
            u_precision(t, i) = 2.0 + 0.9 * t + 0.4 * i;
        }
        for (int i = 0; i < n; i++)
        {
            v_precision(t, i) = 0.6 + 0.5 * t + 0.3 * i;
        }
    }

    arma::mat u_stack(k * tt, k, arma::fill::zeros);
    arma::mat v_stack(n * tt, n, arma::fill::zeros);
    fill_stacked_diagonal(u_stack, 1.0 / u_precision);
    fill_stacked_diagonal(v_stack, 1.0 / v_precision);

    arma::mat lambda_stack = stacked_identified_loadings(k, n, tt);
    fill_stacked_loadings(lambda_stack, moving_loadings(n_lambda, tt), k, n);

    for (const int p : {0, 1, 2})
    {
        arma::mat a_stack;
        if (p == 0)
        {
            a_stack = arma::zeros<arma::mat>(n, n);
        }
        else
        {
            a_stack = arma::mat(n * tt, n * p);
            fill_stacked_transition(a_stack, moving_transition(n, p, tt), n, p);
        }

        const DensePosterior want =
            dense_factor_posterior(x_t, lambda_stack, u_precision, v_precision, a_stack, n, p);

        arma::mat sample(n * tt, reps);
        for (int r = 0; r < reps; r++)
        {
            sample.col(r) = arma::vectorise(draw_factor_path(x_t, lambda_stack, u_stack, v_stack,
                                                             a_stack, n, p > 0 ? p : 1));
        }

        const arma::vec got_mean = arma::mean(sample, 1);
        const arma::mat got_cov = arma::cov(sample.t());

        const std::string at = "p = " + std::to_string(p);
        check_below(at + ", the posterior mean", max_abs_diff(got_mean, want.mean), 0.02);
        check_below(at + ", the posterior covariance", max_abs_diff(got_cov, want.cov), 0.02);
        check(at + ", and the posterior is not trivially small",
              arma::abs(want.mean).max() > 0.2 && want.cov.diag().max() > 0.05);
    }
}

//////////////////////////////////////////////////////////////////////////////
// 2. Recovery.

struct Simulated
{
    arma::mat x;      ///< tt x k
    arma::mat lambda; ///< n_lambda x tt, the free loadings that generated it
    arma::mat a;      ///< (N^2 p) x tt, the transition that generated it
    arma::mat u_h;    ///< tt x k, the idiosyncratic log-volatility
    arma::mat v_h;    ///< tt x n, the factor-innovation log-volatility
};

/// A sample in which the loadings and the volatility both move, and move in
/// opposite directions, so that a model that confused one for the other would
/// have to get at least one of them backwards.
///
/// Every loading ramps *down* over the sample while every idiosyncratic variance
/// ramps *down* as well. Those two pull the same way on the size of an observed
/// series and the opposite way on how much of it the factor explains, which is
/// the discrimination this model claims and the one a constant-loading or
/// constant-variance model cannot make.
Simulated simulate(const int k, const int n, const int p, const int tt)
{
    const int n_lambda = n * (2 * k - n - 1) / 2;

    Simulated s;
    s.lambda = moving_loadings(n_lambda, tt);
    s.a = moving_transition(n, p, tt);

    s.u_h = arma::mat(tt, k);
    s.v_h = arma::mat(tt, n);
    for (int t = 0; t < tt; t++)
    {
        const double share = static_cast<double>(t) / static_cast<double>(tt - 1);
        s.u_h.row(t).fill(std::log(0.25) - 0.8 * share);
        s.v_h.row(t).fill(0.6 * share);
    }

    arma::mat factors(n, tt, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        const arma::mat a_t = arma::reshape(s.a.col(t), n, n * p);
        arma::vec f = arma::randn<arma::vec>(n) % arma::exp(0.5 * arma::trans(s.v_h.row(t)));
        for (int j = 1; j <= p && j <= t; j++)
        {
            f += a_t.cols((j - 1) * n, j * n - 1) * factors.col(t - j);
        }
        factors.col(t) = f;
    }

    arma::mat x(k, tt);
    for (int t = 0; t < tt; t++)
    {
        arma::mat lambda_t = identified_loadings(k, n);
        fill_lambda(lambda_t, s.lambda.col(t));
        x.col(t) = lambda_t * factors.col(t) +
                   arma::randn<arma::vec>(k) % arma::exp(0.5 * arma::trans(s.u_h.row(t)));
    }
    s.x = x.t();
    return s;
}

bayests::DfmTvpStochvolInput make_input(const Simulated &s, const int k, const int n, const int p,
                                        const int tt, const int iterations, const int burnin)
{
    const int n_lambda = n * (2 * k - n - 1) / 2;
    const int n_a = n * n * p;

    bayests::DfmTvpStochvolInput in;
    in.spec.k = k;
    in.spec.p = p;
    in.spec.n_factors = n;
    in.spec.iterations = iterations;
    in.spec.burnin = burnin;

    in.train.y = s.x;

    in.lambda_prior.sigma.shape = arma::vec(n_lambda, arma::fill::value(3.0));
    in.lambda_prior.sigma.rate = arma::vec(n_lambda, arma::fill::value(0.01));
    in.lambda_prior.initial_state.mu = arma::vec(n_lambda, arma::fill::zeros);
    in.lambda_prior.initial_state.v_inv = arma::eye<arma::mat>(n_lambda, n_lambda) * 0.01;

    in.a_prior.sigma.shape = arma::vec(n_a, arma::fill::value(3.0));
    in.a_prior.sigma.rate = arma::vec(n_a, arma::fill::value(0.01));
    in.a_prior.initial_state.mu = arma::vec(n_a, arma::fill::zeros);
    in.a_prior.initial_state.v_inv = arma::eye<arma::mat>(n_a, n_a) * 0.01;

    in.u_sigma_prior.offset = arma::vec(k, arma::fill::value(1e-8));
    in.u_sigma_prior.state.sigma.shape = arma::vec(k, arma::fill::value(3.0));
    in.u_sigma_prior.state.sigma.rate = arma::vec(k, arma::fill::value(0.01));
    in.u_sigma_prior.state.initial_state.mu = arma::vec(k, arma::fill::zeros);
    in.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(k, k) * 0.1;

    in.v_sigma_prior.offset = arma::vec(n, arma::fill::value(1e-8));
    in.v_sigma_prior.state.sigma.shape = arma::vec(n, arma::fill::value(3.0));
    in.v_sigma_prior.state.sigma.rate = arma::vec(n, arma::fill::value(0.01));
    in.v_sigma_prior.state.initial_state.mu = arma::vec(n, arma::fill::zeros);
    in.v_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(n, n) * 0.1;

    in.initial.lambda = arma::mat(n_lambda, tt, arma::fill::value(0.5));
    in.initial.lambda_sigma_inv = arma::diagmat(arma::vec(n_lambda, arma::fill::value(100.0)));
    in.initial.lambda_init = arma::vec(n_lambda, arma::fill::value(0.5));

    in.initial.a = arma::mat(n_a, tt, arma::fill::zeros);
    in.initial.a_sigma_inv = arma::diagmat(arma::vec(n_a, arma::fill::value(100.0)));
    in.initial.a_init = arma::vec(n_a, arma::fill::zeros);

    in.initial.u_h = arma::mat(tt, k, arma::fill::zeros);
    in.initial.u_h_init = arma::vec(k, arma::fill::zeros);
    in.initial.u_h_sigma = arma::vec(k, arma::fill::value(0.05));
    in.initial.v_h = arma::mat(tt, n, arma::fill::zeros);
    in.initial.v_h_init = arma::vec(n, arma::fill::zeros);
    in.initial.v_h_sigma = arma::vec(n, arma::fill::value(0.05));
    return in;
}

/// The weakest of the checks, and the one whose tolerances need thinking about.
/// The seed is fixed, but `arma::randn` and `arma::randg` go through
/// `std::normal_distribution` and `std::gamma_distribution`, whose sequences the
/// standard does not pin down -- so on another library this is a fresh sample and
/// a fresh chain, not a rerun of this one. Four random walks over one sample is
/// as much as this model can be asked to separate, so everything below is checked
/// as a level and a direction of travel rather than pointwise.
void it_recovers_what_it_was_given()
{
    std::printf("recovery from a simulated sample\n");

    const int k = 8, n = 2, p = 1, tt = 500;
    const int iterations = 300, burnin = 200;
    const int n_lambda = n * (2 * k - n - 1) / 2;

    arma::arma_rng::set_seed(20260903);
    const Simulated s = simulate(k, n, p, tt);
    const bayests::DfmTvpStochvolInput in = make_input(s, k, n, p, tt, iterations, burnin);

    bayests::NullReporter reporter;
    const bayests::DfmTvpStochvolDraws out =
        bayests::DfmTvpStochvolSampler{}.draw_coefficients(in, reporter);

    check("the posterior has the shapes the model implies",
          out.lambda.n_rows == static_cast<arma::uword>(k * n * tt) &&
              out.lambda_sigma.n_rows == static_cast<arma::uword>(n_lambda) &&
              out.factors.n_rows == static_cast<arma::uword>(n * tt) &&
              out.a.n_rows == static_cast<arma::uword>(n * n * p * tt) &&
              out.a_sigma.n_rows == static_cast<arma::uword>(n * n * p) &&
              out.u_sigma_inv.n_rows == static_cast<arma::uword>(k * tt) &&
              out.v_sigma_inv.n_rows == static_cast<arma::uword>(n * tt) &&
              out.iterations() == static_cast<arma::uword>(iterations));

    // The loading path, as the free elements the simulation was written in.
    const arma::vec lambda_mean = arma::mean(out.lambda, 1);
    arma::mat lambda_path(n_lambda, tt);
    for (int t = 0; t < tt; t++)
    {
        const arma::mat lambda_t =
            arma::reshape(lambda_mean.subvec(t * k * n, (t + 1) * k * n - 1), k, n);
        int pos = 0;
        for (int i = 1; i < k; i++)
        {
            const int width = std::min(i, n);
            lambda_path.submat(pos, t, pos + width - 1, t) =
                arma::trans(lambda_t.submat(i, 0, i, width - 1));
            pos += width;
        }
    }

    const int quarter = tt / 4;
    const double lambda_early = arma::mean(arma::vectorise(lambda_path.head_cols(quarter)));
    const double lambda_late = arma::mean(arma::vectorise(lambda_path.tail_cols(quarter)));

    check_below("the average loading", std::abs(arma::mean(arma::vectorise(lambda_path)) -
                                                arma::mean(arma::vectorise(s.lambda))),
                0.20);
    check("the loadings are found to fall", lambda_late < lambda_early - 0.05,
          "early " + std::to_string(lambda_early) + ", late " + std::to_string(lambda_late));

    // The two volatilities, as the level they averaged and the direction they
    // moved. The stored object is the precision, so the log variance is minus
    // its log.
    const arma::mat u_log_variance =
        -arma::log(arma::reshape(arma::mean(out.u_sigma_inv, 1), k, tt));
    const arma::mat v_log_variance =
        -arma::log(arma::reshape(arma::mean(out.v_sigma_inv, 1), n, tt));

    check_below("the average idiosyncratic log variance",
                std::abs(arma::mean(arma::vectorise(u_log_variance)) -
                         arma::mean(arma::vectorise(s.u_h))),
                0.40);

    const double u_early = arma::mean(arma::vectorise(u_log_variance.head_cols(quarter)));
    const double u_late = arma::mean(arma::vectorise(u_log_variance.tail_cols(quarter)));
    const double v_early = arma::mean(arma::vectorise(v_log_variance.head_cols(quarter)));
    const double v_late = arma::mean(arma::vectorise(v_log_variance.tail_cols(quarter)));

    // The claim this model exists for: the loadings fell *and* the idiosyncratic
    // volatility fell *and* the factor volatility rose, over one sample. A model
    // carrying only one of the three has to explain the others with what it has,
    // and would fail at least one of these.
    check("the idiosyncratic volatility is found to fall", u_late < u_early - 0.2,
          "early " + std::to_string(u_early) + ", late " + std::to_string(u_late));
    check("the factor innovation volatility is found to rise", v_late > v_early + 0.2,
          "early " + std::to_string(v_early) + ", late " + std::to_string(v_late));

    // The identifying block is not drawn, so it has to be exactly itself in
    // every period of every stored draw.
    const arma::mat first_period = arma::reshape(out.lambda.submat(0, 0, k * n - 1, 0), k, n);
    arma::mat want = arma::eye<arma::mat>(n, n);
    want(1, 0) = first_period(1, 0); // the one free element of the block
    check_below("the identifying block survives every draw",
                max_abs_diff(first_period.submat(0, 0, n - 1, n - 1), want), 0.0);

    // The forecast, from the terminal coefficients the io layer cuts out. The two
    // precisions are handed over whole, which the sampler takes the last block
    // of -- and has to give the same answer as being handed that block alone.
    bayests::DfmTvpStochvolInput fcst_in = in;
    fcst_in.spec.h = 5;

    bayests::DfmTvpStochvolDraws terminal = out;
    terminal.lambda = out.lambda.tail_rows(k * n);
    terminal.a = out.a.tail_rows(n * n * p);

    const bayests::ForecastDraws fcst =
        bayests::DfmTvpStochvolSampler{}.forecast(fcst_in, terminal, reporter);
    check("the forecast is (h k) x draws",
          fcst.values.n_rows == static_cast<arma::uword>(5 * k) &&
              fcst.values.n_cols == static_cast<arma::uword>(iterations));
    check("the forecast is finite", fcst.values.is_finite());

    bayests::DfmTvpStochvolDraws cut = terminal;
    cut.u_sigma_inv = out.u_sigma_inv.tail_rows(k);
    cut.v_sigma_inv = out.v_sigma_inv.tail_rows(n);
    arma::arma_rng::set_seed(5);
    const arma::mat from_path =
        bayests::DfmTvpStochvolSampler{}.forecast(fcst_in, terminal, reporter).values;
    arma::arma_rng::set_seed(5);
    const arma::mat from_terminal =
        bayests::DfmTvpStochvolSampler{}.forecast(fcst_in, cut, reporter).values;
    check_below("the terminal precision alone gives the same forecast",
                max_abs_diff(from_path, from_terminal), 0.0);

    // And that the whole coefficient path is refused rather than read as its
    // first period, which is what a caller that skipped the cut would hand over.
    bool threw = false;
    try
    {
        bayests::DfmTvpStochvolSampler{}.forecast(fcst_in, out, reporter);
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check("a forecast from the whole coefficient path is refused", threw);

    const arma::mat loglik = bayests::DfmTvpStochvolSampler{}.log_likelihood(in, out);
    check("the log likelihood is draws x periods",
          loglik.n_rows == static_cast<arma::uword>(iterations) &&
              loglik.n_cols == static_cast<arma::uword>(tt));
    check("the log likelihood is finite", loglik.is_finite());

    // The mirror: the likelihood scores every period under its own loadings and
    // its own precision, so neither terminal slice is enough for it.
    threw = false;
    try
    {
        bayests::DfmTvpStochvolSampler{}.log_likelihood(in, terminal);
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check("a log likelihood from the terminal period alone is refused", threw);
}

//////////////////////////////////////////////////////////////////////////////
// 3. What validate() has to refuse.

void expect_rejected(const std::string &what, const bayests::DfmTvpStochvolInput &in)
{
    bool threw = false;
    std::string message;
    try
    {
        in.validate();
    }
    catch (const std::exception &e)
    {
        threw = true;
        message = e.what();
    }
    check(what, threw, message);
}

void the_impossible_inputs_are_refused()
{
    std::printf("inputs that do not describe a model\n");

    const int k = 6, n = 2, p = 1, tt = 40;
    const int n_lambda = n * (2 * k - n - 1) / 2;

    arma::arma_rng::set_seed(1);
    const Simulated s = simulate(k, n, p, tt);
    const bayests::DfmTvpStochvolInput good = make_input(s, k, n, p, tt, 20, 10);

    bool accepted = true;
    try
    {
        good.validate();
    }
    catch (const std::exception &e)
    {
        accepted = false;
        std::printf("      unexpectedly rejected: %s\n", e.what());
    }
    check("a well-formed input is accepted", accepted);

    bayests::DfmTvpStochvolInput no_factors = good;
    no_factors.spec.n_factors = 0;
    expect_rejected("a model with no factors", no_factors);

    bayests::DfmTvpStochvolInput too_many = good;
    too_many.spec.n_factors = k + 1;
    expect_rejected("more factors than series", too_many);

    bayests::DfmTvpStochvolInput varsel = good;
    varsel.spec.varsel = bayests::VarSelection::bvs;
    expect_rejected("variable selection, which is not implemented", varsel);

    bayests::DfmTvpStochvolInput structural = good;
    structural.spec.structural = true;
    expect_rejected("a structural factor model", structural);

    // This model carries four widths -- n_lambda, n_factor_a, k and n_factors --
    // and `shape`/`rate` at every one of them, which is the pair of confusions a
    // file is most likely to make.
    bayests::DfmTvpStochvolInput lambda_width = good;
    lambda_width.initial.lambda = arma::mat(n * n * p, tt, arma::fill::zeros);
    expect_rejected("a loading path of the transition's width", lambda_width);

    bayests::DfmTvpStochvolInput a_width = good;
    a_width.initial.a = arma::mat(n_lambda, tt, arma::fill::zeros);
    expect_rejected("a transition path of the loadings' width", a_width);

    bayests::DfmTvpStochvolInput u_width = good;
    u_width.initial.u_h = arma::mat(tt, n, arma::fill::zeros);
    expect_rejected("an idiosyncratic volatility path of the factors' width", u_width);

    bayests::DfmTvpStochvolInput v_width = good;
    v_width.initial.v_h = arma::mat(tt, k, arma::fill::zeros);
    expect_rejected("a factor volatility path of the series' width", v_width);

    bayests::DfmTvpStochvolInput a_prior_width = good;
    a_prior_width.a_prior.sigma.shape = arma::vec(n_lambda, arma::fill::value(3.0));
    expect_rejected("a transition prior of the loadings' width", a_prior_width);

    bayests::DfmTvpStochvolInput v_prior_width = good;
    v_prior_width.v_sigma_prior.state.sigma.shape = arma::vec(k, arma::fill::value(3.0));
    expect_rejected("a factor volatility prior of the series' width", v_prior_width);

    bayests::DfmTvpStochvolInput zero_sigma = good;
    zero_sigma.initial.v_h_sigma = arma::vec(n, arma::fill::zeros);
    expect_rejected("a zero variance of the log-volatility innovations", zero_sigma);

    // One period leaves all four random walks nothing to difference. Everything
    // else has to be made consistent with a sample of one, or a different check
    // fires first and this passes for the wrong reason.
    bayests::DfmTvpStochvolInput one_period = good;
    one_period.spec.p = 0;
    one_period.train.y = s.x.head_rows(1);
    one_period.initial.lambda = arma::mat(n_lambda, 1, arma::fill::value(0.5));
    one_period.initial.u_h = arma::mat(1, k, arma::fill::zeros);
    one_period.initial.v_h = arma::mat(1, n, arma::fill::zeros);
    expect_rejected("a sample of one period", one_period);
}

} // namespace

int main()
{
    std::printf("DfmTvpStochvol\n\n");

    the_factor_path_is_the_posterior_it_should_be();
    it_recovers_what_it_was_given();
    the_impossible_inputs_are_refused();

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
