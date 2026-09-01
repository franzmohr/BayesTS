// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That DfmNormalStochvol draws the model it says it does.
//
// unit_dfm_normal_gamma.cpp already covers everything the two models share --
// the identification of Lambda, the layout of the lagged factors, the transition
// residual -- so this covers only what the volatility changed, in the same
// decreasing order of exactness:
//
//   1. The conventions the per-period covariances introduce. Two of them are
//      places where a plausible mistake still runs and quietly estimates a
//      different model: the period a covariance belongs to, which
//      chan_jeliazkov_2009 indexes one off from this model, and the Kronecker
//      sum that no longer collapses.
//   2. The factor block against the dense posterior, built with a covariance per
//      period. This is what pins the period alignment exactly rather than
//      approximately: the dense construction indexes V_t at period t with no room
//      for an off-by-one, so if the banded sweep agrees, the shift is right.
//   3. Recovery. A sample simulated from known parameters and a moving
//      volatility, and a chain that has to find both.
//
// Plus what validate() has to refuse. No fixture and no file: everything is
// built in memory.

#include "bayests/dfm_normal_stochvol.h"
#include "core/models/dfm_support.h"

#include <cstdio>
#include <string>

namespace
{

using bayests::core::accumulate_transition_moments;
using bayests::core::draw_factor_path_sv;
using bayests::core::fill_lagged_factors;
using bayests::core::fill_lambda;
using bayests::core::fill_stacked_diagonal;
using bayests::core::identified_loadings;
using bayests::core::initial_state_covariance;

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

/// The stacked covariance a period-by-period diagonal variance implies, built
/// the obvious way: one diagonal block per period, laid out by hand.
arma::mat stacked_by_hand(const arma::mat &variance)
{
    const arma::uword tt = variance.n_rows;
    const arma::uword k = variance.n_cols;
    arma::mat out(k * tt, k, arma::fill::zeros);
    for (arma::uword t = 0; t < tt; t++)
    {
        out.rows(t * k, (t + 1) * k - 1) = arma::diagmat(arma::trans(variance.row(t)));
    }
    return out;
}

//////////////////////////////////////////////////////////////////////////////
// 1. The conventions.

/// Period t's variances go in block t, and nothing else is touched.
void the_stacked_covariance_is_period_by_period()
{
    std::printf("the covariance stack is one diagonal block per period\n");

    const arma::uword k = 3, tt = 4;
    arma::mat variance(tt, k);
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            variance(t, i) = 1.0 + static_cast<double>(t) + 0.1 * static_cast<double>(i);
        }
    }

    arma::mat stack(k * tt, k, arma::fill::zeros);
    fill_stacked_diagonal(stack, variance);

    check_below("block t holds diag(variance.row(t))",
                max_abs_diff(stack, stacked_by_hand(variance)), 0.0);

    // The buffer is zeroed once by the caller and never again, so a second fill
    // over a different variance has to leave nothing of the first behind.
    fill_stacked_diagonal(stack, variance * 2.0);
    check_below("a refill leaves nothing of the previous one",
                max_abs_diff(stack, stacked_by_hand(variance * 2.0)), 0.0);
}

/// The prior over the first p factors uses those p periods' own covariances, not
/// the first period's p times over. At p = 1 the two coincide, which is why this
/// needs p = 2 to say anything.
void the_prior_over_the_first_factors_uses_each_period()
{
    std::printf("the prior over the first p factors takes a covariance each\n");

    const int n = 2;
    const arma::mat v0 = arma::diagmat(arma::vec{2.0, 0.5});
    const arma::mat v1 = arma::diagmat(arma::vec{0.25, 3.0});

    arma::mat stack(2 * n, n);
    stack.rows(0, n - 1) = v0;
    stack.rows(n, 2 * n - 1) = v1;

    const arma::mat a1 = {{0.6, 0.2}, {-0.1, 0.4}};
    const arma::mat got = initial_state_covariance(a1, stack, n, 2);

    // f_1 = v_1 and f_2 = A_1 v_1 + v_2, hence
    //   [[V_1, V_1 A_1'], [A_1 V_1, A_1 V_1 A_1' + V_2]].
    arma::mat want(2 * n, 2 * n);
    want.submat(0, 0, n - 1, n - 1) = v0;
    want.submat(0, n, n - 1, 2 * n - 1) = v0 * a1.t();
    want.submat(n, 0, 2 * n - 1, n - 1) = a1 * v0;
    want.submat(n, n, 2 * n - 1, 2 * n - 1) = a1 * v0 * a1.t() + v1;
    check_below("each of the two periods contributes its own V",
                max_abs_diff(got, want), 1e-13);

    // And the constant spelling still means what it used to: one N x N argument
    // is I_p kron V, which is the stack with both blocks equal.
    arma::mat repeated(2 * n, n);
    repeated.rows(0, n - 1) = v0;
    repeated.rows(n, 2 * n - 1) = v0;
    check_below("one covariance for every period is the old behaviour",
                max_abs_diff(initial_state_covariance(a1, v0, n, 2),
                             initial_state_covariance(a1, repeated, n, 2)),
                0.0);
}

/// The transition's posterior precision, against the Kronecker sum it replaces.
///
/// This is the identity the model turns on and the one piece of index arithmetic
/// in it. A transposed scatter, or one that reads row i of A as positions
/// i*Np .. (i+1)*Np - 1 instead of i, N + i, 2N + i, ..., still runs and still
/// produces a symmetric positive definite matrix.
void the_transition_moments_are_the_kronecker_sum()
{
    std::printf("the transition's moments are the Kronecker sum\n");

    const int n = 3, p = 2, tt = 7;
    arma::arma_rng::set_seed(11);

    const arma::mat factors = arma::randn<arma::mat>(n, tt);
    arma::mat x_a(n * p, tt);
    fill_lagged_factors(x_a, factors, n, p);

    // Deliberately different in every period and every factor: a per-period
    // precision that happened to be constant would let the collapsed spelling
    // pass as well.
    arma::mat v_precision(tt, n);
    for (int t = 0; t < tt; t++)
    {
        for (int i = 0; i < n; i++)
        {
            v_precision(t, i) = 0.5 + 0.3 * t + 0.7 * i;
        }
    }

    const int n_a = n * n * p;
    arma::mat got_precision(n_a, n_a, arma::fill::zeros);
    arma::vec got_rhs(n_a, arma::fill::zeros);
    accumulate_transition_moments(got_precision, got_rhs, x_a, factors, v_precision, n, p);

    // sum_t kron(x_t x_t', V_t^-1) and vec(sum_t V_t^-1 f_t x_t').
    arma::mat want_precision(n_a, n_a, arma::fill::zeros);
    arma::mat want_rhs_mat(n, n * p, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        const arma::mat v_prec_t = arma::diagmat(arma::trans(v_precision.row(t)));
        want_precision += arma::kron(x_a.col(t) * arma::trans(x_a.col(t)), v_prec_t);
        want_rhs_mat += v_prec_t * factors.col(t) * arma::trans(x_a.col(t));
    }

    check_below("the precision is sum_t kron(x_t x_t', V_t^-1)",
                max_abs_diff(got_precision, want_precision), 1e-11);
    check_below("the right-hand side is vec(sum_t V_t^-1 f_t x_t')",
                max_abs_diff(got_rhs, arma::vectorise(want_rhs_mat)), 1e-11);

    // As above: a tolerance means nothing without the scale it is measured on.
    check("and the moments are not trivially small",
          arma::abs(want_precision).max() > 1.0 && arma::abs(want_rhs_mat).max() > 0.5);

    // The off-diagonal blocks a Kronecker product would fill with zeros really
    // are zero, which is the reason the cheap spelling is available at all.
    check("equations of different factors do not interact",
          std::abs(want_precision(0, 1)) < 1e-12 && std::abs(got_precision(0, 1)) < 1e-12);
}

/// A factor innovation that can only be large in one period has to be large in
/// that one -- the alignment check, stated so that dropping the shift in
/// draw_factor_path_sv fails it by a period.
void the_volatility_lands_in_its_own_period()
{
    std::printf("a variance large in one period gives one large factor\n");

    const int k = 2, n = 1, tt = 8;
    const arma::mat lambda = identified_loadings(k, n);

    // No dynamics and an all but uninformative measurement, so the path is the
    // innovations and nothing else.
    const arma::mat a_mat = arma::zeros<arma::mat>(n, n);
    const arma::mat x_t = arma::zeros<arma::mat>(k, tt);

    arma::mat u_variance(tt, k, arma::fill::value(1e8));
    arma::mat u_stack(k * tt, k, arma::fill::zeros);
    fill_stacked_diagonal(u_stack, u_variance);

    for (const int loud : {0, 3, tt - 1})
    {
        arma::mat v_variance(tt, n, arma::fill::value(1e-10));
        v_variance(loud, 0) = 1.0;

        arma::mat v_stack(n * tt, n, arma::fill::zeros);
        fill_stacked_diagonal(v_stack, v_variance);

        arma::arma_rng::set_seed(7);
        double quiet = 0.0, at = 0.0;
        for (int r = 0; r < 200; r++)
        {
            const arma::mat path =
                draw_factor_path_sv(x_t, lambda, u_stack, v_stack, a_mat, n, 1);
            for (int t = 0; t < tt; t++)
            {
                const double size = std::abs(path(0, t));
                if (t == loud) { at = std::max(at, size); }
                else { quiet = std::max(quiet, size); }
            }
        }

        const std::string label = "period " + std::to_string(loud);
        check_below(label + ": every other period stays at zero", quiet, 1e-3);
        check(label + ": and that one moves",
              at > 0.5, "largest |f| there was " + std::to_string(at));
    }
}

//////////////////////////////////////////////////////////////////////////////
// 2. The factor block against the dense posterior.

struct DensePosterior
{
    arma::vec mean;
    arma::mat cov;
};

/// The factor path's posterior with a covariance per period, formed densely:
///
///   K = H' blkdiag(V_t^-1) H + blkdiag(Lambda' U_t^-1 Lambda),
///   b = blkdiag(Lambda' U_t^-1) vec(X),
///
/// with H unit block lower triangular carrying -A_j on its j-th subdiagonal,
/// truncated at the top because the factors before the sample are zero.
///
/// Row block t of H f is the transition residual of period t, so V_t^-1 sits in
/// block t of the middle matrix and there is no room for an off-by-one anywhere
/// in this construction. That is what makes it worth comparing against: the
/// sampler reaches the same distribution through a prior-plus-transitions
/// decomposition whose covariance indexing is shifted by a period, and agreement
/// says the shift is applied exactly once and in the right direction.
DensePosterior dense_factor_posterior(const arma::mat &x_t, const arma::mat &lambda,
                                      const arma::mat &u_precision,
                                      const arma::mat &v_precision, const arma::mat &a_mat,
                                      const int n, const int p)
{
    const int tt = static_cast<int>(x_t.n_cols);

    arma::mat h = arma::eye<arma::mat>(tt * n, tt * n);
    for (int t = 0; t < tt; t++)
    {
        for (int j = 1; j <= p && j <= t; j++)
        {
            h.submat(t * n, (t - j) * n, (t + 1) * n - 1, (t - j + 1) * n - 1) =
                -a_mat.cols((j - 1) * n, j * n - 1);
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
        measurement.submat(t * n, t * n, (t + 1) * n - 1, (t + 1) * n - 1) =
            arma::trans(lambda) * u_prec_t * lambda;
        rhs.subvec(t * n, (t + 1) * n - 1) = arma::trans(lambda) * u_prec_t * x_t.col(t);
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
    const int reps = 30000;

    arma::arma_rng::set_seed(20260901);

    arma::mat lambda = identified_loadings(k, n);
    fill_lambda(lambda, arma::vec{0.7, -0.4, 0.9, 0.3, -0.8});
    const arma::mat x_t = arma::randn<arma::mat>(k, tt);

    // Both volatilities move over the sample, and by enough that using the wrong
    // period's would show. A flat path would let an off-by-one pass.
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

    for (const int p : {0, 1, 2})
    {
        arma::mat a_mat;
        if (p == 0)
        {
            a_mat = arma::zeros<arma::mat>(n, n);
        }
        else
        {
            a_mat = arma::mat(n, n * p, arma::fill::zeros);
            a_mat(0, 0) = 0.5;
            a_mat(1, 0) = 0.15;
            a_mat(1, 1) = 0.35;
            if (p == 2)
            {
                a_mat(0, 2) = -0.2;
                a_mat(1, 3) = 0.25;
            }
        }

        const DensePosterior want =
            dense_factor_posterior(x_t, lambda, u_precision, v_precision, a_mat, n, p);

        arma::mat sample(n * tt, reps);
        for (int r = 0; r < reps; r++)
        {
            sample.col(r) = arma::vectorise(
                draw_factor_path_sv(x_t, lambda, u_stack, v_stack, a_mat, n, p > 0 ? p : 1));
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
// 3. Recovery.

struct Simulated
{
    arma::mat x;         ///< tt x k
    arma::mat lambda;    ///< k x n
    arma::mat a;         ///< n x np
    arma::mat u_h;       ///< tt x k, the log-volatility that generated it
    arma::mat v_h;       ///< tt x n
};

/// A sample whose volatility really does move: both log-volatilities are given a
/// deterministic swing over the sample rather than a random walk, so what the
/// chain has to find is a known path and not one more draw.
Simulated simulate(const int k, const int n, const int p, const int tt)
{
    Simulated s;
    s.lambda = identified_loadings(k, n);
    arma::vec free(static_cast<arma::uword>(n * (2 * k - n - 1) / 2));
    for (arma::uword i = 0; i < free.n_elem; i++)
    {
        free(i) = 0.9 - 0.25 * static_cast<double>(i % 5);
    }
    fill_lambda(s.lambda, free);

    s.a = arma::mat(n, n * p, arma::fill::zeros);
    s.a(0, 0) = 0.7;
    s.a(1, 1) = 0.45;
    s.a(1, 0) = 0.2;

    // The idiosyncratic variance halves over the sample; the factor innovation
    // variance doubles. Both swings are well inside what a random walk of the
    // variance the prior below expects would produce.
    s.u_h = arma::mat(tt, k);
    s.v_h = arma::mat(tt, n);
    for (int t = 0; t < tt; t++)
    {
        const double share = static_cast<double>(t) / static_cast<double>(tt - 1);
        s.u_h.row(t).fill(std::log(0.25) - 0.7 * share);
        s.v_h.row(t).fill(0.7 * share);
    }

    arma::mat factors(n, tt, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        arma::vec f = arma::randn<arma::vec>(n) % arma::exp(0.5 * arma::trans(s.v_h.row(t)));
        if (t > 0)
        {
            f += s.a.cols(0, n - 1) * factors.col(t - 1);
        }
        factors.col(t) = f;
    }

    arma::mat x = s.lambda * factors;
    for (int t = 0; t < tt; t++)
    {
        x.col(t) += arma::randn<arma::vec>(k) % arma::exp(0.5 * arma::trans(s.u_h.row(t)));
    }
    s.x = x.t();
    return s;
}

bayests::DfmNormalStochvolInput make_input(const Simulated &s, const int k, const int n,
                                           const int p, const int tt, const int iterations,
                                           const int burnin)
{
    const int n_lambda = n * (2 * k - n - 1) / 2;
    const int n_a = n * n * p;

    bayests::DfmNormalStochvolInput in;
    in.spec.k = k;
    in.spec.p = p;
    in.spec.n_factors = n;
    in.spec.iterations = iterations;
    in.spec.burnin = burnin;

    in.train.y = s.x;

    in.lambda_prior.mu = arma::vec(n_lambda, arma::fill::zeros);
    in.lambda_prior.v_inv = arma::eye<arma::mat>(n_lambda, n_lambda) * 0.01;
    in.a_prior.mu = arma::vec(n_a, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(n_a, n_a) * 0.01;

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

    in.initial.lambda = arma::vec(n_lambda, arma::fill::value(0.5));
    in.initial.a = arma::vec(n_a, arma::fill::zeros);

    in.initial.u_h = arma::mat(tt, k, arma::fill::zeros);
    in.initial.u_h_init = arma::vec(k, arma::fill::zeros);
    in.initial.u_h_sigma = arma::vec(k, arma::fill::value(0.05));
    in.initial.v_h = arma::mat(tt, n, arma::fill::zeros);
    in.initial.v_h_init = arma::vec(n, arma::fill::zeros);
    in.initial.v_h_sigma = arma::vec(n, arma::fill::value(0.05));
    return in;
}

/// The weakest of the three checks, and the one whose tolerances need thinking
/// about. The seed is fixed, but `arma::randn` and `arma::randg` go through
/// `std::normal_distribution` and `std::gamma_distribution`, whose sequences the
/// standard does not pin down -- so on another library this is a fresh sample and
/// a fresh chain, not a rerun of this one. What has to be tolerated is the
/// sampling error of the whole experiment, and the volatility adds to it: a
/// log-volatility path is tt numbers estimated from tt observations, so it is
/// checked as an average level and a direction of travel rather than pointwise.
void it_recovers_what_it_was_given()
{
    std::printf("recovery from a simulated sample\n");

    const int k = 8, n = 2, p = 1, tt = 800;
    const int iterations = 600, burnin = 400;

    arma::arma_rng::set_seed(20260901);
    const Simulated s = simulate(k, n, p, tt);
    const bayests::DfmNormalStochvolInput in =
        make_input(s, k, n, p, tt, iterations, burnin);

    bayests::NullReporter reporter;
    const bayests::DfmNormalStochvolDraws out =
        bayests::DfmNormalStochvolSampler{}.draw_coefficients(in, reporter);

    check("the posterior has the shapes the model implies",
          out.lambda.n_rows == static_cast<arma::uword>(k * n) &&
              out.factors.n_rows == static_cast<arma::uword>(n * tt) &&
              out.a.n_rows == static_cast<arma::uword>(n * n * p) &&
              out.u_sigma_inv.n_rows == static_cast<arma::uword>(k * tt) &&
              out.v_sigma_inv.n_rows == static_cast<arma::uword>(n * tt) &&
              out.iterations() == static_cast<arma::uword>(iterations));

    const arma::mat lambda_mean = arma::reshape(arma::mean(out.lambda, 1), k, n);
    const arma::mat a_mean = arma::reshape(arma::mean(out.a, 1), n, n * p);

    check_below("the loadings", max_abs_diff(lambda_mean, s.lambda), 0.25);
    check_below("the transition", max_abs_diff(a_mean, s.a), 0.20);

    // The volatility, as the level it averaged and as the direction it moved.
    // The stored object is the precision, k or n values per period, so the log
    // variance is minus its log.
    const arma::mat u_log_variance =
        -arma::log(arma::reshape(arma::mean(out.u_sigma_inv, 1), k, tt));
    const arma::mat v_log_variance =
        -arma::log(arma::reshape(arma::mean(out.v_sigma_inv, 1), n, tt));

    check_below("the average idiosyncratic log variance",
                std::abs(arma::mean(arma::vectorise(u_log_variance)) -
                         arma::mean(arma::vectorise(s.u_h))),
                0.30);
    check_below("the average factor innovation log variance",
                std::abs(arma::mean(arma::vectorise(v_log_variance)) -
                         arma::mean(arma::vectorise(s.v_h))),
                0.40);

    // The whole point of the model: it has to see that one variance fell over
    // the sample and the other rose. A sampler that ignored the period index
    // would land on a flat path and fail both of these.
    const int quarter = tt / 4;
    const double u_early = arma::mean(arma::vectorise(u_log_variance.head_cols(quarter)));
    const double u_late = arma::mean(arma::vectorise(u_log_variance.tail_cols(quarter)));
    const double v_early = arma::mean(arma::vectorise(v_log_variance.head_cols(quarter)));
    const double v_late = arma::mean(arma::vectorise(v_log_variance.tail_cols(quarter)));

    check("the idiosyncratic volatility is found to fall", u_late < u_early - 0.2,
          "early " + std::to_string(u_early) + ", late " + std::to_string(u_late));
    check("the factor innovation volatility is found to rise", v_late > v_early + 0.2,
          "early " + std::to_string(v_early) + ", late " + std::to_string(v_late));

    // The identifying block is not drawn, so it has to be exactly itself in
    // every stored draw and not merely on average.
    const arma::mat first_draw = arma::reshape(out.lambda.col(0), k, n);
    const arma::mat identification = first_draw.submat(0, 0, n - 1, n - 1);
    arma::mat want = arma::eye<arma::mat>(n, n);
    want(1, 0) = identification(1, 0); // the one free element of the block
    check_below("the identifying block survives every draw",
                max_abs_diff(identification, want), 0.0);

    // The forecast, from the whole path and from the terminal period alone --
    // the two shapes a host may hand over, which have to be accepted alike.
    bayests::DfmNormalStochvolInput fcst_in = in;
    fcst_in.spec.h = 5;
    const bayests::ForecastDraws fcst =
        bayests::DfmNormalStochvolSampler{}.forecast(fcst_in, out, reporter);
    check("the forecast is (h k) x draws",
          fcst.values.n_rows == static_cast<arma::uword>(5 * k) &&
              fcst.values.n_cols == static_cast<arma::uword>(iterations));
    check("the forecast is finite", fcst.values.is_finite());

    bayests::DfmNormalStochvolDraws terminal = out;
    terminal.u_sigma_inv = out.u_sigma_inv.tail_rows(k);
    terminal.v_sigma_inv = out.v_sigma_inv.tail_rows(n);
    arma::arma_rng::set_seed(5);
    const arma::mat from_path =
        bayests::DfmNormalStochvolSampler{}.forecast(fcst_in, out, reporter).values;
    arma::arma_rng::set_seed(5);
    const arma::mat from_terminal =
        bayests::DfmNormalStochvolSampler{}.forecast(fcst_in, terminal, reporter).values;
    check_below("the terminal period alone gives the same forecast",
                max_abs_diff(from_path, from_terminal), 0.0);

    const arma::mat loglik =
        bayests::DfmNormalStochvolSampler{}.log_likelihood(in, out);
    check("the log likelihood is draws x periods",
          loglik.n_rows == static_cast<arma::uword>(iterations) &&
              loglik.n_cols == static_cast<arma::uword>(tt));
    check("the log likelihood is finite", loglik.is_finite());
}

//////////////////////////////////////////////////////////////////////////////
// 4. What validate() has to refuse.

void expect_rejected(const std::string &what, const bayests::DfmNormalStochvolInput &in)
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
    arma::arma_rng::set_seed(1);
    const Simulated s = simulate(k, n, p, tt);
    const bayests::DfmNormalStochvolInput good = make_input(s, k, n, p, tt, 20, 10);

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

    bayests::DfmNormalStochvolInput no_factors = good;
    no_factors.spec.n_factors = 0;
    expect_rejected("a model with no factors", no_factors);

    bayests::DfmNormalStochvolInput too_many = good;
    too_many.spec.n_factors = k + 1;
    expect_rejected("more factors than series", too_many);

    bayests::DfmNormalStochvolInput varsel = good;
    varsel.spec.varsel = bayests::VarSelection::bvs;
    expect_rejected("variable selection, which is not implemented", varsel);

    bayests::DfmNormalStochvolInput structural = good;
    structural.spec.structural = true;
    expect_rejected("a structural factor model", structural);

    // The two widths a factor model carries, and the pair a file is most likely
    // to swap: the idiosyncratic block is k wide and the factor block n_factors.
    bayests::DfmNormalStochvolInput u_width = good;
    u_width.initial.u_h = arma::mat(tt, n, arma::fill::zeros);
    expect_rejected("an idiosyncratic volatility path of the factors' width", u_width);

    bayests::DfmNormalStochvolInput v_width = good;
    v_width.initial.v_h = arma::mat(tt, k, arma::fill::zeros);
    expect_rejected("a factor volatility path of the series' width", v_width);

    bayests::DfmNormalStochvolInput short_path = good;
    short_path.initial.u_h = arma::mat(tt - 1, k, arma::fill::zeros);
    expect_rejected("a volatility path shorter than the sample", short_path);

    bayests::DfmNormalStochvolInput v_prior_width = good;
    v_prior_width.v_sigma_prior.state.sigma.shape = arma::vec(k, arma::fill::value(3.0));
    expect_rejected("a factor volatility prior of the series' width", v_prior_width);

    bayests::DfmNormalStochvolInput zero_sigma = good;
    zero_sigma.initial.v_h_sigma = arma::vec(n, arma::fill::zeros);
    expect_rejected("a zero variance of the log-volatility innovations", zero_sigma);

    // One period leaves the random walk nothing to difference. Everything else
    // has to be made consistent with a sample of one, or a different check fires
    // first and this passes for the wrong reason -- the transition order, which
    // has to be under the sample length, and the two paths, whose shape is
    // checked before the floor is.
    bayests::DfmNormalStochvolInput one_period = good;
    one_period.spec.p = 0;
    one_period.train.y = s.x.head_rows(1);
    one_period.initial.u_h = arma::mat(1, k, arma::fill::zeros);
    one_period.initial.v_h = arma::mat(1, n, arma::fill::zeros);
    expect_rejected("a sample of one period", one_period);
}

} // namespace

int main()
{
    std::printf("DfmNormalStochvol\n\n");

    the_stacked_covariance_is_period_by_period();
    the_prior_over_the_first_factors_uses_each_period();
    the_transition_moments_are_the_kronecker_sum();
    the_volatility_lands_in_its_own_period();
    the_factor_path_is_the_posterior_it_should_be();
    it_recovers_what_it_was_given();
    the_impossible_inputs_are_refused();

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
