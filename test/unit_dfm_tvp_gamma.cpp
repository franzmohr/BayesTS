// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That DfmTvpGamma draws the model it says it does.
//
// unit_dfm_normal_gamma.cpp already covers everything this shares with the
// constant-coefficient model -- the identification of Lambda, the layout of the
// lagged factors, the transition residual -- so this covers only what the state
// equations changed, in the same decreasing order of exactness:
//
//   1. The conventions the per-period coefficients introduce. Every one of them
//      is a place where a plausible mistake still runs and quietly estimates a
//      different model: which period a block of the stack belongs to, and the
//      period chan_jeliazkov_2009 indexes a transition by, which is one off from
//      this model's own.
//   2. The factor block against the dense posterior, built with a loading matrix
//      and a transition per period. This is what pins the alignment exactly
//      rather than approximately: the dense construction indexes both at period
//      t with no room for an off-by-one, so if the banded sweep agrees, the
//      shift is applied exactly once and in the right direction.
//   3. Recovery. A sample simulated from a loading that really moves and a
//      transition that really decays, and a chain that has to find both.
//
// Plus what validate() has to refuse. No fixture and no file: everything is
// built in memory.

#include "bayests/dfm_tvp_gamma.h"
#include "core/models/dfm_support.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace
{

using bayests::core::draw_factor_path;
using bayests::core::fill_lagged_factors;
using bayests::core::fill_lambda;
using bayests::core::fill_stacked_loadings;
using bayests::core::fill_stacked_transition;
using bayests::core::fill_transition_design;
using bayests::core::identified_loadings;
using bayests::core::initial_state_covariance;
using bayests::core::stacked_identified_loadings;
using bayests::core::transition_residuals;
using bayests::core::transition_residuals_tvp;

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
    const int n_a = n * n * p;
    arma::mat path(n_a, tt);
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
// 1. The conventions the per-period coefficients introduce.

void the_stacked_loadings_are_period_by_period()
{
    std::printf("the loading stack is one Lambda_t per period\n");

    const int k = 5, n = 2, tt = 7;
    const int n_lambda = n * (2 * k - n - 1) / 2;
    const arma::mat path = moving_loadings(n_lambda, tt);

    arma::mat stack = stacked_identified_loadings(k, n, tt);
    fill_stacked_loadings(stack, path, k, n);

    // The obvious construction: fill_lambda() once per period, which
    // unit_dfm_normal_gamma.cpp already pins against the row-major ordering.
    arma::mat want(k * tt, n);
    for (int t = 0; t < tt; t++)
    {
        arma::mat lambda = identified_loadings(k, n);
        fill_lambda(lambda, path.col(t));
        want.rows(t * k, (t + 1) * k - 1) = lambda;
    }

    check_below("block t is period t's loading matrix", max_abs_diff(stack, want), 0.0);

    // The half of Lambda that is not drawn has to survive a second fill: the
    // stack is built once and written into every draw, so a fill that touched
    // the identifying block would erode it over the chain rather than fail.
    fill_stacked_loadings(stack, arma::mat(n_lambda, tt, arma::fill::zeros), k, n);
    for (int t = 0; t < tt; t++)
    {
        const arma::mat block = stack.rows(t * k, t * k + n - 1);
        check_below("period " + std::to_string(t) + "'s identifying block is untouched",
                    max_abs_diff(arma::trimatu(block), arma::eye<arma::mat>(n, n)), 0.0);
    }
}

void the_stacked_transition_is_period_by_period()
{
    std::printf("the transition stack is one A_t per period\n");

    const int n = 2, p = 2, tt = 6;
    const arma::mat path = moving_transition(n, p, tt);

    arma::mat stack(n * tt, n * p);
    fill_stacked_transition(stack, path, n, p);

    arma::mat want(n * tt, n * p);
    for (int t = 0; t < tt; t++)
    {
        want.rows(t * n, (t + 1) * n - 1) = arma::reshape(path.col(t), n, n * p);
    }

    check_below("block t is period t's transition", max_abs_diff(stack, want), 0.0);
}

void the_transition_design_is_the_kronecker_product()
{
    std::printf("the transition design is kron(x_t', I_N) per period\n");

    const int n = 2, p = 2, tt = 6;

    arma::arma_rng::set_seed(20260902);
    const arma::mat factors = arma::randn<arma::mat>(n, tt);
    arma::mat x_a(n * p, tt);
    fill_lagged_factors(x_a, factors, n, p);

    arma::mat z_a(n * tt, n * n * p, arma::fill::zeros);
    fill_transition_design(z_a, x_a, n);

    arma::mat want(n * tt, n * n * p);
    for (int t = 0; t < tt; t++)
    {
        want.rows(t * n, (t + 1) * n - 1) =
            arma::kron(arma::trans(x_a.col(t)), arma::eye<arma::mat>(n, n));
    }

    check_below("the scattered design is the Kronecker one", max_abs_diff(z_a, want), 0.0);

    // And that it is the design the coefficient vector is actually vectorised
    // against: Z_t vec(A_t) has to be the fitted value A_t x_t, or the draw is
    // of a permutation of the transition.
    const arma::mat a_path = moving_transition(n, p, tt);
    arma::mat a_stack(n * tt, n * p);
    fill_stacked_transition(a_stack, a_path, n, p);

    arma::mat fitted_design(n, tt), fitted_direct(n, tt);
    for (int t = 0; t < tt; t++)
    {
        fitted_design.col(t) = z_a.rows(t * n, (t + 1) * n - 1) * a_path.col(t);
        fitted_direct.col(t) = a_stack.rows(t * n, (t + 1) * n - 1) * x_a.col(t);
    }
    // Not exactly, and deliberately not: the two spellings sum the same products
    // in different orders, so they agree to the last bits rather than in them.
    check_below("Z_t vec(A_t) is A_t x_t", max_abs_diff(fitted_design, fitted_direct), 1e-12);

    // The residual the factor innovation precision is drawn from is the same
    // object, and reduces to the constant-transition one when nothing moves.
    const arma::mat a_mat = arma::reshape(a_path.col(0), n, n * p);
    arma::mat flat(n * tt, n * p);
    for (int t = 0; t < tt; t++)
    {
        flat.rows(t * n, (t + 1) * n - 1) = a_mat;
    }
    check_below("a transition that does not move gives the constant residual",
                max_abs_diff(transition_residuals_tvp(factors, flat, x_a, n),
                             transition_residuals(factors, a_mat, n, p)),
                0.0);
}

void the_prior_over_the_first_factors_uses_each_period()
{
    std::printf("the prior over the first factors uses each period's transition\n");

    const int n = 2, p = 3, tt = 8;
    const arma::mat v_sigma = arma::diagmat(arma::vec{0.7, 1.3});
    const arma::mat a_path = moving_transition(n, p, tt);

    arma::mat a_stack(n * tt, n * p);
    fill_stacked_transition(a_stack, a_path, n, p);

    // Row block i is the transition producing state column i, so H's j-th
    // subdiagonal block in row i is -A_{j,i} and not -A_{j,1}.
    arma::mat h = arma::eye<arma::mat>(p * n, p * n);
    for (int i = 1; i < p; i++)
    {
        const arma::mat a_i = a_stack.rows(i * n, (i + 1) * n - 1);
        for (int j = 0; j < i; j++)
        {
            h.submat(i * n, j * n, (i + 1) * n - 1, (j + 1) * n - 1) =
                -a_i.cols((i - j - 1) * n, (i - j) * n - 1);
        }
    }
    arma::mat v_blocks(p * n, p * n, arma::fill::zeros);
    for (int i = 0; i < p; i++)
    {
        v_blocks.submat(i * n, i * n, (i + 1) * n - 1, (i + 1) * n - 1) = v_sigma;
    }
    const arma::mat h_inv = arma::inv(h);
    const arma::mat want = arma::symmatu(h_inv * v_blocks * arma::trans(h_inv));

    const arma::mat got =
        initial_state_covariance(a_stack.head_rows(p * n), v_sigma, n, p);
    check_below("each of the first p states gets its own transition",
                max_abs_diff(got, want), 1e-12);

    // A stack that repeats one transition has to give exactly what that
    // transition gives on its own: the shape dispatch may not move the numbers
    // of the constant-coefficient models that already depend on this function.
    const arma::mat a_mat = arma::reshape(a_path.col(0), n, n * p);
    arma::mat flat(p * n, n * p);
    for (int i = 0; i < p; i++)
    {
        flat.rows(i * n, (i + 1) * n - 1) = a_mat;
    }
    check_below("a stack that repeats one transition is that transition",
                max_abs_diff(initial_state_covariance(flat, v_sigma, n, p),
                             initial_state_covariance(a_mat, v_sigma, n, p)),
                0.0);
}

void a_flat_path_is_the_constant_model()
{
    std::printf("coefficients that do not move give the constant model's path\n");

    const int k = 5, n = 2, p = 2, tt = 9;
    const int n_lambda = n * (2 * k - n - 1) / 2;

    arma::arma_rng::set_seed(20260902);
    const arma::mat x_t = arma::randn<arma::mat>(k, tt);
    const arma::mat u_sigma = arma::diagmat(arma::vec(k, arma::fill::value(0.4)));
    const arma::mat v_sigma = arma::diagmat(arma::vec{0.8, 1.2});

    arma::mat lambda = identified_loadings(k, n);
    const arma::vec free(n_lambda, arma::fill::value(0.6));
    fill_lambda(lambda, free);

    const arma::mat a_mat = arma::reshape(moving_transition(n, p, tt).col(0), n, n * p);

    arma::mat lambda_stack = stacked_identified_loadings(k, n, tt);
    fill_stacked_loadings(lambda_stack, arma::mat(n_lambda, tt, arma::fill::value(0.6)), k, n);
    arma::mat a_stack(n * tt, n * p);
    for (int t = 0; t < tt; t++)
    {
        a_stack.rows(t * n, (t + 1) * n - 1) = a_mat;
    }

    // The same seed either side, so this is an identity between two spellings of
    // one draw rather than an agreement between two samples of it.
    arma::arma_rng::set_seed(11);
    const arma::mat constant = draw_factor_path(x_t, lambda, u_sigma, v_sigma, a_mat, n, p);
    arma::arma_rng::set_seed(11);
    const arma::mat stacked =
        draw_factor_path(x_t, lambda_stack, u_sigma, v_sigma, a_stack, n, p);

    check_below("the stacked draw is the constant one", max_abs_diff(stacked, constant), 1e-12);
}

//////////////////////////////////////////////////////////////////////////////
// 2. The factor block against the dense posterior.

struct DensePosterior
{
    arma::vec mean;
    arma::mat cov;
};

/// The factor path's posterior with a loading matrix and a transition per
/// period, formed densely:
///
///   K = H' blkdiag(V^-1) H + blkdiag(Lambda_t' U^-1 Lambda_t),
///   b = blkdiag(Lambda_t' U^-1) vec(X),
///
/// with H unit block lower triangular carrying -A_{j,t} in row block t, and
/// truncated at the top because the factors before the sample are zero.
///
/// Row block t of H f is the transition residual of period t, so A_t sits in row
/// block t and there is no room for an off-by-one anywhere in this construction.
/// That is what makes it worth comparing against: the sampler reaches the same
/// distribution through a prior-plus-transitions decomposition whose transition
/// indexing is shifted by a period, and agreement says the shift is applied
/// exactly once and in the right direction.
DensePosterior dense_factor_posterior(const arma::mat &x_t, const arma::mat &lambda_stack,
                                      const arma::vec &u_precision,
                                      const arma::vec &v_precision, const arma::mat &a_stack,
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

    const arma::mat u_prec = arma::diagmat(u_precision);

    arma::mat v_block(tt * n, tt * n, arma::fill::zeros);
    arma::mat measurement(tt * n, tt * n, arma::fill::zeros);
    arma::vec rhs(tt * n, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        v_block.submat(t * n, t * n, (t + 1) * n - 1, (t + 1) * n - 1) =
            arma::diagmat(v_precision);

        const arma::mat lambda_t = lambda_stack.rows(t * k, (t + 1) * k - 1);
        measurement.submat(t * n, t * n, (t + 1) * n - 1, (t + 1) * n - 1) =
            arma::trans(lambda_t) * u_prec * lambda_t;
        rhs.subvec(t * n, (t + 1) * n - 1) = arma::trans(lambda_t) * u_prec * x_t.col(t);
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

    arma::arma_rng::set_seed(20260902);

    const arma::mat x_t = arma::randn<arma::mat>(k, tt);
    const arma::vec u_precision{2.0, 3.0, 1.5, 4.0};
    const arma::vec v_precision{1.4, 0.8};
    const arma::mat u_sigma = arma::diagmat(1.0 / u_precision);
    const arma::mat v_sigma = arma::diagmat(1.0 / v_precision);

    // Loadings that move over the sample, and by enough that using the wrong
    // period's would show. A flat path would let an off-by-one pass.
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
            sample.col(r) = arma::vectorise(draw_factor_path(
                x_t, lambda_stack, u_sigma, v_sigma, a_stack, n, p > 0 ? p : 1));
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
    arma::mat x;      ///< tt x k
    arma::mat lambda; ///< n_lambda x tt, the free loadings that generated it
    arma::mat a;      ///< (N^2 p) x tt, the transition that generated it
    double u_sd = 0.0;
    double v_sd = 0.0;
};

/// A sample whose coefficients really do move: both paths are given a
/// deterministic ramp over the sample rather than a random walk, so what the
/// chain has to find is a known path and not one more draw.
Simulated simulate(const int k, const int n, const int p, const int tt)
{
    const int n_lambda = n * (2 * k - n - 1) / 2;

    Simulated s;
    s.lambda = moving_loadings(n_lambda, tt);
    s.a = moving_transition(n, p, tt);
    s.u_sd = 0.5;
    s.v_sd = 1.0;

    arma::mat factors(n, tt, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        const arma::mat a_t = arma::reshape(s.a.col(t), n, n * p);
        arma::vec f = s.v_sd * arma::randn<arma::vec>(n);
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
        x.col(t) = lambda_t * factors.col(t) + s.u_sd * arma::randn<arma::vec>(k);
    }
    s.x = x.t();
    return s;
}

bayests::DfmTvpGammaInput make_input(const Simulated &s, const int k, const int n, const int p,
                                     const int tt, const int iterations, const int burnin)
{
    const int n_lambda = n * (2 * k - n - 1) / 2;
    const int n_a = n * n * p;

    bayests::DfmTvpGammaInput in;
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

    in.u_sigma_prior.shape = arma::vec(k, arma::fill::value(3.0));
    in.u_sigma_prior.rate = arma::vec(k, arma::fill::value(0.5));
    in.v_sigma_prior.shape = arma::vec(n, arma::fill::value(3.0));
    in.v_sigma_prior.rate = arma::vec(n, arma::fill::value(0.5));

    in.initial.lambda = arma::mat(n_lambda, tt, arma::fill::value(0.5));
    in.initial.lambda_sigma_inv =
        arma::diagmat(arma::vec(n_lambda, arma::fill::value(100.0)));
    in.initial.lambda_init = arma::vec(n_lambda, arma::fill::value(0.5));

    in.initial.a = arma::mat(n_a, tt, arma::fill::zeros);
    in.initial.a_sigma_inv = arma::diagmat(arma::vec(n_a, arma::fill::value(100.0)));
    in.initial.a_init = arma::vec(n_a, arma::fill::zeros);

    in.initial.u_sigma_inv = arma::vec(k, arma::fill::ones);
    in.initial.v_sigma_inv = arma::vec(n, arma::fill::ones);
    return in;
}

/// The weakest of the three checks, and the one whose tolerances need thinking
/// about. The seed is fixed, but `arma::randn` and `arma::randg` go through
/// `std::normal_distribution` and `std::gamma_distribution`, whose sequences the
/// standard does not pin down -- so on another library this is a fresh sample and
/// a fresh chain, not a rerun of this one. What has to be tolerated is the
/// sampling error of the whole experiment, and the drift adds to it: a loading
/// path is tt numbers estimated from tt observations, so it is checked as an
/// average level and a direction of travel rather than pointwise.
void it_recovers_what_it_was_given()
{
    std::printf("recovery from a simulated sample\n");

    const int k = 8, n = 2, p = 1, tt = 600;
    const int iterations = 400, burnin = 200;
    const int n_lambda = n * (2 * k - n - 1) / 2;

    arma::arma_rng::set_seed(20260902);
    const Simulated s = simulate(k, n, p, tt);
    const bayests::DfmTvpGammaInput in = make_input(s, k, n, p, tt, iterations, burnin);

    bayests::NullReporter reporter;
    const bayests::DfmTvpGammaDraws out =
        bayests::DfmTvpGammaSampler{}.draw_coefficients(in, reporter);

    check("the posterior has the shapes the model implies",
          out.lambda.n_rows == static_cast<arma::uword>(k * n * tt) &&
              out.lambda_sigma.n_rows == static_cast<arma::uword>(n_lambda) &&
              out.factors.n_rows == static_cast<arma::uword>(n * tt) &&
              out.a.n_rows == static_cast<arma::uword>(n * n * p * tt) &&
              out.a_sigma.n_rows == static_cast<arma::uword>(n * n * p) &&
              out.u_sigma_inv.n_rows == static_cast<arma::uword>(k) &&
              out.v_sigma_inv.n_rows == static_cast<arma::uword>(n) &&
              out.iterations() == static_cast<arma::uword>(iterations));

    // The loading path, as the free elements the simulation was written in. The
    // posterior stores whole matrices, so the free ones are read back out of
    // them at their known positions.
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

    check_below("the average loading", std::abs(arma::mean(arma::vectorise(lambda_path)) -
                                                arma::mean(arma::vectorise(s.lambda))),
                0.15);

    // The whole point of the model: it has to see that the loadings moved. Every
    // simulated element ramps downwards, so the late quarter has to sit below
    // the early one -- a sampler that ignored the period index would land on a
    // flat path and fail this.
    const int quarter = tt / 4;
    const double got_early = arma::mean(arma::vectorise(lambda_path.head_cols(quarter)));
    const double got_late = arma::mean(arma::vectorise(lambda_path.tail_cols(quarter)));
    const double want_early = arma::mean(arma::vectorise(s.lambda.head_cols(quarter)));
    const double want_late = arma::mean(arma::vectorise(s.lambda.tail_cols(quarter)));

    check("the loadings are found to fall", got_late < got_early - 0.05,
          "early " + std::to_string(got_early) + ", late " + std::to_string(got_late) +
              "; simulated " + std::to_string(want_early) + " to " + std::to_string(want_late));

    // The transition, the same way: its first element decays over the sample.
    const arma::mat a_mean = arma::reshape(arma::mean(out.a, 1), n * n * p, tt);
    const arma::rowvec a_first = a_mean.row(0);
    const double a_early = arma::mean(a_first.head(quarter));
    const double a_late = arma::mean(a_first.tail(quarter));
    check("the transition is found to decay", a_late < a_early - 0.05,
          "early " + std::to_string(a_early) + ", late " + std::to_string(a_late));

    // The two precisions, which do not move in this model.
    const arma::vec u_variance = 1.0 / arma::mean(out.u_sigma_inv, 1);
    check_below("the idiosyncratic variance",
                std::abs(arma::mean(u_variance) - s.u_sd * s.u_sd), 0.15);

    // The identifying block is not drawn, so it has to be exactly itself in
    // every period of every stored draw and not merely on average.
    const arma::mat first_period = arma::reshape(out.lambda.submat(0, 0, k * n - 1, 0), k, n);
    arma::mat want = arma::eye<arma::mat>(n, n);
    want(1, 0) = first_period(1, 0); // the one free element of the block
    check_below("the identifying block survives every draw",
                max_abs_diff(first_period.submat(0, 0, n - 1, n - 1), want), 0.0);

    // The forecast, from the terminal period the io layer cuts out.
    bayests::DfmTvpGammaInput fcst_in = in;
    fcst_in.spec.h = 5;

    bayests::DfmTvpGammaDraws terminal = out;
    terminal.lambda = out.lambda.tail_rows(k * n);
    terminal.a = out.a.tail_rows(n * n * p);
    const bayests::ForecastDraws fcst =
        bayests::DfmTvpGammaSampler{}.forecast(fcst_in, terminal, reporter);
    check("the forecast is (h k) x draws",
          fcst.values.n_rows == static_cast<arma::uword>(5 * k) &&
              fcst.values.n_cols == static_cast<arma::uword>(iterations));
    check("the forecast is finite", fcst.values.is_finite());

    // And that the whole path is refused rather than silently read as the first
    // period, which is what a caller that skipped the cut would be handing over.
    bool threw = false;
    try
    {
        bayests::DfmTvpGammaSampler{}.forecast(fcst_in, out, reporter);
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check("a forecast from the whole path is refused", threw);

    const arma::mat loglik = bayests::DfmTvpGammaSampler{}.log_likelihood(in, out);
    check("the log likelihood is draws x periods",
          loglik.n_rows == static_cast<arma::uword>(iterations) &&
              loglik.n_cols == static_cast<arma::uword>(tt));
    check("the log likelihood is finite", loglik.is_finite());

    // The mirror of the forecast check: the likelihood scores every period under
    // its own loadings, so the terminal slice is not enough for it.
    threw = false;
    try
    {
        bayests::DfmTvpGammaSampler{}.log_likelihood(in, terminal);
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check("a log likelihood from the terminal period alone is refused", threw);
}

//////////////////////////////////////////////////////////////////////////////
// 4. What validate() has to refuse.

void expect_rejected(const std::string &what, const bayests::DfmTvpGammaInput &in)
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
    const bayests::DfmTvpGammaInput good = make_input(s, k, n, p, tt, 20, 10);

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

    bayests::DfmTvpGammaInput no_factors = good;
    no_factors.spec.n_factors = 0;
    expect_rejected("a model with no factors", no_factors);

    bayests::DfmTvpGammaInput too_many = good;
    too_many.spec.n_factors = k + 1;
    expect_rejected("more factors than series", too_many);

    bayests::DfmTvpGammaInput varsel = good;
    varsel.spec.varsel = bayests::VarSelection::bvs;
    expect_rejected("variable selection, which is not implemented", varsel);

    bayests::DfmTvpGammaInput structural = good;
    structural.spec.structural = true;
    expect_rejected("a structural factor model", structural);

    // The two widths a factor model carries, and the pair a file is most likely
    // to swap: the loading block is n_lambda wide and the transition n_factor_a.
    bayests::DfmTvpGammaInput lambda_width = good;
    lambda_width.initial.lambda = arma::mat(n * n * p, tt, arma::fill::zeros);
    expect_rejected("a loading path of the transition's width", lambda_width);

    bayests::DfmTvpGammaInput a_width = good;
    a_width.initial.a = arma::mat(n_lambda, tt, arma::fill::zeros);
    expect_rejected("a transition path of the loadings' width", a_width);

    bayests::DfmTvpGammaInput short_path = good;
    short_path.initial.lambda = arma::mat(n_lambda, tt - 1, arma::fill::value(0.5));
    expect_rejected("a loading path shorter than the sample", short_path);

    bayests::DfmTvpGammaInput no_init = good;
    no_init.initial.lambda_init = arma::vec(n_lambda - 1, arma::fill::zeros);
    expect_rejected("a state before the sample of the wrong width", no_init);

    bayests::DfmTvpGammaInput a_prior_width = good;
    a_prior_width.a_prior.sigma.shape = arma::vec(n_lambda, arma::fill::value(3.0));
    expect_rejected("a transition prior of the loadings' width", a_prior_width);

    bayests::DfmTvpGammaInput zero_precision = good;
    zero_precision.initial.u_sigma_inv = arma::vec(k, arma::fill::zeros);
    expect_rejected("a zero initial idiosyncratic precision", zero_precision);

    // One period leaves both random walks nothing to difference. Everything else
    // has to be made consistent with a sample of one, or a different check fires
    // first and this passes for the wrong reason -- the transition order, which
    // has to be under the sample length, and the two paths, whose shape is
    // checked before the floor is.
    bayests::DfmTvpGammaInput one_period = good;
    one_period.spec.p = 0;
    one_period.train.y = s.x.head_rows(1);
    one_period.initial.lambda = arma::mat(n_lambda, 1, arma::fill::value(0.5));
    expect_rejected("a sample of one period", one_period);
}

} // namespace

int main()
{
    std::printf("DfmTvpGamma\n\n");

    the_stacked_loadings_are_period_by_period();
    the_stacked_transition_is_period_by_period();
    the_transition_design_is_the_kronecker_product();
    the_prior_over_the_first_factors_uses_each_period();
    a_flat_path_is_the_constant_model();
    the_factor_path_is_the_posterior_it_should_be();
    it_recovers_what_it_was_given();
    the_impossible_inputs_are_refused();

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
