// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_chan_jeliazkov.cpp
/// @brief Checks the precision based state path draw, including against the
///        simulation smoother it is an alternative to.
///
/// The two algorithms target the same conditional posterior by different routes,
/// so the strongest check available is that they agree on it: the same inputs
/// through both, many draws each, and the two sample means and standard
/// deviations have to match to sampling error. That is what says this is the
/// same distribution and not merely a plausible one, and no amount of internal
/// consistency checking substitutes for it.
///
/// Around that, three exact statements about which period each block belongs to
/// -- the part of a banded sampler with no redundancy in it, and where an
/// off-by-one produces a result that looks entirely reasonable:
///
///   * a constant argument and a stack of T copies of it are the same model, so
///     they must give the same draw from the same seed;
///   * with the state innovation variance negligible except in one period, the
///     path can only move in that period, which pins the transition blocks;
///   * with the measurement precise and the regressors the identity, each state
///     is pinned to its own observation, which pins the measurement blocks.
///
/// `sigma_v` cannot be set to exactly zero the way `unit_kalman.cpp` does, since
/// a precision based sampler inverts it. Negligible-but-invertible gives the
/// same statement with a tolerance instead of an equality.

#include "core/algorithms/chan_jeliazkov_2009.h"
#include "core/algorithms/kalman_durbin_koopman_2002.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace
{

int failures = 0;

void check(const char *what, const bool ok)
{
    std::printf("  %-54s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_below(const char *what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-54s %s   %.3e < %.3e\n", what, ok ? "ok" : "FAILED", got, bound);
    if (!ok)
    {
        failures++;
    }
}

arma::mat stacked(const arma::mat &block, const arma::uword t)
{
    arma::mat out(block.n_rows * t, block.n_cols);
    for (arma::uword i = 0; i < t; i++)
    {
        out.rows(i * block.n_rows, (i + 1) * block.n_rows - 1) = block;
    }
    return out;
}

struct Model
{
    arma::mat y, z, sigma_u, sigma_v, B, P_init;
    arma::vec a_init;
};

Model make_model(const arma::uword k, const arma::uword m, const arma::uword t)
{
    arma::arma_rng::set_seed(20260824);

    Model d;
    d.y = arma::randn<arma::mat>(k, t);
    d.z = arma::randn<arma::mat>(k * t, m);
    d.sigma_u = arma::eye<arma::mat>(k, k) + 0.3;
    d.sigma_v = arma::diagmat(arma::linspace<arma::vec>(0.02, 0.08, m));
    d.B = arma::eye<arma::mat>(m, m);
    d.a_init = arma::zeros<arma::vec>(m);
    d.P_init = arma::eye<arma::mat>(m, m);
    return d;
}

/// One matrix for every period, and a stack of copies of it, are the same model.
void the_two_argument_forms_agree()
{
    std::printf("a constant argument and its replication give the same draw\n");

    for (const arma::uword t : {arma::uword(1), arma::uword(2), arma::uword(30)})
    {
        const Model d = make_model(3, 5, t);

        arma::arma_rng::set_seed(7);
        const arma::mat constant =
            chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init, d.P_init);
        arma::arma_rng::set_seed(7);
        const arma::mat stack =
            chan_jeliazkov_2009(d.y, d.z, stacked(d.sigma_u, t), stacked(d.sigma_v, t),
                                stacked(d.B, t), d.a_init, d.P_init);

        char label[80];
        std::snprintf(label, sizeof(label), "T = %llu, all three arguments",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(constant, stack, "absdiff", 0.0));

        arma::arma_rng::set_seed(7);
        const arma::mat only_v = chan_jeliazkov_2009(d.y, d.z, d.sigma_u, stacked(d.sigma_v, t),
                                                     d.B, d.a_init, d.P_init);
        arma::arma_rng::set_seed(7);
        const arma::mat only_b = chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v,
                                                     stacked(d.B, t), d.a_init, d.P_init);
        std::snprintf(label, sizeof(label), "T = %llu, 'sigma_v' alone",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(constant, only_v, "absdiff", 0.0));
        std::snprintf(label, sizeof(label), "T = %llu, 'B' alone",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(constant, only_b, "absdiff", 0.0));

        // And 'z', which is the fourth of them: one measurement matrix that
        // holds for every period is what a dynamic factor model has -- its
        // loading matrix -- and is the case where Z'U^-1 Z is assembled once
        // instead of T times. Bit-identical, not merely close: the constant path
        // computes the same product the loop would have, once.
        const arma::mat z_block = d.z.head_rows(3);
        arma::arma_rng::set_seed(7);
        const arma::mat z_const = chan_jeliazkov_2009(d.y, z_block, d.sigma_u, d.sigma_v, d.B,
                                                      d.a_init, d.P_init);
        arma::arma_rng::set_seed(7);
        const arma::mat z_stack = chan_jeliazkov_2009(d.y, stacked(z_block, t), d.sigma_u,
                                                      d.sigma_v, d.B, d.a_init, d.P_init);
        std::snprintf(label, sizeof(label), "T = %llu, 'z' alone",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(z_const, z_stack, "absdiff", 0.0));

        // The same with a time varying error covariance, which takes the
        // constant 'z' down the per-period branch and so exercises the other
        // half of the split.
        arma::arma_rng::set_seed(7);
        const arma::mat z_const_uv = chan_jeliazkov_2009(d.y, z_block, stacked(d.sigma_u, t),
                                                         d.sigma_v, d.B, d.a_init, d.P_init);
        arma::arma_rng::set_seed(7);
        const arma::mat z_stack_uv = chan_jeliazkov_2009(d.y, stacked(z_block, t),
                                                         stacked(d.sigma_u, t), d.sigma_v, d.B,
                                                         d.a_init, d.P_init);
        std::snprintf(label, sizeof(label), "T = %llu, constant 'z' with a stacked 'sigma_u'",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(z_const_uv, z_stack_uv, "absdiff", 0.0));
    }
}

/// A state that can only move in one period has to move in that one.
void transition_blocks_land_in_the_right_period()
{
    std::printf("a state variance negligible but for one period gives one jump\n");

    const arma::uword k = 2, m = 3, t = 10;
    const Model d = make_model(k, m, t);

    for (const arma::uword jump : {arma::uword(0), arma::uword(4), arma::uword(t - 1)})
    {
        arma::mat sigma_v = stacked(1e-12 * arma::eye<arma::mat>(m, m), t);
        sigma_v.rows(jump * m, (jump + 1) * m - 1) = arma::eye<arma::mat>(m, m);

        arma::arma_rng::set_seed(3);
        const arma::mat a = chan_jeliazkov_2009(d.y, d.z, d.sigma_u, sigma_v, d.B, d.a_init,
                                                1e-12 * arma::eye<arma::mat>(m, m));

        double elsewhere = 0.0, at = 0.0;
        for (arma::uword i = 0; i < t; i++)
        {
            const double step = arma::abs(a.col(i + 1) - a.col(i)).max();
            if (i == jump) { at = step; }
            else { elsewhere = std::max(elsewhere, step); }
        }

        char label[80];
        std::snprintf(label, sizeof(label), "jump at %llu: flat everywhere else",
                      static_cast<unsigned long long>(jump));
        check_below(label, elsewhere, 1e-4);
        std::snprintf(label, sizeof(label), "jump at %llu: and it does move there",
                      static_cast<unsigned long long>(jump));
        check(label, at > 1e-3);
    }
}

/// Each state is pinned to its own observation when the measurement is precise.
void measurement_blocks_land_in_the_right_period()
{
    std::printf("a precise measurement pins each state to its own observation\n");

    const arma::uword m = 3, t = 8;
    arma::arma_rng::set_seed(5);
    const arma::mat y = arma::randn<arma::mat>(m, t);
    const arma::mat z = stacked(arma::eye<arma::mat>(m, m), t); // Z_i = I, so s_i = y_i

    arma::arma_rng::set_seed(2);
    const arma::mat a = chan_jeliazkov_2009(y, z, 1e-10 * arma::eye<arma::mat>(m, m),
                                            arma::eye<arma::mat>(m, m),
                                            arma::eye<arma::mat>(m, m),
                                            arma::zeros<arma::vec>(m),
                                            arma::eye<arma::mat>(m, m));

    // Columns 0 to T-1 each have an observation; the last has none, so it is
    // left to the transition and is deliberately not checked.
    check_below("every observed state sits on its observation",
                arma::abs(a.cols(0, t - 1) - y).max(), 1e-4);
}

/// The two algorithms are the same distribution, reached two ways.
void it_agrees_with_the_simulation_smoother()
{
    std::printf("agreement with kalman_durbin_koopman_2002 over 20000 draws\n");

    const arma::uword k = 2, m = 3, t = 5;
    const Model d = make_model(k, m, t);
    const int draws = 20000;

    arma::mat sum_cj(m, t + 1, arma::fill::zeros), sumsq_cj(m, t + 1, arma::fill::zeros);
    arma::mat sum_dk(m, t + 1, arma::fill::zeros), sumsq_dk(m, t + 1, arma::fill::zeros);

    arma::arma_rng::set_seed(41);
    for (int i = 0; i < draws; i++)
    {
        const arma::mat a =
            chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init, d.P_init);
        sum_cj += a;
        sumsq_cj += a % a;
    }
    arma::arma_rng::set_seed(41);
    for (int i = 0; i < draws; i++)
    {
        const arma::mat a = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u, d.sigma_v, d.B,
                                                       d.a_init, d.P_init);
        sum_dk += a;
        sumsq_dk += a % a;
    }

    const arma::mat mean_cj = sum_cj / draws, mean_dk = sum_dk / draws;
    const arma::mat sd_cj = arma::sqrt(sumsq_cj / draws - mean_cj % mean_cj);
    const arma::mat sd_dk = arma::sqrt(sumsq_dk / draws - mean_dk % mean_dk);

    // Four standard errors of the difference of two independent means, taken at
    // the widest marginal so that one bound covers every element.
    const double se = std::sqrt(2.0) * sd_dk.max() / std::sqrt(double(draws));
    check_below("posterior means agree", arma::abs(mean_cj - mean_dk).max(), 4.0 * se);
    check_below("posterior standard deviations agree",
                arma::abs(sd_cj - sd_dk).max() / sd_dk.max(), 0.05);
    std::printf("      widest marginal sd %.4f, 4 se = %.4f\n", sd_dk.max(), 4.0 * se);
}

/// A transition of order p, with the higher lags zero and the prior they imply,
/// is the first order model -- so it has to give the first order model's draw.
///
/// This is the sharpest check available on the banded machinery, because it puts
/// the whole of it against a path that is already verified against the simulation
/// smoother. Set `A_2 = 0` and give the joint prior on `(s_0, s_1)` that the
/// first order model actually implies for its first two states,
///
///     mean  (a_0, A_1 a_0),
///     var   [[P_0, P_0 A_1'], [A_1 P_0, A_1 P_0 A_1' + Sigma_v]],
///
/// and the two precisions are the same matrix, so the Cholesky factor is the same
/// matrix, the right hand side is the same vector, and the backward pass consumes
/// the random numbers in the same order. The draws then agree to rounding -- not
/// exactly, since the arithmetic reaches them by different routes, which is
/// precisely what makes the agreement informative.
void order_p_contains_order_one()
{
    std::printf("a zeroed higher lag reproduces the first order draw\n");

    const arma::uword k = 2, m = 3, t = 25;
    const Model d = make_model(k, m, t);
    const arma::mat a1 = 0.6 * arma::eye<arma::mat>(m, m) + 0.05;

    arma::arma_rng::set_seed(13);
    const arma::mat first_order =
        chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v, a1, d.a_init, d.P_init);

    // The same model written as order two.
    arma::mat b2(m, 2 * m, arma::fill::zeros);
    b2.cols(0, m - 1) = a1; // A_2 stays zero

    arma::vec a2(2 * m);
    a2.head(m) = d.a_init;
    a2.tail(m) = a1 * d.a_init;

    arma::mat p2(2 * m, 2 * m);
    p2.submat(0, 0, m - 1, m - 1) = d.P_init;
    p2.submat(0, m, m - 1, 2 * m - 1) = d.P_init * a1.t();
    p2.submat(m, 0, 2 * m - 1, m - 1) = a1 * d.P_init;
    p2.submat(m, m, 2 * m - 1, 2 * m - 1) = a1 * d.P_init * a1.t() + d.sigma_v;

    arma::arma_rng::set_seed(13);
    const arma::mat order_two = chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v, b2, a2, p2);

    check_below("order two with a zero second lag matches order one",
                arma::abs(order_two - first_order).max() / arma::abs(first_order).max(), 1e-10);
}

/// The lags are applied to the right periods, in the right order.
///
/// With the measurement uninformative and both variances negligible, the
/// posterior is the prior, and the prior is a deterministic recursion: the first
/// p states are `a_init` and every later one is `sum_j A_j s_{t-j}`. That is an
/// expected value this test computes for itself, and it pins three things the
/// agreement tests cannot -- that `A_1` is the first block of columns of `B` and
/// not the last, that `a_init` runs forward in time and not backward, and that
/// the transition indexed `t - 1` produces column `t`.
void the_lags_land_on_the_right_periods()
{
    std::printf("the order p recursion reproduces itself\n");

    const arma::uword k = 2, m = 2, t = 15, p = 3;

    arma::arma_rng::set_seed(8);
    const arma::mat y = arma::randn<arma::mat>(k, t);
    const arma::mat z = arma::randn<arma::mat>(k * t, m);

    // Distinct, decaying lag coefficients, so a wrong order is visible and the
    // recursion stays bounded over the sample.
    arma::mat b(m, p * m, arma::fill::zeros);
    b.cols(0, m - 1) = 0.4 * arma::eye<arma::mat>(m, m);
    b.cols(m, 2 * m - 1) = 0.2 * arma::eye<arma::mat>(m, m);
    b.cols(2 * m, 3 * m - 1) = 0.1 * arma::eye<arma::mat>(m, m);

    arma::vec a_init(p * m);
    for (arma::uword i = 0; i < p * m; i++) { a_init(i) = 1.0 + double(i); }

    arma::arma_rng::set_seed(4);
    const arma::mat draw = chan_jeliazkov_2009(y, z, 1e10 * arma::eye<arma::mat>(k, k),
                                               1e-10 * arma::eye<arma::mat>(m, m), b, a_init,
                                               1e-10 * arma::eye<arma::mat>(p * m, p * m));

    // What the prior says the path is.
    arma::mat want(m, t + 1, arma::fill::zeros);
    for (arma::uword i = 0; i < p; i++) { want.col(i) = a_init.subvec(i * m, (i + 1) * m - 1); }
    for (arma::uword i = p; i <= t; i++)
    {
        for (arma::uword j = 1; j <= p; j++)
        {
            want.col(i) += b.cols((j - 1) * m, j * m - 1) * want.col(i - j);
        }
    }

    check_below("the drawn path is the prior recursion",
                arma::abs(draw - want).max(), 1e-3);
    check("and it is not trivially zero", arma::abs(want).max() > 0.5);
}

/// Shape, finiteness, the seed, and the one thing it does not accept.
void the_draw_is_well_formed()
{
    std::printf("shape, finiteness, reproducibility and rejected input\n");

    const arma::uword k = 3, m = 6, t = 20;
    const Model d = make_model(k, m, t);

    arma::arma_rng::set_seed(1);
    const arma::mat first =
        chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init, d.P_init);
    arma::arma_rng::set_seed(1);
    const arma::mat second =
        chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init, d.P_init);

    check("the result is M x (T + 1)", first.n_rows == m && first.n_cols == t + 1);
    check("the result is finite", first.is_finite());
    check("the same seed gives the same draw", arma::approx_equal(first, second, "absdiff", 0.0));

    // A singular P_init is the documented limitation: the precision form needs
    // its inverse. It has to say so rather than return nonsense.
    bool rejected = false;
    try
    {
        chan_jeliazkov_2009(d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init,
                            arma::zeros<arma::mat>(m, m));
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    check("a singular 'P_init' is rejected by name", rejected);

    // K rows and KT rows are both accepted -- one measurement matrix for every
    // period, or one per period -- so a height that is neither has to be K + 1
    // rather than K to test anything.
    bool bad_dims = false;
    try
    {
        chan_jeliazkov_2009(d.y, d.z.head_rows(k + 1), d.sigma_u, d.sigma_v, d.B, d.a_init,
                            d.P_init);
    }
    catch (const std::invalid_argument &)
    {
        bad_dims = true;
    }
    check("a 'z' of the wrong height is rejected", bad_dims);
}

/// The dense precision and right hand side of the whole path, over `n` state
/// columns.
///
/// Built as products of stacked matrices -- a selection matrix for the prior, a
/// block diagonal design for the measurements, a difference operator for the
/// transitions -- rather than by scattering blocks into a band. That is the
/// point of it: the same object reached by a route that shares no index
/// arithmetic with the implementation, so an off-by-one in either one shows up
/// as a disagreement rather than being reproduced on both sides.
struct Dense
{
    arma::mat k;
    arma::vec b;
};

Dense dense_system(const Model &d, const arma::uword m, const arma::uword tt,
                   const arma::uword p, const arma::uword n)
{
    const arma::uword kk = d.y.n_rows;
    const arma::uword side = m * n;

    Dense out;
    out.k = arma::zeros<arma::mat>(side, side);
    out.b = arma::zeros<arma::vec>(side);

    // The prior, over the first p columns jointly.
    arma::mat sp(p * m, side, arma::fill::zeros);
    sp.cols(0, p * m - 1) = arma::eye<arma::mat>(p * m, p * m);
    const arma::mat p_prec = arma::inv_sympd(d.P_init);
    out.k += sp.t() * p_prec * sp;
    out.b += sp.t() * p_prec * d.a_init;

    // The measurements, one per period, against the whole state.
    arma::mat zbig(kk * tt, side, arma::fill::zeros);
    arma::mat ubig(kk * tt, kk * tt, arma::fill::zeros);
    for (arma::uword t = 0; t < tt; t++)
    {
        zbig.submat(t * kk, t * m, (t + 1) * kk - 1, (t + 1) * m - 1) = d.z;
        ubig.submat(t * kk, t * kk, (t + 1) * kk - 1, (t + 1) * kk - 1) = d.sigma_u;
    }
    const arma::mat u_prec = arma::inv_sympd(ubig);
    out.k += zbig.t() * u_prec * zbig;
    out.b += zbig.t() * u_prec * arma::vectorise(d.y);

    // The transitions producing columns p ... n - 1. Their residual has mean
    // zero, so they reach the precision and not the right hand side.
    const arma::uword rows = n - p;
    arma::mat hbig(rows * m, side, arma::fill::zeros);
    arma::mat vbig(rows * m, rows * m, arma::fill::zeros);
    for (arma::uword t = p; t < n; t++)
    {
        const arma::uword r = (t - p) * m;
        hbig.submat(r, t * m, r + m - 1, (t + 1) * m - 1) = arma::eye<arma::mat>(m, m);
        for (arma::uword j = 1; j <= p; j++)
        {
            hbig.submat(r, (t - j) * m, r + m - 1, (t - j + 1) * m - 1) =
                -d.B.cols((j - 1) * m, j * m - 1);
        }
        vbig.submat(r, r, r + m - 1, r + m - 1) = d.sigma_v;
    }
    out.k += hbig.t() * arma::inv_sympd(vbig) * hbig;

    return out;
}

/// A model with a transition of order two whose coefficients are not symmetric
/// and do not commute.
///
/// Both matter for what the conditional draw is checked on below. A symmetric
/// coupling would make the block above the diagonal equal to the one below it,
/// and the cross term this could most plausibly get wrong -- the one that
/// reaches column i + d from the data of column i, through the transpose of the
/// only block that is stored -- would then be indistinguishable from the term
/// beside it.
Model make_asymmetric_model(const arma::uword k, const arma::uword m, const arma::uword t)
{
    arma::arma_rng::set_seed(20260903);

    Model d;
    d.y = arma::randn<arma::mat>(k, t);
    d.z = arma::randn<arma::mat>(k, m);
    d.sigma_u = arma::eye<arma::mat>(k, k) + 0.2;
    d.sigma_v = arma::diagmat(arma::linspace<arma::vec>(0.05, 0.15, m));

    arma::mat a1(m, m, arma::fill::zeros), a2(m, m, arma::fill::zeros);
    a1.diag().fill(0.5);
    a1(0, 1) = 0.3;
    a1(2, 0) = -0.25;
    a2.diag().fill(0.1);
    a2(1, 0) = 0.2;

    d.B = arma::join_horiz(a1, a2);
    d.a_init = arma::linspace<arma::vec>(0.1, 0.4, 2 * m);
    d.P_init = arma::eye<arma::mat>(2 * m, 2 * m) * 0.5 + 0.1;
    return d;
}

/// The state column past the end of the sample carries no information about the
/// ones before it, so leaving it out is exact rather than nearly so.
///
/// This is what lets the conditional entry point decline to build it -- there it
/// would be a column half data and half not. Stated as linear algebra and
/// checked as linear algebra: marginalising the last block out of the dense
/// system that has it must give the dense system that never had it, precision
/// and right hand side both.
void the_trailing_column_marginalises_away()
{
    std::printf("the state column past the sample carries no information\n");

    const arma::uword k = 4, m = 3, tt = 7, p = 2;
    const Model d = make_asymmetric_model(k, m, tt);

    const Dense with_it = dense_system(d, m, tt, p, tt + 1);
    const Dense without = dense_system(d, m, tt, p, tt);

    const arma::uword keep = m * tt;
    const arma::uword all = m * (tt + 1);
    const arma::mat kaa = with_it.k.submat(0, 0, keep - 1, keep - 1);
    const arma::mat kab = with_it.k.submat(0, keep, keep - 1, all - 1);
    const arma::mat kbb = with_it.k.submat(keep, keep, all - 1, all - 1);

    const arma::mat marginal = kaa - kab * arma::solve(kbb, kab.t());
    const arma::vec marginal_b =
        with_it.b.head(keep) - kab * arma::solve(kbb, with_it.b.tail(all - keep));

    check_below("the marginal precision is the truncated one",
                arma::abs(marginal - without.k).max(), 1e-10);
    check_below("the marginal right hand side is the truncated one",
                arma::abs(marginal_b - without.b).max(), 1e-10);
    check("and the trailing column was not trivially uncoupled", arma::abs(kab).max() > 0.1);
}

/// The conditional draw against the conditional distribution, worked out from
/// the dense precision.
///
/// Conditioning a Gaussian on part of itself is K_FF f = b_F - K_FY y, so the
/// distribution the sampler has to produce is known in closed form here: its
/// mean solves that system and its covariance is K_FF inverse. Twenty thousand
/// draws then have to reproduce both, which checks the sub-block extraction and
/// *both* right hand side corrections at once -- dropping either one moves the
/// mean while leaving the standard deviations exactly right, which is the shape
/// this failure takes.
void it_matches_the_dense_conditional()
{
    std::printf("agreement with the dense conditional over 20000 draws\n");

    const arma::uword k = 4, m = 3, tt = 7, p = 2;
    const arma::uword n_free = 2, n_known = m - n_free;
    const Model d = make_asymmetric_model(k, m, tt);

    arma::arma_rng::set_seed(99);
    const arma::mat known = arma::randn<arma::mat>(n_known, tt) + 0.5;

    // The free and the known positions of the stacked path, in its own order:
    // column by column, and within a column the free elements first.
    arma::uvec free_idx(n_free * tt), known_idx(n_known * tt);
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < n_free; i++)
        {
            free_idx(t * n_free + i) = t * m + i;
        }
        for (arma::uword i = 0; i < n_known; i++)
        {
            known_idx(t * n_known + i) = t * m + n_free + i;
        }
    }

    const Dense dense = dense_system(d, m, tt, p, tt);
    const arma::mat kff = dense.k.submat(free_idx, free_idx);
    const arma::mat kfy = dense.k.submat(free_idx, known_idx);

    const arma::vec want_mean =
        arma::solve(kff, dense.b.elem(free_idx) - kfy * arma::vectorise(known));
    const arma::vec want_sd = arma::sqrt(arma::mat(arma::inv_sympd(kff)).diag());

    const int draws = 20000;
    arma::vec sum(n_free * tt, arma::fill::zeros), sumsq(n_free * tt, arma::fill::zeros);

    arma::arma_rng::set_seed(7);
    for (int i = 0; i < draws; i++)
    {
        const arma::mat path = chan_jeliazkov_2009_conditional(
            d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init, d.P_init, known);
        const arma::vec flat = arma::vectorise(path);
        sum += flat;
        sumsq += flat % flat;
    }

    const arma::vec got_mean = sum / draws;
    const arma::vec got_sd = arma::sqrt(sumsq / draws - got_mean % got_mean);

    const double se = want_sd.max() / std::sqrt(double(draws));
    check_below("posterior means agree", arma::abs(got_mean - want_mean).max(), 4.0 * se);
    check_below("posterior standard deviations agree",
                arma::abs(got_sd - want_sd).max() / want_sd.max(), 0.05);
    check("and the conditioning actually moved the mean", arma::abs(want_mean).max() > 0.05);
    std::printf("      widest marginal sd %.4f, 4 se = %.4f\n", want_sd.max(), 4.0 * se);
}

/// Conditioning on nothing is the whole path over the periods, and the shape and
/// the rejections are what the caller of a factor augmented VAR will hit first.
void the_conditional_draw_is_well_formed()
{
    std::printf("shape and rejections of the conditional draw\n");

    const arma::uword k = 4, m = 3, tt = 7;
    const Model d = make_asymmetric_model(k, m, tt);
    const arma::mat known = arma::randn<arma::mat>(1, tt);

    arma::arma_rng::set_seed(3);
    const arma::mat first = chan_jeliazkov_2009_conditional(d.y, d.z, d.sigma_u, d.sigma_v, d.B,
                                                            d.a_init, d.P_init, known);
    arma::arma_rng::set_seed(3);
    const arma::mat second = chan_jeliazkov_2009_conditional(d.y, d.z, d.sigma_u, d.sigma_v, d.B,
                                                             d.a_init, d.P_init, known);

    check("the result is (M - R) x T", first.n_rows == m - 1 && first.n_cols == tt);
    check("the result is finite", first.is_finite());
    check("the same seed gives the same draw", arma::approx_equal(first, second, "absdiff", 0.0));

    const arma::mat nothing_known(0, tt);
    const arma::mat whole = chan_jeliazkov_2009_conditional(d.y, d.z, d.sigma_u, d.sigma_v, d.B,
                                                            d.a_init, d.P_init, nothing_known);
    check("conditioning on nothing draws the whole state",
          whole.n_rows == m && whole.n_cols == tt && whole.is_finite());

    bool rejected = false;
    try
    {
        const arma::mat everything = arma::randn<arma::mat>(m, tt);
        chan_jeliazkov_2009_conditional(d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init, d.P_init,
                                        everything);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    check("a 'known' that leaves nothing to draw is rejected", rejected);

    bool wrong_periods = false;
    try
    {
        const arma::mat short_known = arma::randn<arma::mat>(1, tt - 1);
        chan_jeliazkov_2009_conditional(d.y, d.z, d.sigma_u, d.sigma_v, d.B, d.a_init, d.P_init,
                                        short_known);
    }
    catch (const std::invalid_argument &)
    {
        wrong_periods = true;
    }
    check("a 'known' with the wrong number of periods is rejected", wrong_periods);
}

/// Runs one group and reports a throw as a failure of that group rather than
/// letting it abort the harness. A wrong band can make the posterior precision
/// indefinite, so "it threw" is a result this test has to be able to print --
/// and to print alongside the groups that still ran.
void run_group(const char *name, void (*group)())
{
    try
    {
        group();
    }
    catch (const std::exception &e)
    {
        std::printf("  %-54s %s\n", name, "FAILED");
        std::printf("      threw: %s\n", e.what());
        failures++;
    }
}

} // namespace

int main()
{
    run_group("group: the two argument forms", the_two_argument_forms_agree);
    run_group("group: transition blocks", transition_blocks_land_in_the_right_period);
    run_group("group: measurement blocks", measurement_blocks_land_in_the_right_period);
    run_group("group: order p contains order one", order_p_contains_order_one);
    run_group("group: the lags land right", the_lags_land_on_the_right_periods);
    run_group("group: agreement with the smoother", it_agrees_with_the_simulation_smoother);
    run_group("group: well formed", the_draw_is_well_formed);
    run_group("group: the trailing column", the_trailing_column_marginalises_away);
    run_group("group: the dense conditional", it_matches_the_dense_conditional);
    run_group("group: conditional well formed", the_conditional_draw_is_well_formed);

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
