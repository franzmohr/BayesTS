// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_kalman.cpp
/// @brief Checks the simulation smoother against identities that hold whatever
///        the random number generator does.
///
/// Three of them, and between them they pin the part of the smoother with no
/// redundancy in it: which period each argument's block belongs to, and which
/// period each column of the result belongs to.
///
/// The first is that the two forms every covariance may arrive in have to agree.
/// One matrix that holds for all periods, and a stack of T copies of that same
/// matrix, describe the same model, so they must produce the same draw from the
/// same seed -- exactly, not nearly. The smoother reaches the constant form
/// through a stride of zero rather than by replicating it, and this is what says
/// the two paths through it are the same path.
///
/// The second is what happens when the state innovation variance is zero in
/// every period but one. The state cannot move except at that period, so the
/// drawn path has to be piecewise constant with its single jump in exactly the
/// right place. That is an exact statement about which period a block of
/// `sigma_v` governs, and it fails for an off-by-one in the striding that the
/// agreement test above would not notice -- a uniform argument has no period to
/// be wrong about.

#include "core/algorithms/kalman_durbin_koopman_2002.h"

#include <cmath>
#include <cstdio>

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

void check_equal(const char *what, const double got, const double want, const double tolerance)
{
    const bool ok = std::abs(got - want) <= tolerance;
    std::printf("  %-54s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
    {
        std::printf("      got %.17g, want %.17g\n", got, want);
        failures++;
    }
}

/// The T blocks of a constant matrix, stacked -- the other form the smoother
/// accepts for the same model.
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
    d.sigma_v = arma::diagmat(arma::linspace<arma::vec>(0.01, 0.06, m));
    d.B = arma::eye<arma::mat>(m, m);
    d.a_init = arma::zeros<arma::vec>(m);
    d.P_init = d.sigma_v;
    return d;
}

/// One matrix for every period, and a stack of copies of it, are the same model.
void the_two_argument_forms_agree()
{
    std::printf("a constant argument and its replication give the same draw\n");

    for (const arma::uword t : {arma::uword(2), arma::uword(5), arma::uword(40)})
    {
        const Model d = make_model(3, 6, t);

        arma::arma_rng::set_seed(7);
        const arma::mat constant = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u, d.sigma_v, d.B,
                                                             d.a_init, d.P_init);
        arma::arma_rng::set_seed(7);
        const arma::mat stack = kalman_durbin_koopman_2002(
            d.y, d.z, stacked(d.sigma_u, t), stacked(d.sigma_v, t), stacked(d.B, t), d.a_init,
            d.P_init);

        char label[80];
        std::snprintf(label, sizeof(label), "T = %llu, all three arguments",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(constant, stack, "absdiff", 0.0));

        // One at a time as well, so that a stride confused between two of them
        // cannot cancel out.
        arma::arma_rng::set_seed(7);
        const arma::mat only_u = kalman_durbin_koopman_2002(d.y, d.z, stacked(d.sigma_u, t),
                                                            d.sigma_v, d.B, d.a_init, d.P_init);
        arma::arma_rng::set_seed(7);
        const arma::mat only_v = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u,
                                                            stacked(d.sigma_v, t), d.B, d.a_init,
                                                            d.P_init);
        arma::arma_rng::set_seed(7);
        const arma::mat only_b = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u, d.sigma_v,
                                                            stacked(d.B, t), d.a_init, d.P_init);
        std::snprintf(label, sizeof(label), "T = %llu, 'sigma_u' alone",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(constant, only_u, "absdiff", 0.0));
        std::snprintf(label, sizeof(label), "T = %llu, 'sigma_v' alone",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(constant, only_v, "absdiff", 0.0));
        std::snprintf(label, sizeof(label), "T = %llu, 'B' alone",
                      static_cast<unsigned long long>(t));
        check(label, arma::approx_equal(constant, only_b, "absdiff", 0.0));
    }
}

/// A state that is allowed to move in one period only has to move in that one.
///
/// With `sigma_v` zero the transition is deterministic and, with `B` the
/// identity, the state cannot change: neither the antithetic path of step 1 nor
/// the smoothed increment of the backward pass has anything to add. Making one
/// period's block non-zero buys exactly one jump, between the columns that block
/// governs.
void time_variation_lands_in_the_right_period()
{
    std::printf("a state variance that is zero but for one period gives one jump\n");

    const arma::uword k = 2, m = 3, t = 12;
    const Model d = make_model(k, m, t);

    for (const arma::uword jump : {arma::uword(0), arma::uword(4), arma::uword(t - 1)})
    {
        arma::mat sigma_v = arma::zeros<arma::mat>(m * t, m);
        sigma_v.rows(jump * m, (jump + 1) * m - 1) = arma::eye<arma::mat>(m, m);

        arma::arma_rng::set_seed(3);
        const arma::mat a = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u, sigma_v, d.B, d.a_init,
                                                       arma::zeros<arma::mat>(m, m));

        // Column j governs the step from column j to column j + 1, so the path
        // is constant either side of the one it was given.
        double before = 0.0, after = 0.0, at = 0.0;
        for (arma::uword i = 0; i < t; i++)
        {
            const double step = arma::abs(a.col(i + 1) - a.col(i)).max();
            if (i < jump) { before = std::max(before, step); }
            else if (i > jump) { after = std::max(after, step); }
            else { at = step; }
        }

        char label[80];
        std::snprintf(label, sizeof(label), "jump at %llu: the path is flat before it",
                      static_cast<unsigned long long>(jump));
        check_equal(label, before, 0.0, 1e-12);
        std::snprintf(label, sizeof(label), "jump at %llu: the path is flat after it",
                      static_cast<unsigned long long>(jump));
        check_equal(label, after, 0.0, 1e-12);
        std::snprintf(label, sizeof(label), "jump at %llu: and it does move there",
                      static_cast<unsigned long long>(jump));
        check(label, at > 1e-8);
    }
}

/// Each state is pinned to its own observation when the measurement is precise.
///
/// This is the one the callers get wrong. With `Z_i = I` and a measurement
/// variance next to nothing, the state has no freedom left: column i has to be
/// y_i, and which column that is says where the T + 1 returned columns sit
/// against the T observations. Column 0 is the state the *first* observation
/// loads on -- already smoothed, not a period before the sample -- and column T
/// is the transition applied once past the end, informed by nothing. A caller
/// that keeps `.cols(1, T)` rather than `.cols(0, T - 1)` shifts its whole path
/// by a period, and every other check in this file passes while it does: they
/// compare one call against another, and a shift is in both.
void measurement_blocks_land_in_the_right_period()
{
    std::printf("a precise measurement pins each state to its own observation\n");

    const arma::uword m = 3, t = 8;
    arma::arma_rng::set_seed(5);
    const arma::mat y = arma::randn<arma::mat>(m, t);
    const arma::mat z = stacked(arma::eye<arma::mat>(m, m), t); // Z_i = I, so a_i = y_i

    arma::arma_rng::set_seed(2);
    const arma::mat a = kalman_durbin_koopman_2002(y, z, 1e-10 * arma::eye<arma::mat>(m, m),
                                                   arma::eye<arma::mat>(m, m),
                                                   arma::eye<arma::mat>(m, m),
                                                   arma::zeros<arma::vec>(m),
                                                   arma::eye<arma::mat>(m, m));

    check_equal("every observed state sits on its observation",
                arma::abs(a.cols(0, t - 1) - y).max(), 0.0, 1e-4);

    // And the shifted reading is not merely worse but wrong, so that this cannot
    // be satisfied by a path flat enough for either alignment to fit.
    check("the shifted reading does not fit", arma::abs(a.cols(1, t) - y).max() > 1e-2);
}

/// The shape of the result, that it is finite, and that the seed decides it.
void the_draw_is_well_formed()
{
    std::printf("shape, finiteness and reproducibility\n");

    const arma::uword k = 3, m = 6, t = 20;
    const Model d = make_model(k, m, t);

    arma::arma_rng::set_seed(1);
    const arma::mat first = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u, d.sigma_v, d.B,
                                                       d.a_init, d.P_init);
    arma::arma_rng::set_seed(1);
    const arma::mat second = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u, d.sigma_v, d.B,
                                                        d.a_init, d.P_init);

    check("the result is M x (T + 1)", first.n_rows == m && first.n_cols == t + 1);
    check("the result is finite", first.is_finite());
    check("the same seed gives the same draw", arma::approx_equal(first, second, "absdiff", 0.0));

    // A non-unit transition has to be used rather than ignored.
    arma::arma_rng::set_seed(1);
    const arma::mat damped = kalman_durbin_koopman_2002(d.y, d.z, d.sigma_u, d.sigma_v,
                                                        0.5 * d.B, d.a_init, d.P_init);
    check("a transition of 0.5 I is not the same draw as the identity",
          !arma::approx_equal(first, damped, "absdiff", 1e-10));
}

} // namespace

int main()
{
    the_two_argument_forms_agree();
    time_variation_lands_in_the_right_period();
    measurement_blocks_land_in_the_right_period();
    the_draw_is_well_formed();

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
