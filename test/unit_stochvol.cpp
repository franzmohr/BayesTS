// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_stochvol.cpp
/// @brief Checks both stochastic volatility draws against the limits and the
///        bounds that hold whatever the random number generator does.
///
/// A posterior draw has no exact identity of the kind unit_vec_to_var.cpp
/// asserts, so this test is built out of the next best thing: limits in which
/// the randomness stops mattering, and bounds wide enough that only a broken
/// sampler crosses them.
///
/// The limit does most of the work. Send the innovation variance of the random
/// walk to zero and the prior pins the whole path to its initial state, whatever
/// the data says -- so the draw has an expected value that holds to four decimal
/// places on any machine. That covers the part of the algorithm with no
/// redundancy in it: the assembly of the posterior precision from the two bands
/// of D'D, and the fact that the initial state enters the first period alone
/// because every row of D'D but the first sums to zero.
///
/// The bounds cover the mixture tables, which the limit does not reach. A draw
/// that tracks a known log-volatility path to well inside the noise of the
/// transformed observation, and that does not drift away from it, needs the
/// weights, the means and the variances to be the published ones and to line up
/// with each other. This is the check that a mixture of the wrong length fails:
/// a table padded with zeros divides by a zero variance, and the draw comes out
/// singular on the first call.
///
/// Both algorithms run the same battery. They differ only in how many normal
/// components approximate the same log chi-squared distribution, so anything
/// asserted of one holds of the other.

#include "core/algorithms/stochvol_ksc_1998.h"
#include "core/algorithms/stochvol_ocsn_2007.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace
{

int failures = 0;

void check(const char *what, const bool ok)
{
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_below(const char *what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAILED");
    std::printf("      %.4f, bound %.4f\n", got, bound);
    if (!ok)
    {
        failures++;
    }
}

void check_above(const char *what, const double got, const double bound)
{
    const bool ok = got > bound;
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAILED");
    std::printf("      %.4f, bound %.4f\n", got, bound);
    if (!ok)
    {
        failures++;
    }
}

using Draw = arma::mat (*)(const arma::mat &, const arma::mat &, const arma::vec &,
                           const arma::vec &, const arma::vec &);

struct Algorithm
{
    const char *name;
    Draw draw;
};

/// A valid argument has to produce a draw. Reported rather than left to
/// propagate, so that a broken mixture -- which throws on the first call -- is a
/// failing check with the rest of the battery still to come, and not an abort
/// that takes the harness down before it says which algorithm it was.
arma::mat draw_or_fail(const char *what, const Draw draw, const arma::mat &y, const arma::mat &h,
                       const arma::vec &sigma, const arma::vec &h_init, const arma::vec &constant)
{
    try
    {
        return draw(y, h, sigma, h_init, constant);
    }
    catch (const std::exception &e)
    {
        std::printf("  %-52s %s\n", what, "FAILED");
        std::printf("      threw on valid input: %s\n", e.what());
        failures++;
        // NaN rather than zero, so that whatever the caller goes on to assert
        // about the draw fails too instead of reporting ok on a stand-in.
        return arma::mat(h.n_rows, h.n_cols).fill(arma::datum::nan);
    }
}

/// The argument has to be rejected, and rejected as invalid input rather than by
/// running into an Armadillo bounds error or returning a matrix of NaNs.
template <typename F> void check_rejects(const char *what, F &&call)
{
    try
    {
        call();
    }
    catch (const std::invalid_argument &)
    {
        std::printf("  %-52s %s\n", what, "ok");
        return;
    }
    catch (const std::exception &e)
    {
        std::printf("  %-52s %s\n", what, "FAILED");
        std::printf("      threw the wrong type: %s\n", e.what());
        failures++;
        return;
    }

    std::printf("  %-52s %s\n", what, "FAILED");
    std::printf("      returned instead of throwing\n");
    failures++;
}

/// A log-volatility path and the errors it generated: h is a random walk of
/// innovation variance `sigma` started at `h0`, and y is drawn with that
/// volatility. Both are what the sampler is asked to recover, so the seed is
/// pinned -- a bound that holds for one path is not evidence about a sampler if
/// the path changes between runs.
struct Sample
{
    arma::mat h_true;
    arma::mat y;
};

Sample simulate(const arma::uword tt, const arma::uword k, const double sigma, const double h0)
{
    arma::arma_rng::set_seed(20260824);

    Sample s;
    s.h_true = h0 + arma::cumsum(std::sqrt(sigma) * arma::randn<arma::mat>(tt, k));
    s.y = arma::exp(0.5 * s.h_true) % arma::randn<arma::mat>(tt, k);
    return s;
}

arma::vec filled(const arma::uword k, const double value)
{
    arma::vec v(k);
    v.fill(value);
    return v;
}

/// Runs at the lengths that bracket what a sampler sees, including the shortest
/// one the algorithm accepts, and produces the same draw twice from the same
/// seed.
void runs_at_every_length(const Algorithm &alg)
{
    std::printf("%s: valid draws\n", alg.name);

    for (const arma::uword tt : {arma::uword(2), arma::uword(3), arma::uword(17),
                                 arma::uword(250), arma::uword(1000)})
    {
        const Sample s = simulate(tt, 3, 0.05, -1.0);
        arma::mat y = s.y;
        y(0, 0) = 0.0; // the offset in 'constant' is what keeps log(y^2) finite

        char label[64];
        std::snprintf(label, sizeof(label), "T = %llu, a zero observation, finite",
                      static_cast<unsigned long long>(tt));

        arma::arma_rng::set_seed(11);
        const arma::mat draw = draw_or_fail(label, alg.draw, y, s.h_true, filled(3, 0.05),
                                            filled(3, -1.0), filled(3, 1e-8));

        check(label, draw.is_finite() && draw.n_rows == tt && draw.n_cols == 3);
    }

    const Sample s = simulate(64, 2, 0.05, -1.0);

    arma::arma_rng::set_seed(11);
    const arma::mat first = draw_or_fail("first of a pair", alg.draw, s.y, s.h_true,
                                         filled(2, 0.05), filled(2, -1.0), filled(2, 1e-8));
    arma::arma_rng::set_seed(11);
    const arma::mat second = draw_or_fail("second of a pair", alg.draw, s.y, s.h_true,
                                          filled(2, 0.05), filled(2, -1.0), filled(2, 1e-8));
    check("the same seed gives the same draw", arma::approx_equal(first, second, "absdiff", 0.0));

    // The documentation promises that 'h' may be the matrix the result is
    // assigned to, which is how every sampler calls this.
    arma::arma_rng::set_seed(11);
    arma::mat aliased = s.h_true;
    aliased = draw_or_fail("aliased draw", alg.draw, s.y, aliased, filled(2, 0.05),
                           filled(2, -1.0), filled(2, 1e-8));
    check("drawing into the argument gives the same draw",
          arma::approx_equal(first, aliased, "absdiff", 0.0));
}

/// The limit that has an expected value: with no innovation variance left in the
/// random walk, the prior pins every period to the initial state.
///
/// This is where the two bands of D'D and the right hand side are checked. The
/// initial state reaches the first period only -- the interior rows of D'D sum
/// to zero, the first sums to one -- so an initial state applied to every period
/// instead, or to none, moves the whole path away from h_init and fails here by
/// a wide margin. The deviation left is of order sqrt(sigma).
void tight_prior_pins_the_path(const Algorithm &alg)
{
    std::printf("%s: the zero innovation variance limit\n", alg.name);

    const Sample s = simulate(200, 2, 0.30, 0.0); // data that disagrees with h_init
    const double h0 = -3.5;

    arma::arma_rng::set_seed(5);
    const arma::mat draw =
        draw_or_fail("the draw at a vanishing innovation variance", alg.draw, s.y,
                     arma::mat(200, 2, arma::fill::zeros), filled(2, 1e-10), filled(2, h0),
                     filled(2, 1e-8));

    check_below("every period sits at h_init", arma::abs(draw - h0).max(), 1e-3);
}

/// The bound that covers the mixture: a draw has to track the log-volatility
/// that generated the data, and has to do it far better than the transformed
/// observation it is smoothing.
///
/// log(y^2) is the log-volatility plus a log chi-squared error of variance
/// pi^2 / 2, so the raw transform sits about 2.2 away in root mean square. The
/// sampler is given the variance that generated the path and has 500 periods to
/// pool, so it should come in near a third of that. The mean of the deviation is
/// the other half of the check: the mixture approximates a distribution with a
/// mean of -1.2704, and a table that has not been centred on it, or that has
/// been padded, drifts the whole path.
void recovers_the_volatility(const Algorithm &alg)
{
    std::printf("%s: recovery of a known path\n", alg.name);

    const double sigma = 0.05;
    const Sample s = simulate(500, 4, sigma, -1.0);
    const arma::vec constant = filled(4, 1e-8);

    arma::arma_rng::set_seed(2);
    arma::mat draw = arma::mat(500, 4, arma::fill::zeros);
    arma::mat total = arma::mat(500, 4, arma::fill::zeros);

    const int burn_in = 50;
    const int sweeps = 250;
    const int failures_before = failures;
    for (int sweep = 0; sweep < sweeps; sweep++)
    {
        draw = draw_or_fail("a sweep", alg.draw, s.y, draw, filled(4, sigma), s.h_true.row(0).t(),
                            constant);
        if (failures > failures_before)
        {
            return; // one report is enough; every remaining sweep would repeat it
        }
        if (sweep >= burn_in)
        {
            total += draw;
        }
    }
    const arma::mat mean = total / (sweeps - burn_in);

    const arma::mat raw = arma::log(arma::square(s.y) + 1e-8);

    check_below("root mean square error against the true path",
                std::sqrt(arma::mean(arma::vectorise(arma::square(mean - s.h_true)))), 1.0);
    check_above("the transformed observation it smooths is worse",
                std::sqrt(arma::mean(arma::vectorise(arma::square(raw - s.h_true)))), 2.0);
    check_below("no drift away from the true path",
                std::abs(arma::mean(arma::vectorise(mean - s.h_true))), 0.3);
    check("the last draw is finite", draw.is_finite());
}

/// Every argument the documentation says is rejected, is. The point is not the
/// message but that the failure is an exception at all: `-DARMA_NO_DEBUG`, which
/// an embedded host sets, turns a short vector into a read past the end of it
/// and a zero variance into an infinity that only surfaces as a broken draw
/// several sweeps later.
void rejects_bad_input(const Algorithm &alg)
{
    std::printf("%s: rejected arguments\n", alg.name);

    const Sample s = simulate(20, 2, 0.05, -1.0);
    const arma::mat &y = s.y;
    const arma::mat &h = s.h_true;
    const arma::vec sigma = filled(2, 0.05);
    const arma::vec h_init = filled(2, -1.0);
    const arma::vec constant = filled(2, 1e-8);

    check_rejects("'h' of the wrong shape",
                  [&] { return alg.draw(y, arma::mat(20, 3), sigma, h_init, constant); });
    check_rejects("no columns", [&] {
        return alg.draw(arma::mat(20, 0), arma::mat(20, 0), arma::vec(), arma::vec(), arma::vec());
    });
    check_rejects("a single period", [&] {
        return alg.draw(y.head_rows(1), h.head_rows(1), sigma, h_init, constant);
    });

    arma::mat y_nan = y;
    y_nan(3, 1) = arma::datum::nan;
    check_rejects("'y' with a NaN", [&] { return alg.draw(y_nan, h, sigma, h_init, constant); });

    arma::mat h_inf = h;
    h_inf(3, 1) = arma::datum::inf;
    check_rejects("'h' with an infinity",
                  [&] { return alg.draw(y, h_inf, sigma, h_init, constant); });

    check_rejects("'sigma' of the wrong length",
                  [&] { return alg.draw(y, h, filled(3, 0.05), h_init, constant); });
    check_rejects("'h_init' of the wrong length",
                  [&] { return alg.draw(y, h, sigma, filled(1, -1.0), constant); });
    check_rejects("'constant' of the wrong length",
                  [&] { return alg.draw(y, h, sigma, h_init, filled(3, 1e-8)); });

    check_rejects("a zero in 'sigma'",
                  [&] { return alg.draw(y, h, filled(2, 0.0), h_init, constant); });
    check_rejects("a negative in 'sigma'",
                  [&] { return alg.draw(y, h, filled(2, -0.05), h_init, constant); });
    check_rejects("a NaN in 'sigma'", [&] {
        return alg.draw(y, h, filled(2, arma::datum::nan), h_init, constant);
    });
    check_rejects("a zero in 'constant'",
                  [&] { return alg.draw(y, h, sigma, h_init, filled(2, 0.0)); });
    check_rejects("an infinity in 'h_init'", [&] {
        return alg.draw(y, h, sigma, filled(2, arma::datum::inf), constant);
    });
}

} // namespace

int main()
{
    const Algorithm algorithms[] = {
        {"KSC 1998", &stochvol_ksc_1998},
        {"OCSN 2007", &stochvol_ocsn_2007},
    };

    for (const Algorithm &alg : algorithms)
    {
        runs_at_every_length(alg);
        tight_prior_pins_the_path(alg);
        recovers_the_volatility(alg);
        rejects_bad_input(alg);
    }

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
