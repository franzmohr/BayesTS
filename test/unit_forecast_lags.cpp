// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_forecast_lags.cpp
/// @brief Checks that a simulated forecast path enters the regressors at the
///        right lag.
///
/// core::update_forecast_lags() is shared by every sampler's forecast, so the
/// ordering it implements is load-bearing six times over -- and it went wrong in
/// all six for as long as they existed, because the two spellings agree unless
/// p >= 2 and h >= 3. Nothing else pins it: the golden harness only fails when a
/// fixture throws, and a forecast with its lags reversed does not throw, it just
/// returns the wrong path.
///
/// The check is a race between two computations of the same thing. One drives
/// z the way a sampler does -- fill the lag columns from the path so far,
/// multiply by the coefficients -- and the other applies the recursion
/// y_t = sum_j A_j y_{t-j} directly. Errors are left out of both, so the two must
/// agree to rounding.
///
/// Lag blocks run most recent first: column block j carries y_{t-j}. That is
/// verified against the recorded fixtures rather than assumed -- in
/// VarNormalWishart-13.h5, k = 3 and p = 2, the first training observation has
/// endogen[:,9] in block one and endogen[:,8] in block two.

#include "core/models/model_support.h"

#include <cmath>
#include <cstdio>
#include <vector>

using bayests::core::update_forecast_lags;

namespace
{

int failures = 0;

void check_close(const char *what, const double got, const double want)
{
    const bool ok = std::abs(got - want) < 1e-12;
    std::printf("  %-44s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
    {
        std::printf("      got %.17g, want %.17g\n", got, want);
        failures++;
    }
}

/// The regressor matrix a caller hands to a forecast: h*k rows by k*p columns,
/// row block i holding kron(x_i', I_k) for the lags of horizon i. Only the cells
/// that are actual observations are filled -- lag j of horizon i is a real
/// observation when j > i -- because the rest is what update_forecast_lags()
/// exists to write.
///
/// `history` is y_T, y_{T-1}, ... most recent first, so lag j of horizon i is
/// history[j - i - 1].
arma::mat build_forecast_regressors(const std::vector<arma::vec> &history, const int k,
                                    const int p, const int h)
{
    const arma::mat diag_k = arma::eye<arma::mat>(k, k);
    arma::mat z = arma::zeros<arma::mat>(h * k, k * k * p);

    for (int i = 0; i < h; i++)
    {
        for (int j = 1; j <= p; j++)
        {
            if (j > i)
            {
                z.submat(i * k, (j - 1) * k * k, (i + 1) * k - 1, j * k * k - 1) =
                    arma::kron(arma::trans(history[static_cast<std::size_t>(j - i - 1)]), diag_k);
            }
        }
    }

    return z;
}

/// Drives z the way a sampler's forecast loop does, and returns the path.
arma::vec simulate_through_regressors(const arma::vec &a, arma::mat z,
                                     const std::vector<arma::vec> &history, const int k,
                                     const int p, const int h)
{
    const arma::mat diag_k = arma::eye<arma::mat>(k, k);
    arma::mat fcst = arma::zeros<arma::mat>(h * k, 1);

    for (int i = 0; i < h; i++)
    {
        if (i > 0 && p > 0)
        {
            update_forecast_lags(z, fcst, 0, i, k, p, diag_k);
        }
        fcst.submat(i * k, 0, (i + 1) * k - 1, 0) = z.rows(i * k, (i + 1) * k - 1) * a;
    }

    (void)history;
    return fcst.col(0);
}

/// y_t = sum_{j=1..p} A_j y_{t-j}, straight from the coefficient blocks.
arma::vec simulate_directly(const std::vector<arma::mat> &A, std::vector<arma::vec> history,
                            const int k, const int h)
{
    arma::vec path(static_cast<arma::uword>(h * k));

    for (int i = 0; i < h; i++)
    {
        arma::vec next = arma::zeros<arma::vec>(k);
        for (std::size_t j = 0; j < A.size(); j++)
        {
            next += A[j] * history[j];
        }

        path.subvec(static_cast<arma::uword>(i * k), static_cast<arma::uword>((i + 1) * k - 1)) =
            next;

        // Shift: the value just produced becomes lag one.
        history.insert(history.begin(), next);
        history.pop_back();
    }

    return path;
}

void run_case(const char *label, const int k, const int p, const int h)
{
    std::printf("\n%s\n", label);

    std::vector<arma::mat> A;
    arma::vec a;
    for (int j = 0; j < p; j++)
    {
        // Scaled down so a long horizon does not run off to infinity and drown
        // the comparison in magnitude.
        A.push_back(arma::randn<arma::mat>(k, k) * (0.4 / static_cast<double>(p)));
        a = arma::join_cols(a, arma::vectorise(A.back()));
    }

    // Most recent first, and one entry per lag the model has.
    std::vector<arma::vec> history;
    for (int j = 0; j < p; j++)
    {
        history.push_back(arma::randn<arma::vec>(k));
    }

    const arma::mat z = build_forecast_regressors(history, k, p, h);
    const arma::vec through_z = simulate_through_regressors(a, z, history, k, p, h);
    const arma::vec direct = simulate_directly(A, history, k, h);

    for (int i = 0; i < h * k; i++)
    {
        // Reported per horizon rather than per element: which horizon first
        // disagrees is what says whether the lags are reversed or merely stale.
        if (std::abs(through_z(i) - direct(i)) >= 1e-12)
        {
            std::printf("  horizon %d, element %d\n", i / k + 1, i % k);
        }
        check_close("path agrees", through_z(i), direct(i));
    }
}

} // namespace

int main()
{
    arma::arma_rng::set_seed(20260815);

    // p = 1 cannot get the order wrong: a single block has no order. It is here
    // so that a change breaking the easy case is not mistaken for the hard one.
    run_case("k=2, p=1, h=4", 2, 1, 4);

    // h <= p never reaches the branch where every lag is a forecast.
    run_case("k=2, p=3, h=2", 2, 3, 2);

    // The cases that matter: enough horizons that the path fills every lag, and
    // enough lags that reversing them is visible.
    run_case("k=2, p=2, h=4", 2, 2, 4);
    run_case("k=3, p=2, h=5", 3, 2, 5);
    run_case("k=2, p=4, h=8", 2, 4, 8);
    run_case("k=1, p=3, h=6", 1, 3, 6);

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
