// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_COMPLETION_SUPPORT_H
#define BAYESTS_CORE_MODELS_COMPLETION_SUPPORT_H

#include "bayests/data.h"
#include "bayests/spec.h"
#include "core/algorithms/constrained_var_path.h"

#include <stdexcept>
#include <string>

namespace bayests::core
{

// What a VAR sampler needs to run the data-completion step in its sweep. The
// step itself is draw_constrained_var_path(); these are the pieces on either
// side of it: the regressors a completed panel implies, and the prior over the
// path a draw of the parameters implies.
//
// The lag block is the first k p columns of a regressor row, lag j at columns
// (j - 1) k ... j k - 1, the layout update_forecast_lags() writes. The rest of
// the row -- exogenous and deterministic terms -- is data, and a completed panel
// changes none of it.

/// The compact regressors, tt x n_x, one row per period, read back off the SUR
/// matrix the file holds: z is kron(x, I_k), so x(t, j) is its element at row
/// t k and column j k.
inline arma::mat compact_regressors(const arma::mat &z, const arma::uword k)
{
    const arma::uword tt = z.n_rows / k;
    const arma::uword n_x = z.n_cols / k;
    arma::mat x(tt, n_x);
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword j = 0; j < n_x; j++)
        {
            x(t, j) = z(t * k, j * k);
        }
    }
    return x;
}

/// The p periods before the sample, k x p and oldest first, as the first row of
/// regressors holds them: y_{-m} is lag m of period zero.
///
/// These are conditioned on, as a VAR conditions on its first p observations,
/// so the host fills them for a series whose presample it did not observe --
/// interpolated, say. Drawing them too would need a prior on the state before
/// the sample, which is a later refinement.
inline arma::mat lag_presample(const arma::mat &x, const arma::uword k, const arma::uword p)
{
    arma::mat presample(k, p);
    for (arma::uword m = 1; m <= p; m++)
    {
        presample.col(p - m) = arma::trans(x.submat(0, (m - 1) * k, 0, m * k - 1));
    }
    return presample;
}

/// Overwrites the lag block of every row whose lag reaches into the sample with
/// the completed panel, k x tt. Rows whose lag reaches before the sample keep
/// what they hold: that is the presample, which is conditioned on.
inline void fill_lags(arma::mat &x, const arma::mat &path, const arma::uword p)
{
    const arma::uword k = path.n_rows;
    for (arma::uword t = 1; t < x.n_rows; t++)
    {
        for (arma::uword j = 1; j <= p && j <= t; j++)
        {
            x.submat(t, (j - 1) * k, t, j * k - 1) = arma::trans(path.col(t - j));
        }
    }
}

/// The Gaussian a draw of a constant-coefficient VAR puts on its own path:
/// `a_matrix` is the draw's k x n_x coefficients, `x` the compact regressors
/// (only the columns past the lag block are read), `covariance` the error
/// covariance.
inline VarPathPrior var_path_prior(const arma::mat &a_matrix, const arma::mat &x,
                                   const arma::mat &presample, const arma::mat &covariance,
                                   const arma::uword p)
{
    const arma::uword k = covariance.n_rows;
    const arma::uword lag = k * p;
    const arma::uword n_x = a_matrix.n_cols;

    VarPathPrior prior;
    prior.presample = presample;
    prior.coefficients = lag > 0 ? arma::mat(a_matrix.cols(0, lag - 1)) : arma::mat(k, 0);
    prior.offset = n_x > lag ? arma::mat(a_matrix.cols(lag, n_x - 1) * x.cols(lag, n_x - 1).t())
                             : arma::mat(k, x.n_rows, arma::fill::zeros);
    prior.covariance = covariance;
    return prior;
}

/// The combinations a VAR sampler reading constraints refuses, beside the
/// checks require_supported_constraints() has already made on the set itself.
///
/// Each is refused rather than approximated, and each is something later work
/// may lift: soft rows need their variances drawn, which is a block of its own;
/// a score, or constraints on the horizon, need the forecast to start from each
/// draw's own completed panel, which the scoring code does not yet do; a
/// structural form or an i.i.d. block rearranges the coefficients the lag block
/// is read from.
inline void require_completion_spec(const VarSpec &spec, const TrainData &train,
                                    const TestData &test, const char *algorithm)
{
    if (train.constraints.empty())
    {
        if (!test.constraints.empty())
        {
            throw std::invalid_argument(std::string(algorithm) +
                                        " does not score against /data/test/constraints yet");
        }
        return;
    }

    const std::string name(algorithm);
    if (!test.constraints.empty() || test.y.n_elem > 0)
    {
        throw std::invalid_argument(
            name + " does not score a forecast from a panel not observed whole yet: drop "
                   "/data/test/y and /data/test/constraints, or score from a complete panel");
    }
    if (train.constraints.group.n_elem > 0 && train.constraints.group.max() > 0)
    {
        throw std::invalid_argument(
            name + " reads hard constraints only so far: /data/train/constraints/group is "
                   "above zero somewhere, and the variance of a soft row is not drawn yet");
    }
    if (spec.structural || spec.n_iid > 0)
    {
        throw std::invalid_argument(
            name + " does not combine /data/train/constraints with a structural form or "
                   "n_iid: both rearrange the coefficients the lags are read from");
    }

    const arma::uword k = spec.k > 0 ? static_cast<arma::uword>(spec.k) : 0;
    const arma::uword want = k * static_cast<arma::uword>(spec.n_x());
    if (train.z.n_cols != want && !(train.z.n_cols == 0 && spec.p == 0))
    {
        throw std::invalid_argument(
            name + " rebuilds the lags of a panel it completes from /data/train/z, so z must "
                   "have k n_x = " + std::to_string(want) + " columns, the lag block first; got " +
            std::to_string(train.z.n_cols));
    }
}

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_COMPLETION_SUPPORT_H
