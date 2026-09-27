// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_COMPLETION_SUPPORT_H
#define BAYESTS_CORE_MODELS_COMPLETION_SUPPORT_H

#include "bayests/data.h"
#include "bayests/priors.h"
#include "bayests/spec.h"
#include "core/algorithms/constrained_var_path.h"
#include "core/models/model_support.h"

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

/// The last p periods of a panel, k x p and oldest first: where a horizon
/// starts. `panel` is k x tt.
inline arma::mat horizon_presample(const arma::mat &panel, const arma::uword p)
{
    if (p == 0)
    {
        return arma::mat(panel.n_rows, 0);
    }
    return panel.cols(panel.n_cols - p, panel.n_cols - 1);
}

/// A constraint set pinning every entry of `realised`, one row per period and
/// one column per variable: a horizon realised whole, in the form a horizon
/// realised in part takes. Its log density under a path's prior is the
/// pointwise log likelihood of those periods.
inline Constraints pin_every_entry(const arma::mat &realised)
{
    const arma::uword periods = realised.n_rows;
    const arma::uword k = realised.n_cols;
    const arma::uword n = periods * k;
    Constraints c;
    c.value.set_size(n);
    c.group.zeros(n);
    c.row = arma::regspace<arma::uvec>(0, n - 1);
    c.period.set_size(n);
    c.variable.set_size(n);
    c.weight.ones(n);
    for (arma::uword t = 0; t < periods; t++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            const arma::uword r = t * k + i;
            c.value(r) = realised(t, i);
            c.period(r) = t;
            c.variable(r) = i;
        }
    }
    return c;
}

/// How many soft groups a constraint set has: its largest group, soft groups
/// being numbered 1..G with none left out.
inline arma::uword soft_groups(const Constraints &c)
{
    return c.group.n_elem > 0 ? c.group.max() : 0;
}

/// The error variances the completion step is handed, one per soft group, from
/// their precisions.
inline arma::vec soft_variances(const arma::vec &precision)
{
    return precision.n_elem > 0 ? arma::vec(1.0 / precision) : arma::vec();
}

/// One Gibbs draw of each soft group's error precision given the completed
/// panel, k x tt: a gamma prior is conjugate, so group g's posterior has shape
/// shape_g + n_g / 2 and rate rate_g + (sum of its rows' squared errors) / 2,
/// the error of a row being its value less what the panel says of it. Hard rows
/// hold exactly and contribute nothing.
///
/// Consumes one gamma draw per group and none where no row is soft, so a set of
/// hard rows moves no random number.
inline void draw_soft_precisions(arma::vec &precision, const Constraints &c, const arma::mat &path,
                                 const GammaPrior &prior)
{
    const arma::uword groups = soft_groups(c);
    if (groups == 0)
    {
        return;
    }

    arma::vec fitted(c.value.n_elem, arma::fill::zeros);
    for (arma::uword e = 0; e < c.row.n_elem; e++)
    {
        fitted(c.row(e)) += c.weight(e) * path(c.variable(e), c.period(e));
    }
    arma::vec sse(groups, arma::fill::zeros);
    arma::vec count(groups, arma::fill::zeros);
    for (arma::uword r = 0; r < c.value.n_elem; r++)
    {
        if (c.group(r) > 0)
        {
            const double error = c.value(r) - fitted(r);
            sse(c.group(r) - 1) += error * error;
            count(c.group(r) - 1) += 1.0;
        }
    }
    for (arma::uword g = 0; g < groups; g++)
    {
        precision(g) = arma::randg<double>(arma::distr_param(
            prior.shape(g) + 0.5 * count(g), 1.0 / (prior.rate(g) + 0.5 * sse(g))));
    }
}

/// Refuses soft rows in the training set without a prior and a start for their
/// precisions: one shape, one rate and one initial precision per group, each
/// positive and finite. Nothing is asked of a set without soft rows.
inline void require_soft_prior(const Constraints &c, const GammaPrior &prior,
                               const arma::vec &initial, const char *algorithm)
{
    const arma::uword groups = soft_groups(c);
    if (groups == 0)
    {
        return;
    }
    const auto positive = [&](const arma::vec &v, const std::string &what) {
        if (v.n_elem != groups)
        {
            throw std::invalid_argument(
                std::string(algorithm) + ": /data/train/constraints has " + std::to_string(groups) +
                " soft groups, so " + what + " needs one value per group, got " +
                std::to_string(v.n_elem));
        }
        require_finite(v, what);
        if (!(v.min() > 0.0))
        {
            throw std::invalid_argument(what + " must be positive in every group");
        }
    };
    positive(prior.shape, "/priors/constraints/shape");
    positive(prior.rate, "/priors/constraints/rate");
    positive(initial, "/initial/constraints_inv");
}

/// The combinations a VAR sampler reading constraints refuses, beside the
/// checks require_supported_constraints() has already made on the sets
/// themselves. `forecast` carries the scenario, `test` what the horizon
/// realised.
///
/// Each is refused rather than approximated. Soft rows on the horizon would
/// need a variance nothing draws or gives. A score of a conditional
/// forecast is refused for what it would mean: a score is the density of what
/// the horizon realised under the model's own forecast, and a scenario replaces
/// that forecast with another. A structural form or an i.i.d. block rearranges
/// the coefficients the lag block is read from.
inline void require_completion_spec(const VarSpec &spec, const TrainData &train,
                                    const ForecastData &forecast, const TestData &test,
                                    const char *algorithm)
{
    if (train.constraints.empty() && test.constraints.empty() && forecast.constraints.empty())
    {
        return;
    }

    // Soft rows in the sample have their variances drawn. On the horizon nothing
    // would inform one: a realised value or a scenario with an error would need
    // the error's variance given, which the file has no place for yet.
    const std::string name(algorithm);
    const auto require_hard = [&](const Constraints &c, const char *where) {
        if (soft_groups(c) > 0)
        {
            throw std::invalid_argument(
                name + " reads hard constraints only in " + where + ": its group is above zero "
                "somewhere, and nothing draws or gives the variance of a soft row there");
        }
    };
    require_hard(test.constraints, "/data/test/constraints");
    require_hard(forecast.constraints, "/data/forecast/constraints");

    if (!forecast.constraints.empty() && (test.y.n_elem > 0 || !test.constraints.empty()))
    {
        throw std::invalid_argument(
            name + " does not score a conditional forecast: a score is the density of what the "
                   "horizon realised under the model's own forecast, and "
                   "/data/forecast/constraints replaces that forecast with a scenario. Score "
                   "without the scenario, or condition without /data/test");
    }
    if (spec.structural || spec.n_iid > 0)
    {
        throw std::invalid_argument(
            name + " does not combine constraints with a structural form or n_iid: both "
                   "rearrange the coefficients the lags are read from");
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
