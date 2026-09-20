// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_PREDICTIVE_SCORE_H
#define BAYESTS_CORE_MODELS_PREDICTIVE_SCORE_H

#include "bayests/arma.h"
#include "bayests/data.h"
#include "bayests/spec.h"

#include <stdexcept>
#include <string>

namespace bayests::core
{

/// Scoring a forecast against what the horizon realised.
///
/// The score of horizon i is the log density of the realised observation under
/// each draw, conditional on the realised observations before it -- not on the
/// path the forecast simulated. That is the choice the whole shape rests on:
/// the log of the mean of exp() over draws is then the one step ahead
/// predictive density given everything known up to that period, and the sum of
/// those over the horizons is log p(y*_{T+1..T+h} | data), the joint. Scoring
/// against a simulated history instead would give the marginal density of each
/// horizon on its own, which is a defensible quantity and a different one --
/// marginals do not sum to a joint, so a column of them could not be added up.
///
/// The consequence that makes this cheap: with the history realised rather than
/// simulated, the regressors of the scored periods do not depend on the draw.
/// They are built once, and the score is then the model's own pointwise log
/// likelihood over those periods -- the same expression, the same code, a
/// different sample. Nothing new is written down per model, which is what keeps
/// one density per algorithm rather than two that can drift apart.

/// How many horizons a file can be scored over: the periods the realised values
/// cover, which read_test_observations() has already refused to let exceed `h`.
///
/// Throws where there is nothing to score. Callers check for realised values
/// before asking, so reaching here without them is a caller's mistake rather
/// than a file's.
inline arma::uword scored_horizons(const arma::mat &realised, const VarSpec &spec)
{
    if (realised.n_rows == 0)
    {
        throw std::invalid_argument(
            "scoring a forecast needs the observations the horizon realised, and /data/test/y "
            "holds none");
    }
    if (spec.h <= 0)
    {
        throw std::invalid_argument(
            "scoring a forecast needs a horizon, and this model asks for none");
    }
    return realised.n_rows;
}

/// Refuses a model this cannot be the density of, saying which part is missing
/// rather than producing a number of the right size and the wrong meaning.
///
/// `drifts` is whether anything the density is evaluated at moves over the
/// horizon -- time-varying coefficients, a time-varying covariance block, a
/// stochastic volatility. Those need the state carried forward per draw before
/// the density can be taken, which is the forecast's own machinery and not this
/// one's; until that is written, they are refused. `hold` does not rescue them:
/// it would score them under a model whose drift stops where the sample does,
/// which is a different density and not the one asked for.
///
/// A structural model is refused for a reason that does not go away with more
/// code here: its regressors include the contemporaneous observations, so the
/// realised row is not built from lags alone, and its density carries the
/// Jacobian of A_0 besides.
inline void require_scorable(const VarSpec &spec, const bool drifts,
                             const std::string &algorithm)
{
    if (spec.structural)
    {
        throw std::invalid_argument(
            "a structural model cannot be scored: its regressors include the contemporaneous "
            "observations and its density carries the Jacobian of A_0, neither of which the "
            "pointwise log likelihood over the scored periods accounts for");
    }
    if (drifts)
    {
        throw std::invalid_argument(
            algorithm +
            " lets its coefficients or its error precision move over the horizon, and scoring "
            "one of those needs each draw's state carried forward first, which is not "
            "implemented yet; the models with constant coefficients and a constant precision "
            "can be scored today");
    }
}

/// The regressors of the scored periods, with the lagged endogenous blocks
/// taken from what was realised.
///
/// `forecast_x` is `/data/forecast/x`, the compact layout of one period per row
/// that a forecast is driven by, and it comes in with its deterministic and
/// unmodelled columns filled and its lag blocks holding whatever the caller put
/// there. A forecast overwrites those blocks as it simulates; this overwrites
/// them from `realised` instead, and the two do it by the same rule, so a lag
/// reaching back before the first scored period is left exactly as the caller
/// supplied it -- that value is the end of the estimation sample and is already
/// right. See update_forecast_lags(), which this mirrors.
///
/// Rows beyond the scored periods are dropped: a file may forecast further than
/// it realised, and the horizons it did not realise cannot be scored.
inline arma::mat realised_regressors(const arma::mat &forecast_x, const arma::mat &realised,
                                     const int k, const int p)
{
    const arma::uword periods = realised.n_rows;
    if (forecast_x.n_rows < periods)
    {
        throw std::invalid_argument(
            "/data/forecast/x has " + std::to_string(forecast_x.n_rows) +
            " horizons but /data/test/y realised " + std::to_string(periods) +
            "; the regressors of a period that is scored have to be there");
    }
    if (realised.n_cols != static_cast<arma::uword>(k))
    {
        throw std::invalid_argument(
            "the realised values must have k = " + std::to_string(k) + " columns, got " +
            std::to_string(realised.n_cols));
    }

    arma::mat x = forecast_x.head_rows(periods);
    if (p <= 0)
    {
        return x;
    }

    const arma::uword width = static_cast<arma::uword>(k);
    for (arma::uword i = 1; i < periods; i++)
    {
        const arma::uword filled = i < static_cast<arma::uword>(p) ? i : static_cast<arma::uword>(p);
        for (arma::uword j = 1; j <= filled; j++)
        {
            x.submat(i, (j - 1) * width, i, j * width - 1) = realised.row(i - j);
        }
    }

    return x;
}

/// The SUR spelling of those regressors, which is what a sampler's pointwise
/// log likelihood reads: kron(x, I_k), one block row per period.
inline arma::mat sur_regressors(const arma::mat &x, const int k)
{
    if (x.n_cols == 0)
    {
        return arma::mat();
    }
    return arma::kron(x, arma::eye<arma::mat>(k, k));
}

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_PREDICTIVE_SCORE_H
