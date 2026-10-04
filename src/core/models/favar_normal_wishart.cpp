// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "bayests/favar_normal_wishart.h"

#include "core/algorithms/wishart.h"
#include "core/models/favar_support.h"
#include "core/models/forecast_states.h"
#include "core/models/model_support.h"
#include "core/models/predictive_score.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bayests
{

namespace
{

using core::covariance_root;
using core::deterministic_by_period;
using core::deterministic_coefficients;
using core::draw_measurement_deterministic;
using core::draw_observed_deterministic;
using core::forecast_deterministic;
using core::draw_conditional_factor_path;
using core::draw_diagonal_precision;
using core::draw_normal_precision;
using core::favar_identified_loadings;
using core::favar_lambda_row_width;
using core::fill_favar_lambda;
using core::fill_lagged_factors;
using core::obs_factors_by_period;
using core::response_by_period;
using core::stacked_state;
using core::transition_residuals;

} // namespace

FavarNormalWishartDraws
FavarNormalWishartSampler::draw_coefficients(const FavarNormalWishartInput &input,
                                             Reporter &reporter) const
{
    input.validate();

    const int k = input.spec.k;               // observed series in the panel
    const int n = input.spec.n_factors;       // unobserved factors
    const int n_obs = input.spec.n_obs_factors; // observed factors
    const int ns = input.spec.n_state();      // the state the transition runs over
    const int p = input.spec.p;               // order of the state transition
    const int iterations = input.spec.iterations;
    const int draws = input.spec.draws();

    const int tt = static_cast<int>(input.train.periods(k));
    const arma::mat x_raw = response_by_period(input.train, k, tt);

    // The observed half of the state. Transposed once here rather than once per
    // draw: it is data and does not change over the chain.
    const arma::mat obs_raw = obs_factors_by_period(input.train);

    // Deterministic terms, in two places. The panel's enter the measurement,
    // C d_t, as in a dynamic factor model, and `x_t` is the panel less them.
    // The observed factors' are the mean C_obs d_t they deviate from, and the
    // transition runs over the deviations: `obs` is the observed factors less
    // their deterministic terms, and `state` is built from it. The measurement
    // is written on the observed factors themselves, so it reads `state_raw`.
    // Without deterministic terms both pairs are the data, the same matrices,
    // and nothing below changes.
    const arma::uword n_det = static_cast<arma::uword>(input.spec.n);
    const bool use_c = n_det > 0;
    const arma::mat d_t = deterministic_by_period(input.train);
    arma::mat c_mat = deterministic_coefficients(input.initial.c, k, n_det);
    arma::mat c_obs_mat = deterministic_coefficients(input.initial.c_obs, n_obs, n_det);
    arma::mat x_t = use_c ? arma::mat(x_raw - c_mat * d_t) : x_raw;
    arma::mat obs = use_c ? arma::mat(obs_raw - c_obs_mat * d_t) : obs_raw;

    const int n_a = input.spec.n_favar_a();
    const bool use_a = n_a > 0;

    FavarNormalWishartDraws out;
    out.lambda = arma::mat(k * ns, iterations);
    out.factors = arma::mat(n * tt, iterations);
    out.u_sigma_inv = arma::mat(k, iterations);
    out.v_sigma_inv = arma::mat(ns * ns, iterations);
    if (use_c)
    {
        out.c = arma::mat(k * n_det, iterations);
        out.c_obs = arma::mat(n_obs * n_det, iterations);
    }

    // Loadings. The leading n x n block of the factor columns is unit lower
    // triangular and the observed columns of those rows are zero; neither is
    // ever drawn.
    arma::mat lambda = favar_identified_loadings(k, n, n_obs);
    fill_favar_lambda(lambda, input.initial.lambda, n, n_obs);

    const arma::vec &lambda_prior_mu = input.lambda_prior.mu;
    const arma::mat &lambda_prior_vinv = input.lambda_prior.v_inv;

    // State transition, a VAR over the whole state rather than over the factors
    // alone -- which is the model.
    arma::vec a, a_prior_mu;
    arma::mat a_prior_vinv, a_mat, x_a;
    if (use_a)
    {
        a_prior_mu = input.a_prior.mu;
        a_prior_vinv = input.a_prior.v_inv;
        a = input.initial.a;
        a_mat = arma::reshape(a, ns, ns * p);
        x_a = arma::mat(ns * p, tt);
        out.a = arma::mat(n_a, iterations);
    }
    else
    {
        // A transition of order zero is a zero transition, exactly as in a
        // dynamic factor model: s_t = 0 s_{t-1} + v_t is the serially
        // independent state the model then has, and its prior on the first
        // state is Q.
        a_mat = arma::zeros<arma::mat>(ns, ns);
    }

    // The order chan_jeliazkov_2009 sees, which is one where this model has
    // none.
    const int p_state = std::max(p, 1);

    // Idiosyncratic precision. Diagonal by assumption -- that assumption is what
    // makes this a factor model -- so it is carried as the diagonal.
    const arma::vec u_post_shape = input.u_sigma_prior.shape + tt * 0.5;
    const arma::vec &u_prior_rate = input.u_sigma_prior.rate;
    arma::vec u_sigma_inv = input.initial.u_sigma_inv;

    // State innovation precision. A whole matrix, which is what separates this
    // model from every DFM: see the class comment.
    const arma::mat &v_prior_scale = input.v_sigma_prior.scale;
    const int v_post_df = input.v_sigma_prior.df + tt;
    arma::mat v_sigma_inv = input.initial.v_sigma_inv;

    const arma::mat state_identity = arma::eye<arma::mat>(ns, ns);
    arma::mat factors, state, state_raw, u, v;

    // Start simulation
    for (int draw = 0; draw < draws; draw++)
    {
        reporter.check_interrupt();
        reporter.progress(draw + 1, draws);

        // Block 1: Draw the path of the unobserved factors ----
        //
        // The observed factors are handed over as `obs` and held there. They are
        // not subtracted out of the panel first: their contribution to the
        // measurement is one of the cross terms the conditioning removes, so
        // doing it by hand as well would subtract it twice.
        const arma::mat u_sigma = arma::diagmat(1.0 / u_sigma_inv);
        const arma::mat v_sigma = arma::solve(v_sigma_inv, state_identity);

        //
        // The path is drawn in the deviations the transition runs over, so the
        // panel it is measured against is moved by what the loadings make of
        // the observed factors' deterministic terms as well.
        const arma::mat x_path =
            use_c ? arma::mat(x_t - lambda.tail_cols(n_obs) * (c_obs_mat * d_t)) : x_t;
        factors = draw_conditional_factor_path(x_path, lambda, u_sigma, arma::symmatu(v_sigma),
                                               a_mat, obs, ns, p_state);
        state = stacked_state(factors, obs);
        if (use_c)
        {
            state_raw = stacked_state(factors, obs_raw);
        }
        const arma::mat &measured = use_c ? state_raw : state;

        // Block 2: Draw the loadings, equation by equation ----
        //
        // Not as one vector, because the equations do not share a design: the
        // first n rows are the identifying block and carry no free loading at
        // all, and every row after them regresses on the whole state. Unlike a
        // dynamic factor model there is no partial row and nothing moves to the
        // left-hand side -- the identifying block being the identity rather than
        // a unit triangle is exactly what removes that case.
        int pos = 0;
        for (int i = 1; i < k; i++)
        {
            const int width = favar_lambda_row_width(i, n, n_obs);
            if (width == 0)
            {
                continue;
            }

            const arma::mat s_i = measured.rows(0, width - 1);
            const arma::rowvec response = x_t.row(i);

            const arma::mat prior_vinv =
                lambda_prior_vinv.submat(pos, pos, pos + width - 1, pos + width - 1);
            const arma::mat post_v = prior_vinv + (s_i * arma::trans(s_i)) * u_sigma_inv(i);
            const arma::vec rhs = prior_vinv * lambda_prior_mu.subvec(pos, pos + width - 1) +
                                  s_i * arma::trans(response) * u_sigma_inv(i);

            lambda.submat(i, 0, i, width - 1) = arma::trans(draw_normal_precision(post_v, rhs));
            pos += width;
        }

        // Block 2b: Draw the panel's deterministic terms, row by row ----
        if (use_c)
        {
            draw_measurement_deterministic(c_mat, x_raw - lambda * state_raw, d_t, u_sigma_inv,
                                           input.c_prior);
            x_t = x_raw - c_mat * d_t;
        }

        // Block 3: Draw the idiosyncratic precision ----
        u = x_t - lambda * measured;
        draw_diagonal_precision(u_sigma_inv, u, u_post_shape, u_prior_rate);

        // Block 4: Draw the state innovation precision ----
        //
        // The one block that is a matrix draw rather than a diagonal one. Q
        // couples the factor innovations with the shock to the observed
        // variables, and that cross covariance is what a FAVAR is estimated to
        // measure.
        v = transition_residuals(state, a_mat, ns, p);
        v_sigma_inv = wishart(v, v_prior_scale, v_post_df);

        // Block 5: Draw the state transition ----
        //
        // The ns equations share their regressors, so the SUR design is
        // kron(X_a', I_ns) and its posterior precision collapses to
        // kron(X_a X_a', Q^-1). The identity holds for a full Q exactly as it
        // does for the diagonal one a dynamic factor model has -- it is a
        // statement about the regressors, not about the covariance.
        if (use_a)
        {
            fill_lagged_factors(x_a, state, ns, p);

            const arma::mat post_v =
                a_prior_vinv + arma::kron(x_a * arma::trans(x_a), v_sigma_inv);
            a = draw_normal_precision(
                post_v, a_prior_vinv * a_prior_mu +
                            arma::vectorise(v_sigma_inv * (state * arma::trans(x_a))));
            a_mat = arma::reshape(a, ns, ns * p);
        }

        // Block 6: Draw the observed factors' deterministic terms ----
        //
        // Through the transition, jointly; see draw_observed_deterministic().
        if (use_c)
        {
            c_obs_mat = draw_observed_deterministic(state_raw, d_t, a_mat, v_sigma_inv, ns, n_obs,
                                                    p, input.c_obs_prior);
            obs = obs_raw - c_obs_mat * d_t;
        }

        // Store draws
        if (input.spec.keeps(draw))
        {
            const int draw_pos = input.spec.kept_index(draw);

            out.lambda.col(draw_pos) = arma::vectorise(lambda);
            out.factors.col(draw_pos) = arma::vectorise(factors);
            out.u_sigma_inv.col(draw_pos) = u_sigma_inv;
            out.v_sigma_inv.col(draw_pos) = arma::vectorise(v_sigma_inv);
            if (use_c)
            {
                out.c.col(draw_pos) = arma::vectorise(c_mat);
                out.c_obs.col(draw_pos) = arma::vectorise(c_obs_mat);
            }
            if (use_a)
            {
                out.a.col(draw_pos) = a;
            }
        }
    }

    reporter.finish();
    return out;
}

ForecastDraws FavarNormalWishartSampler::forecast(const FavarNormalWishartInput &input,
                                                  const FavarNormalWishartDraws &coefficients,
                                                  Reporter &reporter) const
{
    const int k = input.spec.k;
    const int n = input.spec.n_factors;
    const int n_obs = input.spec.n_obs_factors;
    const int ns = input.spec.n_state();
    const int p = input.spec.p;
    const int h = input.spec.h;
    const bool use_a = input.use_a();

    if (k <= 0 || n <= 0 || n_obs <= 0)
    {
        throw std::invalid_argument("a factor augmented VAR must have at least one observed "
                                    "series (k), one factor (n_factors) and one observed factor "
                                    "(n_obs_factors)");
    }
    if (h <= 0)
    {
        throw std::invalid_argument("forecast horizon (h) must be positive");
    }
    if (!coefficients.has_factors())
    {
        throw std::invalid_argument("posterior draws of the factors are missing; a factor "
                                    "augmented VAR forecasts by running the transition on from "
                                    "them");
    }
    if (coefficients.lambda.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of lambda are missing");
    }
    if (coefficients.u_sigma_inv.n_elem == 0 || coefficients.v_sigma_inv.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of the error precisions are missing");
    }
    if (use_a && !coefficients.has_a())
    {
        throw std::invalid_argument("the state has a transition of order " + std::to_string(p) +
                                    " but posterior draws of a are missing");
    }

    const int tt = static_cast<int>(input.train.periods(k));
    if (tt < p)
    {
        throw std::invalid_argument("a transition of order " + std::to_string(p) +
                                    " needs that many states to start from, and the sample has " +
                                    std::to_string(tt));
    }
    if (coefficients.factors.n_rows != static_cast<arma::uword>(n * tt))
    {
        throw std::invalid_argument(
            "posterior draws of the factors must have " + std::to_string(n * tt) + " rows for " +
            std::to_string(n) + " factors over " + std::to_string(tt) + " periods, got " +
            std::to_string(coefficients.factors.n_rows));
    }

    // The observed half of the history the transition starts from. A dynamic
    // factor model takes its whole starting state out of the posterior; here
    // half of it is data, and this is where that half comes from.
    const arma::mat obs = core::obs_factors_by_period(input.train);
    if (obs.n_cols != static_cast<arma::uword>(tt) ||
        obs.n_rows != static_cast<arma::uword>(n_obs))
    {
        throw std::invalid_argument("the observed factors must cover the whole sample, " +
                                    std::to_string(n_obs) + " of them over " +
                                    std::to_string(tt) + " periods");
    }

    // The deterministic terms over the horizon, n x h, and over the sample, for
    // the observed factors' deviations the transition starts from. Both empty
    // without them.
    const arma::mat d_h = forecast_deterministic(input.spec, input.forecast, coefficients.c,
                                                 static_cast<arma::uword>(k),
                                                 static_cast<arma::uword>(h));
    const arma::mat d_t = deterministic_by_period(input.train);
    if (d_h.n_rows > 0 &&
        coefficients.c_obs.n_rows != static_cast<arma::uword>(n_obs) * d_h.n_rows)
    {
        throw std::invalid_argument("posterior draws of the observed factors' deterministic "
                                    "coefficients c_obs are missing");
    }

    const arma::uword draws = coefficients.iterations();
    const int width = k + n_obs;
    arma::mat fcst(h * width, draws);

    // The path carries the p states the transition needs before the first
    // horizon, so column p + i is horizon i and the lag lookup is one expression
    // at every horizon rather than a split between history and forecast.
    arma::mat path(ns, p + h);

    for (arma::uword draw = 0; draw < draws; draw++)
    {
        reporter.check_interrupt();
        reporter.progress(static_cast<long long>(draw) + 1, static_cast<long long>(draws));

        const arma::mat lambda = arma::reshape(coefficients.lambda.col(draw), k, ns);
        const arma::vec u_sd = 1.0 / arma::sqrt(coefficients.u_sigma_inv.col(draw));
        // Once per draw rather than once per horizon: Q does not move over the
        // forecast, so its factorisation does not either.
        const arma::mat v_root =
            covariance_root(arma::reshape(coefficients.v_sigma_inv.col(draw), ns, ns));

        const bool use_c = d_h.n_rows > 0;
        const arma::mat c_mat =
            use_c ? arma::mat(arma::reshape(coefficients.c.col(draw), k, d_h.n_rows)) : arma::mat();
        const arma::mat c_obs_mat =
            use_c ? arma::mat(arma::reshape(coefficients.c_obs.col(draw), n_obs, d_h.n_rows))
                  : arma::mat();

        if (p > 0)
        {
            const arma::mat drawn = arma::reshape(coefficients.factors.col(draw), n, tt);
            path.submat(0, 0, n - 1, p - 1) = drawn.tail_cols(p);
            // The transition runs over the deviations, so it starts from them.
            path.submat(n, 0, ns - 1, p - 1) =
                use_c ? arma::mat(obs.tail_cols(p) - c_obs_mat * d_t.tail_cols(p))
                      : arma::mat(obs.tail_cols(p));
        }

        const arma::mat a_mat =
            use_a ? arma::reshape(coefficients.a.col(draw), ns, ns * p) : arma::mat();

        for (int i = 0; i < h; i++)
        {
            arma::vec s = v_root * arma::randn<arma::vec>(ns);
            for (int j = 1; j <= p; j++)
            {
                s += a_mat.cols((j - 1) * ns, j * ns - 1) * path.col(p + i - j);
            }
            path.col(p + i) = s;

            // The panel first, then the observed factors of the same horizon.
            // They are part of the state and so are forecast rather than
            // measured, which is why they are not read off the loadings.
            //
            // With deterministic terms the path is in deviations: the observed
            // factors are their deviation plus their mean, and the panel is
            // measured on the observed factors themselves plus its own.
            arma::vec measured = s;
            if (use_c)
            {
                measured.tail(n_obs) += c_obs_mat * d_h.col(i);
            }
            fcst.submat(i * width, draw, i * width + k - 1, draw) =
                lambda * measured + u_sd % arma::randn<arma::vec>(k);
            if (use_c)
            {
                fcst.submat(i * width, draw, i * width + k - 1, draw) += c_mat * d_h.col(i);
            }
            fcst.submat(i * width + k, draw, (i + 1) * width - 1, draw) = measured.tail(n_obs);
        }
    }

    reporter.finish();
    return ForecastDraws{fcst};
}

arma::mat
FavarNormalWishartSampler::log_likelihood(const FavarNormalWishartInput &input,
                                          const FavarNormalWishartDraws &coefficients) const
{
    const int k = input.spec.k;
    const int n = input.spec.n_factors;
    const int ns = input.spec.n_state();

    if (k <= 0 || n <= 0)
    {
        throw std::invalid_argument("model must have at least one observed series (k) and one "
                                    "factor (n_factors)");
    }
    if (coefficients.u_sigma_inv.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of u_sigma_inv are missing");
    }
    if (!coefficients.has_factors())
    {
        throw std::invalid_argument("posterior draws of the factors are missing; the likelihood "
                                    "of a factor augmented VAR is conditional on them");
    }
    if (coefficients.lambda.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of lambda are missing");
    }

    const int tt = static_cast<int>(input.train.periods(k));
    const arma::mat x_t = response_by_period(input.train, k, tt);
    const arma::mat obs = core::obs_factors_by_period(input.train);
    const arma::uword draws = coefficients.iterations();

    const arma::uword n_det = static_cast<arma::uword>(std::max(input.spec.n, 0));
    const arma::mat d_t = deterministic_by_period(input.train);
    if (n_det > 0 && coefficients.c.n_rows != static_cast<arma::uword>(k) * n_det)
    {
        throw std::invalid_argument("posterior draws of the deterministic coefficients c are "
                                    "missing");
    }

    arma::mat loglik(draws, tt);
    const double part_a = -k * std::log(2 * arma::datum::pi) / 2;

    for (arma::uword draw = 0; draw < draws; draw++)
    {
        const arma::mat lambda = arma::reshape(coefficients.lambda.col(draw), k, ns);
        const arma::mat factors = arma::reshape(coefficients.factors.col(draw), n, tt);
        const arma::vec u_sigma_inv = coefficients.u_sigma_inv.col(draw);

        // The panel is scored against the whole state, drawn half and observed
        // half alike. The observed factors are data the state is conditioned on
        // rather than a further n_obs columns of fit, so they contribute to what
        // the panel is explained by and not to the count of what is explained.
        const arma::mat state = stacked_state(factors, obs);

        // R is diagonal, so the determinant term is a sum of logs and the
        // quadratic form is a weighted sum of squares -- no k x k anything.
        const double part_b = arma::accu(arma::log(u_sigma_inv)) / 2;
        arma::mat u = x_t - lambda * state;
        if (n_det > 0)
        {
            u -= arma::reshape(coefficients.c.col(draw), k, n_det) * d_t;
        }

        for (int i = 0; i < tt; i++)
        {
            const double part_c = -arma::dot(u_sigma_inv, arma::square(u.col(i))) / 2;
            loglik(draw, i) = part_a + part_b + part_c;
        }
    }

    return loglik;
}

arma::mat FavarNormalWishartSampler::predictive_log_density(
    const FavarNormalWishartInput &input, const FavarNormalWishartDraws &coefficients) const
{
    const int k = input.spec.k;
    const int n = input.spec.n_factors;
    const int n_obs = input.spec.n_obs_factors;
    const int ns = input.spec.n_state();
    const int p = input.spec.p;
    const bool use_a = input.use_a();

    core::require_scorable(input.spec, "a factor augmented VAR");
    if (k <= 0 || n <= 0 || n_obs <= 0)
    {
        throw std::invalid_argument("a factor augmented VAR must have at least one observed "
                                    "series (k), one factor (n_factors) and one observed factor "
                                    "(n_obs_factors)");
    }
    if (!coefficients.has_factors())
    {
        throw std::invalid_argument("posterior draws of the factors are missing; a factor "
                                    "augmented VAR is scored by filtering on from them");
    }
    if (coefficients.lambda.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of lambda are missing");
    }
    if (coefficients.u_sigma_inv.n_elem == 0 || coefficients.v_sigma_inv.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of the error precisions are missing");
    }
    if (use_a && !coefficients.has_a())
    {
        throw std::invalid_argument("the state has a transition of order " + std::to_string(p) +
                                    " but posterior draws of a are missing");
    }

    // What the horizon realised, one row per period: the k panel series and then
    // the n_obs observed factors -- the width the forecast has, since both are
    // what a FAVAR forecasts.
    const arma::mat &realised = input.test.y;
    const arma::uword periods = core::scored_horizons(realised, input.spec);
    const arma::uword width = static_cast<arma::uword>(k + n_obs);
    if (realised.n_cols != width)
    {
        throw std::invalid_argument("the realised values of a factor augmented VAR must have "
                                    "k + n_obs_factors = " + std::to_string(width) +
                                    " columns, the panel and then the observed factors, got " +
                                    std::to_string(realised.n_cols));
    }

    const int tt = static_cast<int>(input.train.periods(k));
    if (tt < p)
    {
        throw std::invalid_argument("a transition of order " + std::to_string(p) +
                                    " needs that many states to start from, and the sample has " +
                                    std::to_string(tt));
    }
    if (coefficients.factors.n_rows != static_cast<arma::uword>(n * tt))
    {
        throw std::invalid_argument(
            "posterior draws of the factors must have " + std::to_string(n * tt) + " rows for " +
            std::to_string(n) + " factors over " + std::to_string(tt) + " periods, got " +
            std::to_string(coefficients.factors.n_rows));
    }
    const arma::mat obs = obs_factors_by_period(input.train);
    if (obs.n_cols != static_cast<arma::uword>(tt) ||
        obs.n_rows != static_cast<arma::uword>(n_obs))
    {
        throw std::invalid_argument("the observed factors must cover the whole sample, " +
                                    std::to_string(n_obs) + " of them over " +
                                    std::to_string(tt) + " periods");
    }

    // Deterministic terms over the scored periods and over the sample, as the
    // forecast reads them. Both empty without them.
    const arma::mat d_h = forecast_deterministic(input.spec, input.forecast, coefficients.c,
                                                 static_cast<arma::uword>(k), periods);
    const arma::mat d_t = deterministic_by_period(input.train);
    const bool use_c = d_h.n_rows > 0;
    if (use_c && coefficients.c_obs.n_rows != static_cast<arma::uword>(n_obs) * d_h.n_rows)
    {
        throw std::invalid_argument("posterior draws of the observed factors' deterministic "
                                    "coefficients c_obs are missing");
    }

    // The filter of factor_score.h, with the two things that make a FAVAR
    // different from a dynamic factor model:
    //
    //   * the observation is the panel and the observed factors together,
    //     z_i = (x_i', y_i')', and the observed factors are the second block of
    //     the state measured without error, so Z = [Lambda; 0 I] and
    //     R = diag(u, 0). F stays positive definite through the state noise,
    //     and after the update the filter is certain of the observed block, as
    //     it should be: it has just seen it.
    //   * the state noise Q is a full matrix (a Wishart precision), not a
    //     diagonal.
    //
    // With deterministic terms the state is in deviations, as in the forecast:
    // the observed factors are taken less C_obs d_i, and the panel less C d_i
    // and less the loadings on the observed factors' means, Lambda_obs C_obs d_i.
    //
    // The state stacks the p states the transition reaches back over, most
    // recent first. The filter starts from the drawn factors and the observed
    // history, certain of both -- the convention of every factor model here.
    const arma::uword blocks = p > 0 ? static_cast<arma::uword>(p) : 1;
    const arma::uword uns = static_cast<arma::uword>(ns);
    const arma::uword n_big = uns * blocks;
    const arma::uword uk = static_cast<arma::uword>(k);
    const arma::uword un = static_cast<arma::uword>(n);
    const arma::uword uobs = static_cast<arma::uword>(n_obs);

    const arma::uword draws = coefficients.iterations();
    arma::mat loglik(draws, periods);
    const double constant = -static_cast<double>(width) * std::log(2 * arma::datum::pi) / 2;

    // The shift that carries yesterday's blocks down one, which never changes.
    arma::mat transition(n_big, n_big, arma::fill::zeros);
    if (blocks > 1)
    {
        transition.submat(uns, 0, n_big - 1, n_big - uns - 1) =
            arma::eye<arma::mat>(n_big - uns, n_big - uns);
    }
    arma::mat state_noise(n_big, n_big, arma::fill::zeros);
    arma::mat observation(width, n_big, arma::fill::zeros);
    observation.submat(uk, un, width - 1, uns - 1) = arma::eye<arma::mat>(uobs, uobs);
    arma::vec measurement_var(width, arma::fill::zeros);

    for (arma::uword draw = 0; draw < draws; draw++)
    {
        const arma::mat lambda = arma::reshape(coefficients.lambda.col(draw), uk, uns);
        observation.submat(0, 0, uk - 1, uns - 1) = lambda;
        measurement_var.head(uk) = 1.0 / coefficients.u_sigma_inv.col(draw);

        const arma::mat q = arma::inv_sympd(
            arma::symmatu(arma::reshape(coefficients.v_sigma_inv.col(draw), uns, uns)));
        state_noise.submat(0, 0, uns - 1, uns - 1) = (q + arma::trans(q)) / 2;
        if (use_a)
        {
            transition.submat(0, 0, uns - 1, n_big - 1) =
                arma::reshape(coefficients.a.col(draw), uns, uns * static_cast<arma::uword>(p));
        }
        else
        {
            transition.submat(0, 0, uns - 1, n_big - 1).zeros();
        }

        const arma::mat c_mat =
            use_c ? arma::mat(arma::reshape(coefficients.c.col(draw), uk, d_h.n_rows)) : arma::mat();
        const arma::mat c_obs_mat =
            use_c ? arma::mat(arma::reshape(coefficients.c_obs.col(draw), uobs, d_h.n_rows))
                  : arma::mat();

        // The starting state: the last p periods of the drawn factors and of the
        // observed factors' deviations, most recent first.
        arma::vec state(n_big, arma::fill::zeros);
        if (p > 0)
        {
            const arma::mat drawn = arma::reshape(coefficients.factors.col(draw), un, tt);
            for (arma::uword j = 0; j < blocks; j++)
            {
                const arma::uword t = static_cast<arma::uword>(tt) - 1 - j;
                state.subvec(j * uns, j * uns + un - 1) = drawn.col(t);
                arma::vec y_dev = obs.col(t);
                if (use_c)
                {
                    y_dev -= c_obs_mat * d_t.col(t);
                }
                state.subvec(j * uns + un, (j + 1) * uns - 1) = y_dev;
            }
        }
        arma::mat variance(n_big, n_big, arma::fill::zeros);

        for (arma::uword i = 0; i < periods; i++)
        {
            // Predict.
            state = transition * state;
            variance = transition * variance * arma::trans(transition) + state_noise;

            // The realised observation, in deviations where the model has them.
            arma::vec z = arma::trans(realised.row(i));
            if (use_c)
            {
                const arma::vec mean_obs = c_obs_mat * d_h.col(i);
                z.head(uk) -= c_mat * d_h.col(i) + lambda.tail_cols(uobs) * mean_obs;
                z.tail(uobs) -= mean_obs;
            }

            const arma::vec innovation = z - observation * state;
            const arma::mat gain = variance * arma::trans(observation);
            arma::mat scale = observation * gain + arma::diagmat(measurement_var);
            scale = (scale + arma::trans(scale)) / 2;

            // A Cholesky rather than arma::log_det(): F is symmetric positive
            // definite by construction, and this fails exactly when it is not.
            arma::mat chol_scale;
            if (!arma::chol(chol_scale, scale, "lower"))
            {
                throw std::runtime_error(
                    "the one step ahead forecast variance of a scored period is not positive "
                    "definite: a FAVAR measures its observed factors without error, so their "
                    "state noise must be positive definite");
            }
            const double log_det = 2.0 * arma::accu(arma::log(chol_scale.diag()));
            const arma::vec weighted =
                arma::solve(arma::trimatu(arma::trans(chol_scale)),
                            arma::solve(arma::trimatl(chol_scale), innovation));
            loglik(draw, i) = constant - log_det / 2 - arma::dot(innovation, weighted) / 2;

            // Update, K = P Z' F^-1, through the same factorisation.
            const arma::mat k_gain = arma::trans(arma::solve(
                arma::trimatu(arma::trans(chol_scale)),
                arma::solve(arma::trimatl(chol_scale), arma::trans(gain))));
            state += k_gain * innovation;
            variance -= k_gain * arma::trans(gain);
            variance = (variance + arma::trans(variance)) / 2;
        }
    }

    return loglik;
}

} // namespace bayests
