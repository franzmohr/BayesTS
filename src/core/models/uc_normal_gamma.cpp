// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "bayests/uc_normal_gamma.h"

#include "core/algorithms/chan_jeliazkov_2009.h"
#include "core/algorithms/triangular_packing.h"
#include "core/models/forecast_states.h"
#include "core/models/model_support.h"
#include "core/models/predictive_score.h"
#include "core/models/shrinkage_support.h"
#include "core/models/uc_support.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace bayests
{

namespace
{

using core::build_psi_regressors;
using core::draw_normal_precision;
using core::fill_strict_lower_triangle;
using core::stacked_response;
using core::cycle_matrix;
using core::cycle_positions;
using core::drift_path;
using core::draw_uc_trend;
using core::filter_log_density;
using core::shock_regression;
using core::ShockRegression;
using core::state_space;
using core::UcStateSpace;

void require_uc_draws(const UcNormalGammaDraws &draws, const VarSpec &spec)
{
    const arma::uword shocks = static_cast<arma::uword>(spec.n_uc_shocks());
    if (draws.u_sigma_inv.n_elem == 0)
    {
        throw std::invalid_argument("posterior draws of u_sigma_inv are missing");
    }
    if (draws.u_sigma_inv.n_rows != shocks * shocks)
    {
        throw std::invalid_argument("posterior draws of u_sigma_inv must have " +
                                    std::to_string(shocks * shocks) +
                                    " rows, the precision of the 2k shocks, got " +
                                    std::to_string(draws.u_sigma_inv.n_rows));
    }
    if (draws.tau0.n_rows != static_cast<arma::uword>(spec.k * spec.uc_trend_order()))
    {
        throw std::invalid_argument("posterior draws of tau0 must have " +
                                    std::to_string(spec.k * spec.uc_trend_order()) +
                                    " rows, got " + std::to_string(draws.tau0.n_rows));
    }
    if (spec.p > 0 && draws.a.n_rows != static_cast<arma::uword>(spec.k * spec.k * spec.p))
    {
        throw std::invalid_argument("posterior draws of the cycle coefficients a must have " +
                                    std::to_string(spec.k * spec.k * spec.p) + " rows, got " +
                                    std::to_string(draws.a.n_rows));
    }
    if (spec.n > 0 && draws.c.n_rows != static_cast<arma::uword>(spec.k * spec.n))
    {
        throw std::invalid_argument("posterior draws of the drift c must have " +
                                    std::to_string(spec.k * spec.n) + " rows, got " +
                                    std::to_string(draws.c.n_rows));
    }
}

/// One draw's parameters in the shapes the filter and the forecast read.
struct UcDraw
{
    arma::mat phi;   ///< k x kp, empty without lags.
    arma::mat c;     ///< k x n, empty without deterministic terms.
    arma::mat q;     ///< 2k x 2k precision.
    arma::vec tau0;
};

UcDraw draw_at(const VarSpec &spec, const UcNormalGammaDraws &draws, const arma::uword i)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword shocks = 2 * k;
    UcDraw out;
    if (spec.p > 0)
    {
        out.phi = arma::reshape(draws.a.col(i), k, k * static_cast<arma::uword>(spec.p));
    }
    if (spec.n > 0)
    {
        out.c = arma::reshape(draws.c.col(i), k, static_cast<arma::uword>(spec.n));
    }
    out.q = arma::reshape(draws.u_sigma_inv.col(i), shocks, shocks);
    out.tau0 = draws.tau0.col(i);
    return out;
}

arma::mat covariance_of_precision(const arma::mat &q)
{
    arma::mat sigma;
    if (!arma::inv_sympd(sigma, arma::symmatu(q)))
    {
        throw std::runtime_error("UcNormalGamma: a drawn shock precision is not symmetric "
                                 "positive definite");
    }
    return sigma;
}

/// The pointwise log density over the periods of `y`, one row per draw.
arma::mat filtered_log_density(const VarSpec &spec, const UcNormalGammaDraws &draws,
                               const arma::mat &y, const arma::mat &d)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword n_draws = draws.iterations();
    arma::mat out(n_draws, y.n_cols);
    for (arma::uword i = 0; i < n_draws; i++)
    {
        const UcDraw one = draw_at(spec, draws, i);
        const UcStateSpace ss = state_space(spec, one.phi, covariance_of_precision(one.q));
        out.row(i) = filter_log_density(ss, y, drift_path(one.c, d, k, y.n_cols), one.tau0);
    }
    return out;
}

} // namespace

UcNormalGammaDraws UcNormalGammaSampler::draw_coefficients(const UcNormalGammaInput &input,
                                                           Reporter &reporter) const
{
    input.validate();

    const VarSpec &spec = input.spec;
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword n = static_cast<arma::uword>(spec.n);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword shocks = 2 * k;
    const int iterations = spec.iterations;
    const int draws = spec.draws();

    const arma::vec stacked = stacked_response(input.train);
    const arma::uword tt = stacked.n_elem / k;
    const arma::mat y = arma::reshape(stacked, k, tt);
    const arma::mat &d = input.train.x;

    const bool use_a = input.use_a();
    const bool use_c = input.use_c();
    const bool use_psi = input.use_psi();

    // The coefficients drawn jointly given the trend: the free cycle
    // coefficients, vec(C) and tau0, under a block diagonal prior.
    const arma::uvec positions = use_a ? cycle_positions(spec) : arma::uvec();
    const arma::uword n_a = positions.n_elem;
    const arma::uword n_c = use_c ? k * n : 0;
    const arma::uword n_tau0 = k * order;
    const arma::uword n_theta = n_a + n_c + n_tau0;

    arma::mat theta_prior_vinv(n_theta, n_theta, arma::fill::zeros);
    arma::vec theta_prior_rhs(n_theta, arma::fill::zeros);
    arma::vec theta(n_theta);
    if (use_a)
    {
        theta_prior_vinv.submat(0, 0, n_a - 1, n_a - 1) = input.a_prior.v_inv;
        theta_prior_rhs.head(n_a) = input.a_prior.v_inv * input.a_prior.mu;
        theta.head(n_a) = input.initial.a;
    }
    if (use_c)
    {
        theta_prior_vinv.submat(n_a, n_a, n_a + n_c - 1, n_a + n_c - 1) = input.c_prior.v_inv;
        theta_prior_rhs.subvec(n_a, n_a + n_c - 1) = input.c_prior.v_inv * input.c_prior.mu;
        theta.subvec(n_a, n_a + n_c - 1) = input.initial.c;
    }
    theta_prior_vinv.submat(n_a + n_c, n_a + n_c, n_theta - 1, n_theta - 1) =
        input.tau0_prior.v_inv;
    theta_prior_rhs.tail(n_tau0) = input.tau0_prior.v_inv * input.tau0_prior.mu;
    theta.tail(n_tau0) = input.initial.tau0;

    // The parts of theta, as the trend draw and the forecast read them.
    const auto phi_of = [&](const arma::vec &th) {
        return use_a ? cycle_matrix(spec, th.head(n_a)) : arma::mat(k, 0);
    };
    const auto c_of = [&](const arma::vec &th) {
        return use_c ? arma::mat(arma::reshape(th.subvec(n_a, n_a + n_c - 1), k, n)) : arma::mat();
    };

    // Covariance block over the 2k shocks, exactly as VarNormalGamma's over its
    // k errors: Psi e_t ~ N(0, Omega^-1).
    const arma::uword n_psi = use_psi ? static_cast<arma::uword>(spec.n_uc_psi()) : 0;
    arma::vec psi;
    arma::mat Psi = arma::eye<arma::mat>(shocks, shocks);
    arma::mat psi_z;
    if (use_psi)
    {
        psi = input.initial.psi;
        psi_z = arma::zeros<arma::mat>(tt * (shocks - 1), n_psi);
        fill_strict_lower_triangle(Psi, psi);
    }

    const arma::vec omega_post_shape = input.u_sigma_prior.shape + static_cast<double>(tt) * 0.5;
    arma::vec omega = input.initial.u_omega_inv;
    arma::mat q = arma::trans(Psi) * arma::diagmat(omega) * Psi;

    UcNormalGammaDraws out;
    out.trend = arma::mat(k * tt, iterations);
    if (use_a)
    {
        out.a = arma::mat(k * k * p, iterations);
    }
    if (use_c)
    {
        out.c = arma::mat(n_c, iterations);
    }
    out.tau0 = arma::mat(n_tau0, iterations);
    if (use_psi)
    {
        out.psi = arma::mat(shocks * shocks, iterations);
    }
    out.u_omega_inv = arma::mat(shocks, iterations);
    out.u_sigma_inv = arma::mat(shocks * shocks, iterations);

    int unstationary_sweeps = 0;
    arma::mat tau;
    arma::mat shocks_path(shocks, tt);

    for (int draw = 0; draw < draws; draw++)
    {
        reporter.check_interrupt();
        reporter.progress(draw + 1, draws);

        // Update the trend path
        tau = draw_uc_trend(spec, y, phi_of(theta), drift_path(c_of(theta), d, k, tt),
                         theta.tail(n_tau0), q);

        // Update the cycle, the drift and tau0, jointly given the trend
        const ShockRegression reg = shock_regression(spec, y, tau, d, positions);
        arma::mat post_vinv = theta_prior_vinv;
        arma::vec post_rhs = theta_prior_rhs;
        for (arma::uword t = 0; t < tt; t++)
        {
            const arma::mat xq = arma::trans(reg.x.slice(t)) * q;
            post_vinv += xq * reg.x.slice(t);
            post_rhs += xq * reg.w.col(t);
        }
        if (spec.stationary && use_a)
        {
            // The draw_stationary() rule over the joint block: up to
            // kStationaryTries proposals, and the previous draw kept if none
            // has a stationary cycle.
            bool found = false;
            for (int attempt = 0; attempt < core::kStationaryTries && !found; attempt++)
            {
                const arma::vec proposal = draw_normal_precision(post_vinv, post_rhs);
                if (core::is_stationary(arma::vectorise(phi_of(proposal)), k, p))
                {
                    theta = proposal;
                    found = true;
                }
            }
            if (!found)
            {
                unstationary_sweeps++;
            }
        }
        else
        {
            theta = draw_normal_precision(post_vinv, post_rhs);
        }

        for (arma::uword t = 0; t < tt; t++)
        {
            shocks_path.col(t) = reg.w.col(t) - reg.x.slice(t) * theta;
        }

        // Update psi
        if (use_psi)
        {
            const arma::vec psi_y = arma::vectorise(shocks_path.rows(1, shocks - 1));
            build_psi_regressors(psi_z, shocks_path);
            const arma::mat omega_lower = arma::diagmat(omega.tail(shocks - 1));
            const arma::sp_mat weight =
                arma::kron(arma::speye<arma::sp_mat>(tt, tt), arma::sp_mat(omega_lower));
            const arma::mat wz = weight * psi_z;
            psi = draw_normal_precision(input.psi_prior.v_inv + arma::trans(wz) * psi_z,
                                        input.psi_prior.v_inv * input.psi_prior.mu +
                                            arma::trans(wz) * psi_y);
            fill_strict_lower_triangle(Psi, psi);
            shocks_path = Psi * shocks_path;
        }

        // Update Omega
        const arma::vec sse = arma::sum(arma::square(shocks_path), 1);
        for (arma::uword i = 0; i < shocks; i++)
        {
            omega(i) = arma::randg<double>(arma::distr_param(
                omega_post_shape(i), 1.0 / (input.u_sigma_prior.rate(i) + sse(i) * 0.5)));
        }
        q = arma::trans(Psi) * arma::diagmat(omega) * Psi;

        // Store draws
        if (spec.keeps(draw))
        {
            const int pos = spec.kept_index(draw);
            out.trend.col(pos) = arma::vectorise(tau);
            if (use_a)
            {
                out.a.col(pos) = arma::vectorise(phi_of(theta));
            }
            if (use_c)
            {
                out.c.col(pos) = theta.subvec(n_a, n_a + n_c - 1);
            }
            out.tau0.col(pos) = theta.tail(n_tau0);
            if (use_psi)
            {
                out.psi.col(pos) = arma::vectorise(Psi);
            }
            out.u_omega_inv.col(pos) = omega;
            out.u_sigma_inv.col(pos) = arma::vectorise(q);
        }
    }

    reporter.finish();
    if (unstationary_sweeps > 0)
    {
        reporter.message(std::to_string(unstationary_sweeps) + " of " + std::to_string(draws) +
                         " sweeps found no stationary draw of the cycle in " +
                         std::to_string(core::kStationaryTries) +
                         " tries and kept the one before: the posterior may put much of its mass "
                         "on explosive cycle coefficients");
    }
    return out;
}

ForecastDraws UcNormalGammaSampler::forecast(const UcNormalGammaInput &input,
                                             const UcNormalGammaDraws &draws,
                                             Reporter &reporter) const
{
    const VarSpec &spec = input.spec;
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const bool smooth = spec.trend == UcTrend::smooth;

    if (spec.k <= 0)
    {
        throw std::invalid_argument("model must have at least one series (k)");
    }
    if (spec.h <= 0)
    {
        throw std::invalid_argument("forecast horizon (h) must be positive");
    }
    const arma::uword h = static_cast<arma::uword>(spec.h);
    require_uc_draws(draws, spec);
    if (!draws.has_trend())
    {
        throw std::invalid_argument("posterior draws of the trend path are missing, and the "
                                    "forecast starts from its end");
    }
    if (spec.n > 0 &&
        (input.forecast.x.n_rows != h || input.forecast.x.n_cols != static_cast<arma::uword>(spec.n)))
    {
        throw std::invalid_argument(
            "the drift's " + std::to_string(spec.n) + " deterministic terms have to be continued "
            "over the horizon: /data/forecast/x must be " + std::to_string(h) + " x " +
            std::to_string(spec.n) + ", got " + std::to_string(input.forecast.x.n_rows) + " x " +
            std::to_string(input.forecast.x.n_cols));
    }

    const arma::vec stacked = stacked_response(input.train);
    const arma::uword tt = stacked.n_elem / k;
    const arma::mat y = arma::reshape(stacked, k, tt);
    if (draws.trend.n_rows != k * tt)
    {
        throw std::invalid_argument("posterior draws of the trend path must have " +
                                    std::to_string(k * tt) + " rows, k per period of the sample, "
                                    "got " + std::to_string(draws.trend.n_rows));
    }
    if (smooth && tt < 2)
    {
        throw std::invalid_argument("a smooth trend is forecast from its last two periods, and the "
                                    "sample has one");
    }

    const arma::uword n_draws = draws.iterations();
    arma::mat fcst(h * k, n_draws);

    for (arma::uword i = 0; i < n_draws; i++)
    {
        reporter.check_interrupt();
        reporter.progress(static_cast<long long>(i) + 1, static_cast<long long>(n_draws));

        const UcDraw one = draw_at(spec, draws, i);
        const arma::mat root = core::covariance_root(one.q);
        const arma::mat tau = arma::reshape(draws.trend.col(i), k, tt);

        // The cycle's last p values, most recent first, and the trend's last
        // one or two.
        arma::mat cycle_lags(k, p, arma::fill::zeros);
        for (arma::uword j = 0; j < p && j < tt; j++)
        {
            cycle_lags.col(j) = y.col(tt - 1 - j) - tau.col(tt - 1 - j);
        }
        arma::vec tau_last = tau.col(tt - 1);
        arma::vec tau_prev = smooth ? arma::vec(tau.col(tt - 2)) : arma::vec();

        for (arma::uword step = 0; step < h; step++)
        {
            const arma::vec e = root * arma::randn<arma::vec>(2 * k);

            arma::vec cycle = e.head(k);
            for (arma::uword j = 0; j < p; j++)
            {
                cycle += one.phi.cols(j * k, (j + 1) * k - 1) * cycle_lags.col(j);
            }

            arma::vec next = smooth ? arma::vec(2.0 * tau_last - tau_prev) : tau_last;
            next += e.tail(k);
            if (spec.n > 0)
            {
                next += one.c * arma::trans(input.forecast.x.row(step));
            }

            fcst.submat(step * k, i, (step + 1) * k - 1, i) = next + cycle;

            if (p > 0)
            {
                if (p > 1)
                {
                    cycle_lags.cols(1, p - 1) = arma::mat(cycle_lags.cols(0, p - 2));
                }
                cycle_lags.col(0) = cycle;
            }
            if (smooth)
            {
                tau_prev = tau_last;
            }
            tau_last = next;
        }
    }

    reporter.finish();
    return ForecastDraws{fcst};
}

arma::mat UcNormalGammaSampler::log_likelihood(const UcNormalGammaInput &input,
                                               const UcNormalGammaDraws &draws) const
{
    const VarSpec &spec = input.spec;
    if (spec.k <= 0)
    {
        throw std::invalid_argument("model must have at least one series (k)");
    }
    require_uc_draws(draws, spec);

    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::vec stacked = stacked_response(input.train);
    const arma::uword tt = stacked.n_elem / k;
    const arma::mat y = arma::reshape(stacked, k, tt);
    if (spec.n > 0 && input.train.x.n_rows != tt)
    {
        throw std::invalid_argument("the deterministic terms /data/train/x must have one row per "
                                    "period, " + std::to_string(tt) + ", got " +
                                    std::to_string(input.train.x.n_rows));
    }

    return filtered_log_density(spec, draws, y, input.train.x);
}

arma::mat UcNormalGammaSampler::predictive_log_density(const UcNormalGammaInput &input,
                                                       const UcNormalGammaDraws &draws) const
{
    const VarSpec &spec = input.spec;
    core::require_scorable(spec, "UcNormalGamma");
    const arma::uword periods = core::scored_horizons(input.test.y, spec);
    require_uc_draws(draws, spec);

    const arma::uword k = static_cast<arma::uword>(spec.k);
    if (input.test.y.n_cols != k)
    {
        throw std::invalid_argument("the realised observations /data/test/y must have one column "
                                    "per series, " + std::to_string(k) + ", got " +
                                    std::to_string(input.test.y.n_cols));
    }
    const arma::vec stacked = stacked_response(input.train);
    const arma::uword tt = stacked.n_elem / k;

    // The sample and then the scored periods, as one series the filter runs
    // over: each scored period is predicted from everything before it.
    const arma::mat y = arma::join_horiz(arma::reshape(stacked, k, tt),
                                         arma::trans(input.test.y.head_rows(periods)));
    arma::mat d;
    if (spec.n > 0)
    {
        if (input.forecast.x.n_rows < periods ||
            input.forecast.x.n_cols != static_cast<arma::uword>(spec.n))
        {
            throw std::invalid_argument(
                "scoring needs the drift's deterministic terms over the scored periods: "
                "/data/forecast/x must have at least " + std::to_string(periods) + " rows and " +
                std::to_string(spec.n) + " columns");
        }
        d = arma::join_vert(input.train.x, input.forecast.x.head_rows(periods));
    }

    const arma::mat all = filtered_log_density(spec, draws, y, d);
    return all.tail_cols(periods);
}

} // namespace bayests
