// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_UC_SUPPORT_H
#define BAYESTS_CORE_MODELS_UC_SUPPORT_H

// The pieces of an unobserved components model that do not depend on how its
// shocks are distributed: the cycle's coefficients, the trend's posterior
// precision, the shocks as a regression on the coefficients, and the Kalman
// filter the likelihood integrates the trend out with. UcNormalGamma reads
// them with a constant shock precision; they are here rather than in its
// source so that unit.uc_normal_gamma can put them against the dense Gaussian
// they factorise, and so that a sibling with another error block reuses them
// rather than a copy.

#include "bayests/spec.h"
#include "core/algorithms/chan_jeliazkov_2009.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bayests::core
{

/// The k x kp cycle matrix [Phi_1 .. Phi_p] the free coefficients describe.
/// Under UcCycle::own the free ones are the diagonals, lag by lag, and the rest
/// is zero.
inline arma::mat cycle_matrix(const VarSpec &spec, const arma::vec &a)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    if (spec.cycle == UcCycle::var)
    {
        return arma::reshape(a, k, k * p);
    }
    arma::mat phi(k, k * p, arma::fill::zeros);
    for (arma::uword j = 0; j < p; j++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            phi(i, j * k + i) = a(j * k + i);
        }
    }
    return phi;
}

/// Where each free cycle coefficient sits in vec([Phi_1 .. Phi_p]): every
/// position under UcCycle::var, the diagonals alone under UcCycle::own.
inline arma::uvec cycle_positions(const VarSpec &spec)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    if (spec.cycle == UcCycle::var)
    {
        return arma::regspace<arma::uvec>(0, k * k * p - 1);
    }
    arma::uvec positions(k * p);
    for (arma::uword j = 0; j < p; j++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            positions(j * k + i) = (j * k + i) * k + i;
        }
    }
    return positions;
}

/// C d_t for every period, k x periods. Zero without deterministic terms.
inline arma::mat drift_path(const arma::mat &c, const arma::mat &d, const arma::uword k,
                     const arma::uword periods)
{
    if (c.n_elem == 0)
    {
        return arma::zeros<arma::mat>(k, periods);
    }
    return c * arma::trans(d);
}

/// The posterior precision of the k x tt trend path given everything else,
/// as the upper blocks of a band, and its right hand side.
///
/// Stack the 2k shocks of period t, e_t = (u^c_t', u^tau_t')'. Both halves are
/// linear in the trend at t and at most L = max(p, trend order) periods before
/// it,
///
///     e_t = r_t + sum_{l=0..L} M_l tau_{t-l},
///
/// with the terms before the sample moved into r_t: the cycle is zero there,
/// so a lag of it that reaches back past the sample contributes nothing at all,
/// and the trend is tau0 there, which is a constant given the rest. Each M_l is
/// 2k x k, the cycle's part on top -- M_0 = [-I; I], M_j = [Phi_j; 0] for the
/// cycle's lags -- and the trend's below: -I at l = 1 for a random walk, -2I at
/// l = 1 and I at l = 2 for a smooth trend.
///
/// The log density is -sum_t e_t' Q e_t / 2 with Q the shock precision, so the
/// trend's posterior precision collects M_a' Q M_b in block (t-a, t-b) and its
/// right hand side -M_a' Q r_t in block t-a. That is block banded of bandwidth
/// L whatever Q correlates -- a trend shock correlated with the cycle shock of
/// the same period couples nothing further apart than the period itself -- and
/// it is the whole of the prior: the trend equation is a term of the joint
/// density like the cycle's, and needs no separate initialisation.
struct UcTrendBand
{
    std::vector<arma::mat> blocks; ///< tt (band + 1) blocks of k x k, as chan_jeliazkov_2009_precision reads them.
    arma::mat rhs;                 ///< k x tt.
    arma::uword band = 0;
};

inline UcTrendBand uc_trend_band(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                          const arma::mat &drift, const arma::vec &tau0, const arma::mat &q)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = y.n_cols;
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword band = std::max(p, order);
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);

    std::vector<arma::mat> m(band + 1, arma::zeros<arma::mat>(2 * k, k));
    m[0].rows(0, k - 1) = -eye_k;
    m[0].rows(k, 2 * k - 1) = eye_k;
    for (arma::uword j = 1; j <= p; j++)
    {
        m[j].rows(0, k - 1) = phi.cols((j - 1) * k, j * k - 1);
    }
    if (order == 1)
    {
        m[1].rows(k, 2 * k - 1) = -eye_k;
    }
    else
    {
        m[1].rows(k, 2 * k - 1) = -2.0 * eye_k;
        m[2].rows(k, 2 * k - 1) = eye_k;
    }

    // M_a' Q, the same in every period.
    std::vector<arma::mat> mq(band + 1);
    for (arma::uword a = 0; a <= band; a++)
    {
        mq[a] = arma::trans(m[a]) * q;
    }

    const arma::vec tau_a = tau0.head(k);
    const arma::vec tau_b = order == 2 ? arma::vec(tau0.subvec(k, 2 * k - 1)) : arma::vec();

    std::vector<arma::mat> blocks(tt * (band + 1), arma::zeros<arma::mat>(k, k));
    arma::mat rhs(k, tt, arma::fill::zeros);
    arma::vec r(2 * k);

    for (arma::uword t = 0; t < tt; t++)
    {
        // The cycle half of r_t: the data part of c_t - sum_j Phi_j c_{t-j}.
        arma::vec ry = y.col(t);
        for (arma::uword j = 1; j <= p && j <= t; j++)
        {
            ry -= phi.cols((j - 1) * k, j * k - 1) * y.col(t - j);
        }
        r.head(k) = ry;

        // The trend half: minus the drift, and the trend before the sample.
        arma::vec rt = -drift.col(t);
        if (order == 1)
        {
            if (t == 0)
            {
                rt -= tau_a;
            }
        }
        else if (t == 0)
        {
            rt += -2.0 * tau_a + tau_b;
        }
        else if (t == 1)
        {
            rt += tau_a;
        }
        r.tail(k) = rt;

        for (arma::uword a = 0; a <= band && a <= t; a++)
        {
            rhs.col(t - a) -= mq[a] * r;
            for (arma::uword b = 0; b <= a; b++)
            {
                blocks[(t - a) * (band + 1) + (a - b)] += mq[a] * m[b];
            }
        }
    }

    UcTrendBand out;
    out.blocks = std::move(blocks);
    out.rhs = std::move(rhs);
    out.band = band;
    return out;
}

/// Draws the k x tt trend path given everything else; see uc_trend_band().
inline arma::mat draw_uc_trend(const VarSpec &spec, const arma::mat &y, const arma::mat &phi,
                               const arma::mat &drift, const arma::vec &tau0, const arma::mat &q)
{
    UcTrendBand sys = uc_trend_band(spec, y, phi, drift, tau0, q);
    return chan_jeliazkov_2009_precision(std::move(sys.blocks), sys.rhs, sys.band);
}

/// The 2k shocks of every period as linear in the coefficients: for period t,
/// e_t = w.col(t) - x.slice(t) * theta, with theta the free cycle coefficients,
/// vec(C) and tau0, in that order.
struct ShockRegression
{
    arma::mat w;  ///< 2k x tt.
    arma::cube x; ///< 2k x n_theta x tt.
};

inline ShockRegression shock_regression(const VarSpec &spec, const arma::mat &y, const arma::mat &tau,
                                 const arma::mat &d, const arma::uvec &positions)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = y.n_cols;
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword n = static_cast<arma::uword>(spec.n);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword n_a = positions.n_elem;
    const arma::uword n_c = k * n;
    const arma::uword n_theta = n_a + n_c + k * order;
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);

    const arma::mat cycle = y - tau;

    ShockRegression out;
    out.w = arma::mat(2 * k, tt);
    out.x = arma::cube(2 * k, n_theta, tt, arma::fill::zeros);

    for (arma::uword t = 0; t < tt; t++)
    {
        arma::mat x_t(2 * k, n_theta, arma::fill::zeros);

        // The cycle: c_t on its p lags, (x_t' kron I_k) vec(Phi), zero before
        // the sample.
        out.w.col(t).head(k) = cycle.col(t);
        if (n_a > 0)
        {
            arma::vec lags(k * p, arma::fill::zeros);
            for (arma::uword j = 1; j <= p && j <= t; j++)
            {
                lags.subvec((j - 1) * k, j * k - 1) = cycle.col(t - j);
            }
            const arma::mat full = arma::kron(arma::trans(lags), eye_k);
            x_t.submat(0, 0, k - 1, n_a - 1) = full.cols(positions);
        }

        // The trend: its difference on the drift and, near the start, on the
        // trend before the sample.
        arma::vec diff = tau.col(t);
        if (order == 1)
        {
            if (t >= 1)
            {
                diff -= tau.col(t - 1);
            }
        }
        else
        {
            if (t >= 1)
            {
                diff -= 2.0 * tau.col(t - 1);
            }
            if (t >= 2)
            {
                diff += tau.col(t - 2);
            }
        }
        out.w.col(t).tail(k) = diff;

        if (n_c > 0)
        {
            x_t.submat(k, n_a, 2 * k - 1, n_a + n_c - 1) = arma::kron(d.row(t), eye_k);
        }
        const arma::uword at = n_a + n_c;
        if (order == 1)
        {
            if (t == 0)
            {
                x_t.submat(k, at, 2 * k - 1, at + k - 1) = eye_k;
            }
        }
        else if (t == 0)
        {
            x_t.submat(k, at, 2 * k - 1, at + k - 1) = 2.0 * eye_k;
            x_t.submat(k, at + k, 2 * k - 1, at + 2 * k - 1) = -eye_k;
        }
        else if (t == 1)
        {
            x_t.submat(k, at, 2 * k - 1, at + k - 1) = -eye_k;
        }

        out.x.slice(t) = x_t;
    }

    return out;
}

/// The innovation structure the Kalman filter below runs the model in, given
/// one draw's parameters.
///
///     s_t = T s_{t-1} + (C d_t, 0)' + R e_t,   y_t = Z s_t,
///
/// with s_t = (tau_t', [tau_{t-1}'], c_t', .., c_{t-q+1}')' and q = max(p, 1),
/// the lagged trend only for a smooth one. There is no measurement error: the
/// series is the trend plus the cycle exactly, and what makes F invertible is
/// that both carry a shock every period.
struct UcStateSpace
{
    arma::mat transition; ///< T, m x m.
    arma::mat state_cov;  ///< R Sigma R', m x m.
    arma::mat z;          ///< k x m.
    arma::uword m = 0;
    arma::uword k = 0;
};

inline UcStateSpace state_space(const VarSpec &spec, const arma::mat &phi, const arma::mat &sigma)
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword p = static_cast<arma::uword>(spec.p);
    const arma::uword order = static_cast<arma::uword>(spec.uc_trend_order());
    const arma::uword q = std::max<arma::uword>(p, 1);
    const arma::uword cyc = k * order; // where the cycle block starts
    const arma::mat eye_k = arma::eye<arma::mat>(k, k);

    UcStateSpace ss;
    ss.k = k;
    ss.m = k * order + k * q;
    ss.transition = arma::zeros<arma::mat>(ss.m, ss.m);
    if (order == 1)
    {
        ss.transition.submat(0, 0, k - 1, k - 1) = eye_k;
    }
    else
    {
        ss.transition.submat(0, 0, k - 1, k - 1) = 2.0 * eye_k;
        ss.transition.submat(0, k, k - 1, 2 * k - 1) = -eye_k;
        ss.transition.submat(k, 0, 2 * k - 1, k - 1) = eye_k;
    }
    if (p > 0)
    {
        ss.transition.submat(cyc, cyc, cyc + k - 1, cyc + k * p - 1) = phi;
    }
    for (arma::uword j = 1; j < q; j++)
    {
        ss.transition.submat(cyc + j * k, cyc + (j - 1) * k, cyc + (j + 1) * k - 1,
                             cyc + j * k - 1) = eye_k;
    }

    arma::mat r(ss.m, 2 * k, arma::fill::zeros);
    r.submat(cyc, 0, cyc + k - 1, k - 1) = eye_k;   // u^c into c_t
    r.submat(0, k, k - 1, 2 * k - 1) = eye_k;       // u^tau into tau_t
    ss.state_cov = r * sigma * arma::trans(r);

    ss.z = arma::zeros<arma::mat>(k, ss.m);
    ss.z.cols(0, k - 1) = eye_k;
    ss.z.cols(cyc, cyc + k - 1) = eye_k;
    return ss;
}

/// One step ahead log densities of y.col(0) .. y.col(periods - 1), started from
/// the trend before the sample at tau0 and the cycle at zero, both known: given
/// one draw of everything else, the start is not uncertain, and P_0 is zero.
inline arma::rowvec filter_log_density(const UcStateSpace &ss, const arma::mat &y, const arma::mat &drift,
                                const arma::vec &tau0)
{
    const arma::uword k = ss.k;
    const arma::uword periods = y.n_cols;
    const double constant = -static_cast<double>(k) * std::log(2.0 * arma::datum::pi) / 2.0;

    arma::vec s(ss.m, arma::fill::zeros);
    s.head(tau0.n_elem) = tau0;
    arma::mat p_cov(ss.m, ss.m, arma::fill::zeros);
    const arma::mat zt = arma::trans(ss.z);

    arma::rowvec out(periods);
    for (arma::uword t = 0; t < periods; t++)
    {
        arma::vec a = ss.transition * s;
        a.head(k) += drift.col(t);
        const arma::mat pred = ss.transition * p_cov * arma::trans(ss.transition) + ss.state_cov;

        const arma::vec v = y.col(t) - ss.z * a;
        const arma::mat pz = pred * zt;
        const arma::mat f = arma::symmatu(ss.z * pz);

        arma::mat root;
        if (!arma::chol(root, f))
        {
            throw std::runtime_error("UcNormalGamma: the one step prediction variance is not "
                                     "positive definite at period " + std::to_string(t + 1));
        }
        const arma::vec scaled = arma::solve(arma::trimatl(arma::trans(root)), v);
        out(t) = constant - arma::accu(arma::log(root.diag())) - arma::dot(scaled, scaled) / 2.0;

        // Update: K = P Z' F^-1, through the factor rather than an inverse.
        const arma::mat gain = arma::trans(
            arma::solve(arma::trimatu(root), arma::solve(arma::trimatl(arma::trans(root)),
                                                         arma::trans(pz))));
        s = a + gain * v;
        p_cov = arma::symmatu(pred - gain * arma::trans(pz));
    }
    return out;
}

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_UC_SUPPORT_H
