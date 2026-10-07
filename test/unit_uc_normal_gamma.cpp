// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_uc_normal_gamma.cpp
/// @brief The unobserved components model against the Gaussian it is.
///
/// Given everything but the trend path, an unobserved components model is
/// linear and Gaussian throughout: the trend and the series are fixed linear
/// maps of the stacked shocks plus a constant. Building those maps densely, by
/// running the model's recursions on unit shocks, gives an independent route to
/// every quantity the sampler computes by a structured one, and so four
/// identities, to rounding:
///
///   * the trend's posterior precision and mean, assembled as a band by
///     uc_trend_band(), against the covariance and mean of tau given y in the
///     joint normal -- the block of the sampler where an off-by-one in a lag,
///     a presample term or a sign produces a path that looks entirely
///     reasonable;
///   * the pointwise log likelihood the Kalman filter returns, summed, against
///     the log density of the stacked series under that normal;
///   * the draw chan_jeliazkov_2009_precision() makes against the mean plus the
///     inverse dense Cholesky factor applied to the same standard normals,
///     which the factor's uniqueness makes the same number;
///   * the shocks the coefficient block regresses on, rebuilt by
///     shock_regression() from the trend and the free coefficients, against the
///     shocks the sample was generated from;
///   * the predictive density of a stretch handed back as what the horizon
///     realised against the log likelihood of the same stretch.
///
/// Each over several shapes: one series and two, a cycle of order zero, one and
/// two, a VAR cycle and an own-lags one, both trends, a drift with a break, and
/// a shock precision with and without a covariance block.

#include "bayests/uc_normal_gamma.h"

#include "core/algorithms/chan_jeliazkov_2009.h"
#include "core/models/uc_skew_support.h"
#include "core/models/uc_support.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace bayests;

namespace
{

int failures = 0;

void check_below(const std::string &what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-62s %s   %.3e < %.3e\n", what.c_str(), ok ? "ok" : "FAILED", got, bound);
    if (!ok)
    {
        failures++;
    }
}

struct Case
{
    std::string name;
    int k = 1;
    int p = 0;
    int n = 0;
    UcTrend trend = UcTrend::random_walk;
    UcCycle cycle = UcCycle::var;
    bool correlated = false;
};

/// One draw's worth of parameters for a case, chosen to be admissible and
/// nothing more: a stationary cycle, a positive definite shock precision.
struct Params
{
    arma::mat phi; ///< k x kp, the zeros of an own-lags cycle included.
    arma::mat c;   ///< k x n.
    arma::mat d;   ///< periods x n.
    arma::vec tau0;
    arma::mat q;   ///< 2k x 2k.
};

VarSpec spec_of(const Case &c)
{
    VarSpec spec;
    spec.k = c.k;
    spec.p = c.p;
    spec.n = c.n;
    spec.trend = c.trend;
    spec.cycle = c.cycle;
    spec.covar = c.correlated;
    spec.iterations = 1;
    return spec;
}

Params params_of(const Case &c, const arma::uword periods)
{
    const arma::uword k = static_cast<arma::uword>(c.k);
    const arma::uword p = static_cast<arma::uword>(c.p);
    Params out;

    out.phi = arma::zeros<arma::mat>(k, k * p);
    for (arma::uword j = 0; j < p; j++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            out.phi(i, j * k + i) = j == 0 ? 0.6 : -0.2;
            if (c.cycle == UcCycle::var)
            {
                out.phi(i, j * k + (i + 1) % k) += k > 1 ? 0.1 : 0.0;
            }
        }
    }

    out.d = arma::zeros<arma::mat>(periods, static_cast<arma::uword>(c.n));
    out.c = arma::zeros<arma::mat>(k, static_cast<arma::uword>(c.n));
    for (arma::uword t = 0; t < periods; t++)
    {
        if (c.n > 0)
        {
            out.d(t, 0) = 1.0;
        }
        if (c.n > 1)
        {
            out.d(t, 1) = t >= periods / 2 ? 1.0 : 0.0; // a break in the drift
        }
    }
    for (arma::uword i = 0; i < k; i++)
    {
        if (c.n > 0)
        {
            out.c(i, 0) = 0.5 + 0.1 * static_cast<double>(i);
        }
        if (c.n > 1)
        {
            out.c(i, 1) = -0.3;
        }
    }

    const arma::uword order = c.trend == UcTrend::smooth ? 2 : 1;
    out.tau0 = arma::linspace<arma::vec>(1.0, 2.0, k * order);

    const arma::uword shocks = 2 * k;
    arma::mat psi = arma::eye<arma::mat>(shocks, shocks);
    if (c.correlated)
    {
        for (arma::uword i = 1; i < shocks; i++)
        {
            for (arma::uword j = 0; j < i; j++)
            {
                psi(i, j) = 0.3 - 0.1 * static_cast<double>(i + j);
            }
        }
    }
    const arma::vec omega = arma::linspace<arma::vec>(1.5, 4.0, shocks);
    out.q = arma::trans(psi) * arma::diagmat(omega) * psi;
    return out;
}

/// The series, the trend and the cycle as affine maps of the stacked shocks
/// e = (e_0', .., e_{T-1}')', each e_t = (u^c_t', u^tau_t')': tau = alpha + A e,
/// y = alpha + B e. Built by running the recursions once with every shock at
/// zero, for the constant, and once per unit shock, for the columns.
struct Dense
{
    arma::vec alpha;
    arma::mat a; ///< kT x 2kT, the trend.
    arma::mat b; ///< kT x 2kT, the series.
};

void run(const Case &c, const Params &pr, const arma::vec &e, arma::vec &tau_out, arma::vec &y_out,
         const arma::uword periods)
{
    const arma::uword k = static_cast<arma::uword>(c.k);
    const arma::uword p = static_cast<arma::uword>(c.p);
    const bool smooth = c.trend == UcTrend::smooth;
    arma::mat tau(k, periods), cyc(k, periods, arma::fill::zeros);
    arma::vec last = pr.tau0.head(k);
    arma::vec prev = smooth ? arma::vec(pr.tau0.subvec(k, 2 * k - 1)) : arma::vec();
    for (arma::uword t = 0; t < periods; t++)
    {
        const arma::vec et = e.subvec(t * 2 * k, (t + 1) * 2 * k - 1);
        arma::vec ct = et.head(k);
        for (arma::uword j = 1; j <= p && j <= t; j++)
        {
            ct += pr.phi.cols((j - 1) * k, j * k - 1) * cyc.col(t - j);
        }
        cyc.col(t) = ct;
        arma::vec next = smooth ? arma::vec(2.0 * last - prev) : last;
        next += et.tail(k);
        if (c.n > 0)
        {
            next += pr.c * arma::trans(pr.d.row(t));
        }
        tau.col(t) = next;
        if (smooth)
        {
            prev = last;
        }
        last = next;
    }
    tau_out = arma::vectorise(tau);
    y_out = arma::vectorise(tau + cyc);
}

Dense dense_of(const Case &c, const Params &pr, const arma::uword periods)
{
    const arma::uword k = static_cast<arma::uword>(c.k);
    const arma::uword ne = 2 * k * periods;
    Dense out;
    arma::vec tau0_path, y0;
    run(c, pr, arma::zeros<arma::vec>(ne), tau0_path, y0, periods);
    out.alpha = tau0_path;
    out.a = arma::mat(k * periods, ne);
    out.b = arma::mat(k * periods, ne);
    for (arma::uword i = 0; i < ne; i++)
    {
        arma::vec unit(ne, arma::fill::zeros);
        unit(i) = 1.0;
        arma::vec tau, y;
        run(c, pr, unit, tau, y, periods);
        out.a.col(i) = tau - tau0_path;
        out.b.col(i) = y - y0;
    }
    // The constant of y is the constant of tau: the cycle has none.
    return out;
}

double relative(const arma::mat &got, const arma::mat &want)
{
    return arma::abs(got - want).max() / std::max(1.0, arma::abs(want).max());
}

void check_case(const Case &c)
{
    std::printf("%s\n", c.name.c_str());
    const arma::uword periods = 9;
    const arma::uword k = static_cast<arma::uword>(c.k);
    const VarSpec spec = spec_of(c);
    const Params pr = params_of(c, periods);
    const Dense dn = dense_of(c, pr, periods);

    // A sample from the model itself, so the numbers are of a plausible size.
    arma::arma_rng::set_seed(20261007);
    const arma::mat sigma = arma::inv_sympd(pr.q);
    const arma::mat sigma_big = arma::kron(arma::eye<arma::mat>(periods, periods), sigma);
    const arma::vec e = arma::chol(sigma_big, "lower") * arma::randn<arma::vec>(2 * k * periods);
    const arma::vec y = dn.alpha + dn.b * e;
    const arma::mat y_mat = arma::reshape(y, k, periods);
    const arma::mat drift = core::drift_path(pr.c, pr.d, k, periods);

    // tau | y in the joint normal.
    const arma::mat cov_ty = dn.a * sigma_big * arma::trans(dn.b);
    const arma::mat cov_yy = dn.b * sigma_big * arma::trans(dn.b);
    const arma::mat cov_tt = dn.a * sigma_big * arma::trans(dn.a);
    const arma::mat gain = cov_ty * arma::inv_sympd(cov_yy);
    const arma::vec mean = dn.alpha + gain * (y - dn.alpha);
    const arma::mat cov = cov_tt - gain * arma::trans(cov_ty);

    // The band, made dense.
    core::UcTrendBand band = core::uc_trend_band(spec, y_mat, pr.phi, drift, pr.tau0, pr.q);
    arma::mat kk(k * periods, k * periods, arma::fill::zeros);
    for (arma::uword i = 0; i < periods; i++)
    {
        for (arma::uword d = 0; d <= band.band && i + d < periods; d++)
        {
            const arma::mat &blk = band.blocks[i * (band.band + 1) + d];
            kk.submat(i * k, (i + d) * k, (i + 1) * k - 1, (i + d + 1) * k - 1) = blk;
            if (d > 0)
            {
                kk.submat((i + d) * k, i * k, (i + d + 1) * k - 1, (i + 1) * k - 1) = arma::trans(blk);
            }
        }
    }
    const arma::vec rhs = arma::vectorise(band.rhs);
    check_below("trend precision against the joint normal's", relative(arma::inv_sympd(kk), cov), 1e-9);
    check_below("trend mean against the joint normal's", relative(arma::solve(kk, rhs), mean), 1e-9);

    // The draw: K = R'R, R upper, and the band's backward pass consumes one
    // block of k standard normals per period, last period first.
    arma::arma_rng::set_seed(42);
    const arma::mat drawn = chan_jeliazkov_2009_precision(band.blocks, band.rhs, band.band);
    arma::arma_rng::set_seed(42);
    arma::vec z(k * periods);
    for (arma::uword i = periods; i-- > 0;)
    {
        z.subvec(i * k, (i + 1) * k - 1) = arma::randn<arma::vec>(k);
    }
    const arma::mat r = arma::chol(kk);
    const arma::vec want = arma::solve(kk, rhs) + arma::solve(arma::trimatu(r), z);
    check_below("band draw against the dense Cholesky draw", relative(arma::vectorise(drawn), want), 1e-9);

    // The shocks, from the trend the sample was generated with and theta laid
    // out as the coefficient block draws it: the free cycle coefficients,
    // vec(C), tau0.
    const arma::vec tau_true = dn.alpha + dn.a * e;
    const arma::uvec positions = c.p > 0 ? core::cycle_positions(spec) : arma::uvec();
    const arma::vec theta = arma::join_vert(arma::vec(arma::vectorise(pr.phi).eval().elem(positions)),
                                            arma::vec(arma::vectorise(pr.c)), pr.tau0);
    const core::ShockRegression reg =
        core::shock_regression(spec, y_mat, arma::reshape(tau_true, k, periods), pr.d, positions);
    arma::vec rebuilt(2 * k * periods);
    for (arma::uword t = 0; t < periods; t++)
    {
        rebuilt.subvec(t * 2 * k, (t + 1) * 2 * k - 1) = reg.w.col(t) - reg.x.slice(t) * theta;
    }
    check_below("shocks rebuilt from the trend against the generating ones", relative(rebuilt, e), 1e-9);

    // The integrated likelihood.
    UcNormalGammaInput input;
    input.spec = spec;
    input.train.y = arma::trans(y_mat);
    input.train.x = pr.d;
    UcNormalGammaDraws draws;
    draws.a = c.p > 0 ? arma::mat(arma::vectorise(pr.phi)) : arma::mat();
    draws.c = c.n > 0 ? arma::mat(arma::vectorise(pr.c)) : arma::mat();
    draws.tau0 = pr.tau0;
    draws.u_sigma_inv = arma::vectorise(pr.q);

    const arma::mat ll = UcNormalGammaSampler{}.log_likelihood(input, draws);
    double log_det = 0.0;
    arma::log_det_sympd(log_det, cov_yy);
    const arma::vec dev = y - dn.alpha;
    const double want_ll = -0.5 * (static_cast<double>(k * periods) * std::log(2.0 * arma::datum::pi) +
                                   log_det + arma::as_scalar(arma::trans(dev) * arma::solve(cov_yy, dev)));
    check_below("integrated log likelihood against the joint normal's",
                std::abs(arma::accu(ll) - want_ll) / std::abs(want_ll), 1e-10);

    // The last three periods handed back as what a horizon realised.
    const arma::uword head = periods - 3;
    UcNormalGammaInput scored;
    scored.spec = spec;
    scored.spec.h = 3;
    scored.train.y = arma::trans(y_mat.cols(0, head - 1));
    scored.train.x = c.n > 0 ? arma::mat(pr.d.head_rows(head)) : arma::mat();
    scored.forecast.x = c.n > 0 ? arma::mat(pr.d.tail_rows(3)) : arma::mat();
    scored.test.y = arma::trans(y_mat.cols(head, periods - 1));
    const arma::mat score = UcNormalGammaSampler{}.predictive_log_density(scored, draws);
    check_below("score of a held-back stretch against its log likelihood",
                arma::abs(score - ll.tail_cols(3)).max(), 1e-10);
}

/// The general case the error blocks of the family's other members feed: a
/// shock precision that moves with the period, as under stochastic volatility,
/// shock means, as the skewed shocks' mixture representation gives them, and an
/// irregular component, under which the drawn path is the trend and the cycle
/// together. Checked against the same dense joint normal, now over the shocks
/// and the irregular component stacked, with the means in its constant.
void check_general_case(const Case &c, const bool irregular, const bool moving,
                        const bool with_mean)
{
    std::printf("%s%s%s%s\n", c.name.c_str(), irregular ? ", irregular" : "",
                moving ? ", moving precision" : "", with_mean ? ", shock means" : "");
    const arma::uword periods = 8;
    const arma::uword k = static_cast<arma::uword>(c.k);
    const arma::uword w = 2 * k;
    VarSpec spec = spec_of(c);
    spec.irregular = irregular;
    const Params pr = params_of(c, periods);

    // Sigma_t, its precision stacked one block per period, the means and the
    // irregular variances.
    const arma::mat sigma0 = arma::inv_sympd(pr.q);
    arma::mat q_path(w * periods, w), sigma_path(w * periods, w), mean(w, periods);
    arma::mat noise_cov(w * periods + k * periods, w * periods + k * periods, arma::fill::zeros);
    for (arma::uword t = 0; t < periods; t++)
    {
        const double f = moving ? std::exp(0.4 * std::sin(static_cast<double>(t))) : 1.0;
        sigma_path.rows(t * w, (t + 1) * w - 1) = sigma0 * f;
        q_path.rows(t * w, (t + 1) * w - 1) = pr.q / f;
        noise_cov.submat(t * w, t * w, (t + 1) * w - 1, (t + 1) * w - 1) = sigma0 * f;
        for (arma::uword i = 0; i < w; i++)
        {
            mean(i, t) = with_mean ? 0.2 * std::cos(static_cast<double>(t + 3 * i)) : 0.0;
        }
    }
    const arma::vec irregular_var = arma::linspace<arma::vec>(0.3, 0.6, k);
    for (arma::uword t = 0; t < periods; t++)
    {
        const arma::uword at = w * periods + t * k;
        noise_cov.submat(at, at, at + k - 1, at + k - 1) =
            irregular ? arma::mat(arma::diagmat(irregular_var)) : arma::mat(k, k, arma::fill::zeros);
    }
    const arma::uword nn = noise_cov.n_rows;

    // The recursions on (e, eps): trend, cycle and series as affine maps.
    const auto simulate = [&](const arma::vec &nu, arma::vec &tau_out, arma::vec &cyc_out,
                              arma::vec &y_out) {
        arma::vec tau, y;
        run(c, pr, nu.head(w * periods), tau, y, periods);
        y_out = y + nu.tail(k * periods);
        tau_out = tau;
        cyc_out = y - tau;
    };
    arma::vec nu0(nn, arma::fill::zeros);
    nu0.head(w * periods) = arma::vectorise(mean);
    arma::vec tau_c, cyc_c, y_c;
    simulate(nu0, tau_c, cyc_c, y_c);
    arma::mat a_tau(k * periods, nn), a_cyc(k * periods, nn), b(k * periods, nn);
    for (arma::uword i = 0; i < nn; i++)
    {
        arma::vec unit = nu0;
        unit(i) += 1.0;
        arma::vec tau, cyc, y;
        simulate(unit, tau, cyc, y);
        a_tau.col(i) = tau - tau_c;
        a_cyc.col(i) = cyc - cyc_c;
        b.col(i) = y - y_c;
    }

    // A sample: the shocks around their means, and the irregular component.
    arma::arma_rng::set_seed(20261008);
    arma::vec nu = nu0;
    for (arma::uword t = 0; t < periods; t++)
    {
        nu.subvec(t * w, (t + 1) * w - 1) +=
            arma::chol(arma::mat(sigma_path.rows(t * w, (t + 1) * w - 1)), "lower") *
            arma::randn<arma::vec>(w);
        if (irregular)
        {
            nu.subvec(w * periods + t * k, w * periods + (t + 1) * k - 1) =
                arma::sqrt(irregular_var) % arma::randn<arma::vec>(k);
        }
    }
    const arma::vec dev_nu = nu - nu0;
    const arma::vec y = y_c + b * dev_nu;
    const arma::mat y_mat = arma::reshape(y, k, periods);
    const arma::mat tau_true = arma::reshape(tau_c + a_tau * dev_nu, k, periods);
    const arma::mat cyc_true = arma::reshape(cyc_c + a_cyc * dev_nu, k, periods);
    const arma::mat drift = core::drift_path(pr.c, pr.d, k, periods);

    // The path given y. Its rows are (tau_t', c_t')' per period with an
    // irregular component, tau_t alone without.
    const arma::uword m = irregular ? w : k;
    arma::mat a_x(m * periods, nn);
    arma::vec x_c(m * periods);
    for (arma::uword t = 0; t < periods; t++)
    {
        a_x.rows(t * m, t * m + k - 1) = a_tau.rows(t * k, (t + 1) * k - 1);
        x_c.subvec(t * m, t * m + k - 1) = tau_c.subvec(t * k, (t + 1) * k - 1);
        if (irregular)
        {
            a_x.rows(t * m + k, (t + 1) * m - 1) = a_cyc.rows(t * k, (t + 1) * k - 1);
            x_c.subvec(t * m + k, (t + 1) * m - 1) = cyc_c.subvec(t * k, (t + 1) * k - 1);
        }
    }
    const arma::mat cov_xy = a_x * noise_cov * arma::trans(b);
    const arma::mat cov_yy = b * noise_cov * arma::trans(b);
    const arma::mat gain = cov_xy * arma::inv_sympd(cov_yy);
    const arma::vec want_mean = x_c + gain * (y - y_c);
    const arma::mat want_cov = a_x * noise_cov * arma::trans(a_x) - gain * arma::trans(cov_xy);

    const arma::mat q_arg = moving ? q_path : pr.q;
    const arma::mat mean_arg = with_mean ? mean : arma::mat();
    core::UcTrendBand band =
        irregular ? core::uc_state_band(spec, y_mat, pr.phi, drift, pr.tau0, q_arg,
                                        1.0 / irregular_var, mean_arg)
                  : core::uc_trend_band(spec, y_mat, pr.phi, drift, pr.tau0, q_arg, mean_arg);
    arma::mat kk(m * periods, m * periods, arma::fill::zeros);
    for (arma::uword i = 0; i < periods; i++)
    {
        for (arma::uword d = 0; d <= band.band && i + d < periods; d++)
        {
            const arma::mat &blk = band.blocks[i * (band.band + 1) + d];
            kk.submat(i * m, (i + d) * m, (i + 1) * m - 1, (i + d + 1) * m - 1) = blk;
            if (d > 0)
            {
                kk.submat((i + d) * m, i * m, (i + d + 1) * m - 1, (i + 1) * m - 1) = arma::trans(blk);
            }
        }
    }
    check_below("path precision against the joint normal's", relative(arma::inv_sympd(kk), want_cov), 1e-9);
    check_below("path mean against the joint normal's",
                relative(arma::solve(kk, arma::vectorise(band.rhs)), want_mean), 1e-9);

    // The shocks from the true path, which with an irregular component the
    // regression reads as two paths.
    const arma::uvec positions = c.p > 0 ? core::cycle_positions(spec) : arma::uvec();
    const arma::vec theta = arma::join_vert(arma::vec(arma::vectorise(pr.phi).eval().elem(positions)),
                                            arma::vec(arma::vectorise(pr.c)), pr.tau0);
    const core::ShockRegression reg =
        core::shock_regression_paths(spec, tau_true, cyc_true, pr.d, positions);
    arma::vec rebuilt(w * periods);
    for (arma::uword t = 0; t < periods; t++)
    {
        rebuilt.subvec(t * w, (t + 1) * w - 1) = reg.w.col(t) - reg.x.slice(t) * theta;
    }
    check_below("shocks rebuilt from the path against the generating ones",
                relative(rebuilt, nu.head(w * periods)), 1e-9);

    // The filter.
    const core::UcStateSpace ss = core::state_space(spec, pr.phi, moving ? arma::mat() : sigma0);
    core::UcFilterTerms terms;
    if (moving)
    {
        terms.sigma_path = sigma_path;
    }
    terms.shock_mean = mean_arg;
    if (irregular)
    {
        terms.irregular_var = irregular_var;
    }
    const arma::rowvec ll = core::filter_log_density(ss, y_mat, drift, pr.tau0, terms);
    double log_det = 0.0;
    arma::log_det_sympd(log_det, cov_yy);
    const arma::vec dev = y - y_c;
    const double want_ll = -0.5 * (static_cast<double>(k * periods) * std::log(2.0 * arma::datum::pi) +
                                   log_det + arma::as_scalar(arma::trans(dev) * arma::solve(cov_yy, dev)));
    check_below("filtered log likelihood against the joint normal's",
                std::abs(arma::accu(ll) - want_ll) / std::abs(want_ll), 1e-10);
}

/// int_0^inf g(L) p(L) dL for the latent's density p -- the half normal or the
/// unit exponential -- by Simpson's rule in u = sqrt(L), which takes out the
/// square root singularity the asymmetric Laplace's mixture has at zero.
template <typename F>
double integrate_latent(const UcSkew skew, F g)
{
    const int n = 200000;
    const double top = 9.0; // u up to 9: L up to 81, past either density's support
    const double h = top / n;
    double sum = 0.0;
    for (int i = 0; i <= n; i++)
    {
        const double u = h * i;
        const double l = u * u;
        const double density = skew == UcSkew::snd
                                   ? 2.0 * std::exp(-0.5 * l * l) / std::sqrt(2.0 * arma::datum::pi)
                                   : std::exp(-l);
        const double f = g(l) * density * 2.0 * u;
        sum += f * ((i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0));
    }
    return sum * h / 3.0;
}

double normal_pdf(const double x, const double mean, const double var)
{
    if (var <= 0.0)
    {
        return 0.0;
    }
    return std::exp(-0.5 * (x - mean) * (x - mean) / var) / std::sqrt(2.0 * arma::datum::pi * var);
}

/// The skewed shocks: each closed form marginal density against its normal
/// mixture integrated numerically, and the latent draw's mean against the
/// conditional mean the same mixture gives -- the latter statistically, at
/// five Monte Carlo standard errors. Over both distributions, both signs of
/// shock, and scales either way round.
void check_skew()
{
    std::printf("skewed shocks\n");
    const double cases[][3] = {{0.3, 0.6, -1.1}, {0.3, 0.6, 0.4}, {0.8, 0.2, 0.05},
                               {0.5, 0.5, -0.3}, {0.2, 0.9, 2.0}};
    for (const UcSkew skew : {UcSkew::snd, UcSkew::ald})
    {
        const char *name = skew == UcSkew::snd ? "skew normal" : "asymmetric Laplace";
        double worst_density = 0.0;
        double worst_mean = 0.0;
        for (const auto &c : cases)
        {
            const double a = c[0], b = c[1], x = c[2];
            core::SkewScales s;
            s.a = arma::vec{a};
            s.b = arma::vec{b};
            const double closed = core::log_marginal(skew, arma::rowvec{x}, s);

            const auto density_given = [&](const double l) {
                double m = 0.0, v = 0.0;
                core::conditional_moments(skew, a, b, l, m, v);
                return normal_pdf(x, m, v);
            };
            const double mixture = integrate_latent(skew, density_given);
            worst_density = std::max(worst_density, std::abs(closed - std::log(mixture)));

            const double conditional_mean =
                integrate_latent(skew, [&](const double l) { return l * density_given(l); }) /
                mixture;
            const double conditional_sq =
                integrate_latent(skew, [&](const double l) { return l * l * density_given(l); }) /
                mixture;
            const double sd = std::sqrt(conditional_sq - conditional_mean * conditional_mean);

            arma::arma_rng::set_seed(20261009);
            const int n = 20000;
            double total = 0.0;
            for (int i = 0; i < n; i++)
            {
                total += core::draw_latent(skew, a, b, x);
            }
            const double se = sd / std::sqrt(static_cast<double>(n));
            worst_mean = std::max(worst_mean, std::abs(total / n - conditional_mean) / se);
        }
        check_below(std::string(name) + ": log density against its mixture", worst_density, 1e-7);
        check_below(std::string(name) + ": latent draws' mean, in standard errors", worst_mean, 5.0);
    }

    // log Phi either side of where it switches to the Mills ratio expansion.
    const double inside = core::log_normal_cdf(-29.999);
    const double outside = core::log_normal_cdf(-30.001);
    check_below("log Phi continuous where its expansion takes over",
                std::abs((inside - outside) - (-0.5 * (29.999 * 29.999 - 30.001 * 30.001) +
                                               std::log(30.001 / 29.999))),
                1e-6);
}

} // namespace

int main()
{
    Case c;

    c = Case{};
    c.name = "local level: one series, white noise cycle";
    check_case(c);

    c = Case{};
    c.name = "Clark: one series, AR(2) cycle, constant drift";
    c.p = 2;
    c.n = 1;
    check_case(c);

    c = Case{};
    c.name = "Morley-Nelson-Zivot: as above, correlated shocks, a break";
    c.p = 2;
    c.n = 2;
    c.correlated = true;
    check_case(c);

    c = Case{};
    c.name = "smooth trend: one series, AR(2) cycle";
    c.p = 2;
    c.trend = UcTrend::smooth;
    check_case(c);

    c = Case{};
    c.name = "two series, VAR(2) cycle, correlated, a break";
    c.k = 2;
    c.p = 2;
    c.n = 2;
    c.correlated = true;
    check_case(c);

    c = Case{};
    c.name = "two series, own-lags AR(1) cycles, smooth, correlated";
    c.k = 2;
    c.p = 1;
    c.cycle = UcCycle::own;
    c.trend = UcTrend::smooth;
    c.correlated = true;
    check_case(c);

    c = Case{};
    c.name = "Clark with a break";
    c.p = 2;
    c.n = 2;
    check_general_case(c, true, false, false);
    check_general_case(c, false, true, false);
    check_general_case(c, false, false, true);
    check_general_case(c, true, true, true);

    c = Case{};
    c.name = "two series, VAR(2) cycle, smooth, correlated";
    c.k = 2;
    c.p = 2;
    c.trend = UcTrend::smooth;
    c.correlated = true;
    check_general_case(c, true, true, true);
    check_general_case(c, false, true, true);

    check_skew();

    if (failures > 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
