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

    if (failures > 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
