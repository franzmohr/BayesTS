// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_constrained_var_path.cpp
/// @brief Checks the data-completion step against the Gaussian it conditions,
///        written out densely.
///
/// Given its parameters a VAR is a Gaussian over its own path, and what was
/// observed of that path is a set of linear rows, exact or with an error. So
/// the distribution the completion step draws from has a closed form: write
/// the path's mean and covariance out as dense matrices, stack the rows into
/// one matrix G, and condition on G x + e = value by the textbook formula. The
/// same for the log density, period by period, as a difference of two dense
/// normal densities. Every check here is that identity, to rounding, on a
/// small path -- the sampler's arithmetic is band storage, a partition, a
/// Cholesky and a correction for the hard rows, and the dense route shares
/// none of it.
///
/// The cases are chosen so that each part is exercised on its own and then
/// together: pinned entries, hard aggregates, soft rows, a soft row wide
/// enough to widen the band past the VAR's own, coefficients and covariances
/// that change every period, and no lags at all. Two degenerate cases have an
/// answer of their own: a path pinned whole has the VAR's conditional log
/// likelihood as its log density, and a path with nothing observed has the
/// VAR's unconditional moments.
///
/// Around those, the draws: many of them, whose mean has to land on the exact
/// one to sampling error and which have to satisfy every hard row to rounding
/// in every single draw; a constant argument and a stack of copies of it giving
/// the same draw from one seed; and the shapes the step refuses.

#include "core/algorithms/constrained_var_path.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using bayests::Constraints;
using bayests::core::constrained_var_path_log_density;
using bayests::core::constrained_var_path_moments;
using bayests::core::draw_constrained_var_path;
using bayests::core::VarPathPrior;

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

void check_below(const std::string &what, const double got, const double bound)
{
    const bool ok = got < bound;
    std::printf("  %-66s %s   %.3e < %.3e\n", what.c_str(), ok ? "ok" : "FAILED", got, bound);
    if (!ok)
    {
        failures++;
    }
}

/// Largest absolute difference, relative to the larger of one and the largest
/// absolute reference value.
double gap(const arma::mat &got, const arma::mat &want)
{
    return arma::abs(got - want).max() / std::max(1.0, arma::abs(want).max());
}

// ---------------------------------------------------------------------------
// Building constraint sets
// ---------------------------------------------------------------------------

struct RowBuilder
{
    std::vector<double> value, weight;
    std::vector<arma::uword> group, row, period, variable;

    void add(const double v, const arma::uword g,
             const std::vector<std::pair<arma::uword, arma::uword>> &entries,
             const std::vector<double> &weights)
    {
        const arma::uword r = value.size();
        value.push_back(v);
        group.push_back(g);
        for (arma::uword e = 0; e < entries.size(); e++)
        {
            row.push_back(r);
            period.push_back(entries[e].first);
            variable.push_back(entries[e].second);
            weight.push_back(weights[e]);
        }
    }

    Constraints build() const
    {
        Constraints c;
        c.value = arma::vec(value);
        c.weight = arma::vec(weight);
        c.group = arma::conv_to<arma::uvec>::from(group);
        c.row = arma::conv_to<arma::uvec>::from(row);
        c.period = arma::conv_to<arma::uvec>::from(period);
        c.variable = arma::conv_to<arma::uvec>::from(variable);
        return c;
    }
};

// ---------------------------------------------------------------------------
// The dense reference
// ---------------------------------------------------------------------------

struct Dense
{
    arma::vec mean;       // of vec(Y)
    arma::mat covariance; // of vec(Y)
};

arma::mat block_of(const arma::mat &m, const arma::uword t, const arma::uword k)
{
    return m.n_rows == k ? m : arma::mat(m.rows(t * k, t * k + k - 1));
}

/// The VAR's path as a dense Gaussian: H x = d + u.
Dense dense_prior(const VarPathPrior &prior)
{
    const arma::uword k = prior.offset.n_rows;
    const arma::uword tt = prior.offset.n_cols;
    const arma::uword p = prior.coefficients.n_cols / k;
    const arma::uword n = k * tt;

    arma::mat h = arma::eye<arma::mat>(n, n);
    arma::mat s(n, n, arma::fill::zeros);
    arma::vec d(n);
    for (arma::uword t = 0; t < tt; t++)
    {
        const arma::mat a = p > 0 ? block_of(prior.coefficients, t, k) : arma::mat();
        arma::vec dt = prior.offset.col(t);
        for (arma::uword j = 1; j <= p; j++)
        {
            const arma::mat aj = a.cols((j - 1) * k, j * k - 1);
            if (t >= j)
            {
                h.submat(t * k, (t - j) * k, t * k + k - 1, (t - j) * k + k - 1) = -aj;
            }
            else
            {
                dt += aj * prior.presample.col(p + t - j);
            }
        }
        d.subvec(t * k, t * k + k - 1) = dt;
        s.submat(t * k, t * k, t * k + k - 1, t * k + k - 1) = block_of(prior.covariance, t, k);
    }
    const arma::mat h_inv = arma::inv(h);
    return {h_inv * d, h_inv * s * h_inv.t()};
}

struct DenseRows
{
    arma::mat g;     // rows x n
    arma::vec value; // rows
    arma::vec noise; // rows: 0 hard, the variance soft
    arma::uvec last; // the last period of each row
};

DenseRows dense_rows(const Constraints &c, const arma::vec &variances, const arma::uword k,
                     const arma::uword n)
{
    DenseRows out;
    const arma::uword rows = c.value.n_elem;
    out.g.zeros(rows, n);
    out.value = c.value;
    out.noise.zeros(rows);
    out.last.zeros(rows);
    for (arma::uword r = 0; r < rows; r++)
    {
        if (c.group(r) > 0)
        {
            out.noise(r) = variances(c.group(r) - 1);
        }
    }
    for (arma::uword e = 0; e < c.row.n_elem; e++)
    {
        out.g(c.row(e), c.period(e) * k + c.variable(e)) += c.weight(e);
        out.last(c.row(e)) = std::max(out.last(c.row(e)), c.period(e));
    }
    return out;
}

Dense dense_conditioned(const Dense &prior, const DenseRows &rows)
{
    if (rows.g.n_rows == 0)
    {
        return prior;
    }
    const arma::mat cross = prior.covariance * rows.g.t();
    const arma::mat cov_o = rows.g * cross + arma::diagmat(rows.noise);
    const arma::mat gain = cross * arma::inv_sympd(0.5 * (cov_o + cov_o.t()));
    const arma::vec mean = prior.mean + gain * (rows.value - rows.g * prior.mean);
    arma::mat cov = prior.covariance - gain * cross.t();
    return {mean, 0.5 * (cov + cov.t())};
}

double dense_log_normal(const arma::vec &x, const arma::vec &mean, const arma::mat &cov)
{
    if (x.n_elem == 0)
    {
        return 0.0;
    }
    double log_det = 0.0, sign = 0.0;
    arma::log_det(log_det, sign, cov);
    const arma::vec r = x - mean;
    return -0.5 * (static_cast<double>(x.n_elem) * std::log(2.0 * arma::datum::pi) + log_det +
                   arma::as_scalar(r.t() * arma::solve(cov, r)));
}

/// The log density of the rows ending by t, less that of the rows ending
/// before t.
arma::vec dense_log_density(const Dense &prior, const DenseRows &rows, const arma::uword tt)
{
    arma::vec out(tt, arma::fill::zeros);
    const auto upto = [&](const arma::sword t) {
        const arma::uvec keep = arma::find(arma::conv_to<arma::ivec>::from(rows.last) <= t);
        const arma::mat g = rows.g.rows(keep);
        const arma::mat cov = g * prior.covariance * g.t() + arma::diagmat(rows.noise(keep));
        return dense_log_normal(rows.value(keep), g * prior.mean, cov);
    };
    for (arma::uword t = 0; t < tt; t++)
    {
        out(t) = upto(static_cast<arma::sword>(t)) - upto(static_cast<arma::sword>(t) - 1);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

struct Case
{
    VarPathPrior prior;
    Constraints c;
    arma::vec variances;
};

arma::mat spd(const arma::uword k, const double scale, const double shift)
{
    arma::mat a(k, k);
    for (arma::uword i = 0; i < k; i++)
    {
        for (arma::uword j = 0; j < k; j++)
        {
            a(i, j) = std::sin(1.3 * static_cast<double>(i + 1) + 0.7 * static_cast<double>(j) + shift);
        }
    }
    return scale * (a * a.t() / static_cast<double>(k) + 0.5 * arma::eye<arma::mat>(k, k));
}

arma::mat coefficients(const arma::uword k, const arma::uword p, const double shift)
{
    arma::mat a(k, k * p);
    for (arma::uword i = 0; i < a.n_rows; i++)
    {
        for (arma::uword j = 0; j < a.n_cols; j++)
        {
            a(i, j) = 0.35 * std::cos(0.9 * static_cast<double>(i) + 1.7 * static_cast<double>(j) + shift) /
                      static_cast<double>(1 + j / k);
        }
    }
    return a;
}

/// Three series over nine months with two lags: the first monthly and pinned
/// throughout; the second quarterly, each quarter's average observed exactly;
/// the third pinned for six months and then a ragged edge, with a noisy
/// reading of month eight. Two soft rows beside those: a cross-section of the
/// first two series in month five, and a six-month sum of the third that is
/// wider than the VAR's own band.
Case mixed(const bool stacked, const arma::uword p = 2)
{
    const arma::uword k = 3, tt = 9;
    Case out;
    out.prior.offset.set_size(k, tt);
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            out.prior.offset(i, t) = 0.2 * static_cast<double>(i) - 0.05 * static_cast<double>(t);
        }
    }
    out.prior.presample.set_size(k, p);
    for (arma::uword j = 0; j < p; j++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            out.prior.presample(i, j) = 0.4 * static_cast<double>(i) - 0.3 * static_cast<double>(j) + 0.1;
        }
    }
    if (stacked)
    {
        out.prior.coefficients.set_size(tt * k, k * p);
        out.prior.covariance.set_size(tt * k, k);
        for (arma::uword t = 0; t < tt; t++)
        {
            out.prior.coefficients.rows(t * k, t * k + k - 1) =
                coefficients(k, p, 0.3 * static_cast<double>(t));
            out.prior.covariance.rows(t * k, t * k + k - 1) =
                spd(k, 0.5 + 0.1 * static_cast<double>(t), 0.2 * static_cast<double>(t));
        }
    }
    else
    {
        out.prior.coefficients = coefficients(k, p, 0.0);
        out.prior.covariance = spd(k, 0.8, 0.0);
    }

    RowBuilder rows;
    for (arma::uword t = 0; t < tt; t++)
    {
        rows.add(1.0 + 0.1 * static_cast<double>(t), 0, {{t, 0}}, {1.0});
    }
    for (arma::uword q = 0; q < 3; q++)
    {
        rows.add(0.5 * static_cast<double>(q) - 0.2, 0, {{3 * q, 1}, {3 * q + 1, 1}, {3 * q + 2, 1}},
                 {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
    }
    for (arma::uword t = 0; t < 6; t++)
    {
        rows.add(-0.3 + 0.2 * static_cast<double>(t), 0, {{t, 2}}, {2.0});
    }
    rows.add(0.7, 1, {{7, 2}}, {1.0});
    rows.add(1.1, 2, {{4, 0}, {4, 1}}, {0.6, 0.4});
    rows.add(2.0, 2, {{3, 2}, {4, 2}, {5, 2}, {6, 2}, {7, 2}, {8, 2}}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    out.c = rows.build();
    out.variances = {0.3, 0.05};
    return out;
}

void check_identities(const std::string &name, const Case &cs)
{
    const arma::uword k = cs.prior.offset.n_rows;
    const arma::uword tt = cs.prior.offset.n_cols;
    const Dense prior = dense_prior(cs.prior);
    const DenseRows rows = dense_rows(cs.c, cs.variances, k, k * tt);
    const Dense want = dense_conditioned(prior, rows);

    const auto got = constrained_var_path_moments(cs.prior, cs.c, cs.variances);
    check_below(name + ": mean", gap(arma::vectorise(got.mean), want.mean), 1e-9);
    check_below(name + ": covariance", gap(got.covariance, want.covariance), 1e-9);

    const arma::vec log_density = constrained_var_path_log_density(cs.prior, cs.c, cs.variances);
    check_below(name + ": log density, period by period",
                gap(log_density, dense_log_density(prior, rows, tt)), 1e-9);
}

void test_identities()
{
    std::printf("the conditioned Gaussian, against the dense formula\n");
    check_identities("mixed, constant", mixed(false));
    check_identities("mixed, a block per period", mixed(true));
    check_identities("mixed, one lag", mixed(false, 1));
    check_identities("mixed, no lags", mixed(false, 0));

    // Each part on its own.
    Case pins = mixed(false);
    {
        RowBuilder rows;
        rows.add(0.4, 0, {{2, 1}}, {1.0});
        rows.add(-0.1, 0, {{5, 0}}, {0.5});
        pins.c = rows.build();
        pins.variances.reset();
    }
    check_identities("pinned entries only", pins);

    Case aggregates = mixed(false);
    {
        RowBuilder rows;
        rows.add(0.4, 0, {{0, 1}, {1, 1}, {2, 1}}, {1.0, 2.0, 3.0});
        rows.add(-0.2, 0, {{4, 0}, {4, 2}}, {1.0, -1.0});
        aggregates.c = rows.build();
        aggregates.variances.reset();
    }
    check_identities("hard aggregates only", aggregates);

    Case soft = mixed(false);
    {
        RowBuilder rows;
        rows.add(0.4, 1, {{0, 1}, {8, 1}}, {1.0, 1.0});
        rows.add(-0.2, 1, {{4, 0}}, {1.0});
        soft.c = rows.build();
        soft.variances = {0.2};
    }
    check_identities("soft rows only, one across the whole path", soft);

    // The comparison has teeth: against the dense answer for a soft variance
    // half as large, every one of the three is off by far more than rounding.
    Case wrong = soft;
    wrong.variances = {0.1};
    const Dense prior = dense_prior(soft.prior);
    const DenseRows rows = dense_rows(wrong.c, wrong.variances, 3, 27);
    const Dense want = dense_conditioned(prior, rows);
    const auto got = constrained_var_path_moments(soft.prior, soft.c, soft.variances);
    check("a wrong reference is told apart: mean",
          gap(arma::vectorise(got.mean), want.mean) > 1e-3);
    check("a wrong reference is told apart: covariance", gap(got.covariance, want.covariance) > 1e-3);
    check("a wrong reference is told apart: log density",
          gap(constrained_var_path_log_density(soft.prior, soft.c, soft.variances),
              dense_log_density(prior, rows, 9)) > 1e-3);
}

/// Pinned whole, the path is the data and its log density is the VAR's
/// conditional likelihood: each period's error against its own covariance.
void test_pinned_whole()
{
    std::printf("a path pinned whole\n");
    Case cs = mixed(true);
    const arma::uword k = 3, tt = 9, p = 2;
    arma::mat y(k, tt);
    RowBuilder rows;
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword i = 0; i < k; i++)
        {
            y(i, t) = std::sin(1.1 * static_cast<double>(t) + static_cast<double>(i));
            rows.add(y(i, t), 0, {{t, i}}, {1.0});
        }
    }
    cs.c = rows.build();
    cs.variances.reset();

    arma::vec want(tt);
    for (arma::uword t = 0; t < tt; t++)
    {
        const arma::mat a = block_of(cs.prior.coefficients, t, k);
        arma::vec mean = cs.prior.offset.col(t);
        for (arma::uword j = 1; j <= p; j++)
        {
            const arma::vec lag = t >= j ? arma::vec(y.col(t - j)) : arma::vec(cs.prior.presample.col(p + t - j));
            mean += a.cols((j - 1) * k, j * k - 1) * lag;
        }
        want(t) = dense_log_normal(y.col(t), mean, block_of(cs.prior.covariance, t, k));
    }
    check_below("log density is the VAR's conditional log likelihood",
                gap(constrained_var_path_log_density(cs.prior, cs.c, cs.variances), want), 1e-10);
    check_below("and the draw is the data",
                gap(draw_constrained_var_path(cs.prior, cs.c, cs.variances), y), 1e-14);
}

/// Nothing observed, the moments are the VAR's own and every period's density
/// is that of nothing, zero.
void test_nothing_observed()
{
    std::printf("a path with nothing observed\n");
    Case cs = mixed(true);
    cs.c = Constraints();
    cs.variances.reset();
    const Dense prior = dense_prior(cs.prior);
    const auto got = constrained_var_path_moments(cs.prior, cs.c, cs.variances);
    check_below("mean is the VAR's", gap(arma::vectorise(got.mean), prior.mean), 1e-10);
    check_below("covariance is the VAR's", gap(got.covariance, prior.covariance), 1e-10);
    check("log density is zero in every period",
          arma::all(constrained_var_path_log_density(cs.prior, cs.c, cs.variances) == 0.0));
}

void test_draws()
{
    std::printf("draws\n");
    const Case cs = mixed(true);
    const arma::uword k = 3, tt = 9;
    const auto moments = constrained_var_path_moments(cs.prior, cs.c, cs.variances);
    const Dense prior = dense_prior(cs.prior);
    const DenseRows rows = dense_rows(cs.c, cs.variances, k, k * tt);
    const arma::uvec hard = arma::find(rows.noise == 0.0);

    arma::arma_rng::set_seed(20260927);
    const arma::uword n_draws = 20000;
    arma::mat draws(k * tt, n_draws);
    double worst_hard = 0.0;
    for (arma::uword i = 0; i < n_draws; i++)
    {
        draws.col(i) = arma::vectorise(draw_constrained_var_path(cs.prior, cs.c, cs.variances));
        const arma::vec residual = rows.g.rows(hard) * draws.col(i) - rows.value(hard);
        worst_hard = std::max(worst_hard, arma::abs(residual).max());
    }
    check_below("every hard row holds in every draw", worst_hard, 1e-10);

    const arma::vec mean = arma::vectorise(moments.mean);
    const arma::vec sd = arma::sqrt(moments.covariance.diag());
    const arma::vec sample_mean = arma::mean(draws, 1);
    const arma::vec sample_sd = arma::stddev(draws, 0, 1);
    double worst_z = 0.0, worst_sd = 0.0;
    for (arma::uword j = 0; j < mean.n_elem; j++)
    {
        if (sd(j) < 1e-8)
        {
            continue; // pinned, or determined by a hard row: checked above
        }
        worst_z = std::max(worst_z, std::abs(sample_mean(j) - mean(j)) /
                                        (sd(j) / std::sqrt(static_cast<double>(n_draws))));
        worst_sd = std::max(worst_sd, std::abs(sample_sd(j) / sd(j) - 1.0));
    }
    check_below("sample means within five standard errors of the exact ones", worst_z, 5.0);
    check_below("sample standard deviations within 5 percent", worst_sd, 0.05);

    // A constant argument and a stack of copies of it are the same model.
    Case constant = mixed(false);
    Case copies = constant;
    copies.prior.coefficients = arma::repmat(constant.prior.coefficients, tt, 1);
    copies.prior.covariance = arma::repmat(constant.prior.covariance, tt, 1);
    arma::arma_rng::set_seed(7);
    const arma::mat one = draw_constrained_var_path(constant.prior, constant.c, constant.variances);
    arma::arma_rng::set_seed(7);
    const arma::mat two = draw_constrained_var_path(copies.prior, copies.c, copies.variances);
    check_below("a constant block and a stack of copies draw the same path", gap(one, two), 1e-12);
    (void)prior;
}

bool refuses(const Case &cs)
{
    try
    {
        draw_constrained_var_path(cs.prior, cs.c, cs.variances);
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

void test_refusals()
{
    std::printf("refusals\n");
    Case cs = mixed(false);
    cs.variances = {0.3};
    check("fewer soft variances than groups", refuses(cs));

    cs = mixed(false);
    cs.variances = {0.3, 0.0};
    check("a soft variance of zero", refuses(cs));

    cs = mixed(false);
    cs.prior.covariance = arma::mat(2, 3, arma::fill::ones);
    check("a covariance of the wrong shape", refuses(cs));

    cs = mixed(false);
    cs.prior.coefficients = arma::mat(3, 4, arma::fill::zeros);
    check("coefficients that are not whole lags", refuses(cs));

    cs = mixed(false);
    cs.prior.presample = arma::mat(3, 1, arma::fill::zeros);
    check("a presample shorter than p", refuses(cs));

    cs = mixed(false);
    cs.c.period(0) = 9;
    check("a period past the path", refuses(cs));

    cs = mixed(false);
    cs.prior.offset(0, 0) = arma::datum::nan;
    check("a NaN in the offset", refuses(cs));
}

} // namespace

int main()
{
    try
    {
        test_identities();
        test_pinned_whole();
        test_nothing_observed();
        test_draws();
        test_refusals();
    }
    catch (const std::exception &e)
    {
        std::printf("FAILED: threw: %s\n", e.what());
        failures++;
    }

    if (failures != 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
