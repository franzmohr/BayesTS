// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_var_tvp_discount.cpp
/// @brief Checks the discounted VAR against the closed form it is supposed to
///        compute one period at a time.
///
/// This is the one model in the project that draws nothing, so unlike its
/// sampling siblings it has an answer rather than a distribution, and the whole
/// of it can be asserted against an expected value. Two identities pin it:
///
/// 1. **At both discounts one the model is a constant coefficient VAR**, whose
///    matrix normal inverse Wishart posterior is textbook. The filter's final
///    period must equal it -- not approximate it. Anything above the tolerance
///    below is a wrong recursion, because every operation on the two sides is
///    the same sum of the same products in a different order.
/// 2. **The reported predictive is the multivariate Student t it claims to
///    be**, recomputed here from the quantities the recursion carries.
///
/// Beside those, the retrospective pass at a discount of one has to return the
/// final posterior in every period -- a state that does not drift has one
/// posterior, not a path of them -- and validate() has to refuse the
/// combinations the model cannot express rather than running on them.
///
/// What none of this needs: a fixture, a file, or an RNG. estimate() consumes
/// no random numbers at all, which is itself asserted below by running it twice
/// and requiring the bits to agree.

#include "bayests/var_tvp_discount.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using bayests::MatrixNormalPrior;
using bayests::NullReporter;
using bayests::VarTvpDiscountEstimator;
using bayests::VarTvpDiscountInput;
using bayests::VarTvpDiscountPosterior;
using bayests::WishartPrior;

namespace
{

int failures = 0;

/// Exact to within an accumulation of rounding over a few thousand operations.
/// The recursion is not an approximation of the closed form; it is the closed
/// form computed incrementally, so anything larger is a wrong update.
constexpr double tolerance = 1e-9;

void check(const bool condition, const std::string &what)
{
    if (!condition)
    {
        std::printf("FAIL: %s\n", what.c_str());
        failures++;
    }
}

void check_close(const double actual, const double expected, const std::string &what)
{
    const double scale = std::max(1.0, std::abs(expected));
    if (!(std::abs(actual - expected) / scale < tolerance))
    {
        std::printf("FAIL: %s (got %.17g, expected %.17g)\n", what.c_str(), actual, expected);
        failures++;
    }
}

/// A small deterministic sample: an autoregression driven by a fixed sequence,
/// so the test depends on no generator.
struct Sample
{
    arma::mat y;   ///< tt x k
    arma::mat x;   ///< tt x n_reg
    arma::uword k = 2;
    arma::uword tt = 40;
    arma::uword n_reg = 3; ///< two own lags plus an intercept
};

Sample make_sample()
{
    Sample out;
    arma::mat levels(out.tt + 1, out.k, arma::fill::zeros);
    for (arma::uword t = 1; t <= out.tt; t++)
    {
        for (arma::uword j = 0; j < out.k; j++)
        {
            const double drive = std::sin(0.7 * static_cast<double>(t) +
                                          2.0 * static_cast<double>(j));
            levels(t, j) = 0.5 * levels(t - 1, j) + drive;
        }
    }

    out.y = levels.rows(1, out.tt);
    out.x.set_size(out.tt, out.n_reg);
    out.x.cols(0, out.k - 1) = levels.rows(0, out.tt - 1);
    out.x.col(out.n_reg - 1).ones();
    return out;
}

VarTvpDiscountInput make_input(const Sample &sample, const double delta_beta,
                               const double delta_sigma)
{
    VarTvpDiscountInput input;
    input.spec.k = static_cast<int>(sample.k);
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.delta_beta = delta_beta;
    input.spec.delta_sigma = delta_sigma;

    // Stored the way the files store it: vec(y'), period-major.
    input.train.y = arma::vectorise(sample.y.t());
    input.train.x = sample.x;

    input.a_prior.mean.zeros(sample.n_reg, sample.k);
    input.a_prior.cov = arma::eye<arma::mat>(sample.n_reg, sample.n_reg);
    input.u_sigma_prior.df = static_cast<int>(sample.k) + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(sample.k, sample.k);
    return input;
}

/// The conjugate posterior of a constant coefficient VAR, computed in one go.
void conjugate_posterior(const Sample &sample, const VarTvpDiscountInput &input,
                         arma::mat &mean, arma::mat &cov, arma::mat &sigma,
                         double &df)
{
    const arma::mat cov_inv = arma::inv_sympd(input.a_prior.cov);
    const double df0 = static_cast<double>(input.u_sigma_prior.df);
    const arma::mat d0 = df0 * input.u_sigma_prior.scale;

    cov = arma::inv_sympd(cov_inv + sample.x.t() * sample.x);
    mean = cov * (cov_inv * input.a_prior.mean + sample.x.t() * sample.y);
    df = df0 + static_cast<double>(sample.tt);
    const arma::mat d = d0 + sample.y.t() * sample.y +
                        input.a_prior.mean.t() * cov_inv * input.a_prior.mean -
                        mean.t() * arma::inv_sympd(cov) * mean;
    sigma = d / df;
}

void test_conjugate_identity()
{
    const Sample sample = make_sample();
    const VarTvpDiscountInput input = make_input(sample, 1.0, 1.0);

    NullReporter reporter;
    const VarTvpDiscountEstimator estimator;
    const VarTvpDiscountPosterior posterior = estimator.estimate(input, reporter);

    arma::mat mean;
    arma::mat cov;
    arma::mat sigma;
    double df = 0.0;
    conjugate_posterior(sample, input, mean, cov, sigma, df);

    const arma::uword last = posterior.periods() - 1;

    // vec(B_T) with B_T of k x n_reg, which is the transpose of the matrix the
    // closed form produces.
    const arma::vec expected_a = arma::vectorise(mean.t());
    for (arma::uword i = 0; i < expected_a.n_elem; i++)
    {
        check_close(posterior.a(i, last), expected_a(i), "coefficient " + std::to_string(i));
    }

    const arma::vec expected_sigma = arma::vectorise(sigma);
    for (arma::uword i = 0; i < expected_sigma.n_elem; i++)
    {
        check_close(posterior.u_sigma(i, last), expected_sigma(i), "sigma " + std::to_string(i));
    }

    check_close(posterior.df(last), df, "degrees of freedom");

    const arma::vec expected_cov = arma::vectorise(cov);
    for (arma::uword i = 0; i < expected_cov.n_elem; i++)
    {
        check_close(posterior.a_cov(i, last), expected_cov(i), "coefficient covariance " + std::to_string(i));
    }

    // And the reported marginal scale is the one the matrix t implies.
    const arma::mat expected_scale = arma::sqrt(cov.diag() * sigma.diag().t());
    const arma::vec flat = arma::vectorise(expected_scale.t());
    for (arma::uword i = 0; i < flat.n_elem; i++)
    {
        check_close(posterior.a_scale(i, last), flat(i), "scale " + std::to_string(i));
    }
}

void test_predictive_is_student_t()
{
    const Sample sample = make_sample();
    const double delta_beta = 0.98;
    const double delta_sigma = 0.95;
    const VarTvpDiscountInput input = make_input(sample, delta_beta, delta_sigma);

    NullReporter reporter;
    const VarTvpDiscountEstimator estimator;
    const VarTvpDiscountPosterior posterior = estimator.estimate(input, reporter);

    arma::mat m = input.a_prior.mean;
    arma::mat c = input.a_prior.cov;
    double df = static_cast<double>(input.u_sigma_prior.df);
    arma::mat d = df * input.u_sigma_prior.scale;
    const double k = static_cast<double>(sample.k);

    for (arma::uword t = 0; t < sample.tt; t++)
    {
        const arma::mat r = c / delta_beta;
        const double df_pred = delta_sigma * df;
        const arma::mat d_pred = delta_sigma * d;

        const arma::vec z = sample.x.row(t).t();
        const arma::vec rz = r * z;
        const double q = arma::dot(z, rz) + 1.0;
        const arma::vec e = sample.y.row(t).t() - m.t() * z;
        const arma::mat psi = q * d_pred / df_pred;

        const double expected =
            std::lgamma(0.5 * (df_pred + k)) - std::lgamma(0.5 * df_pred) -
            0.5 * k * std::log(df_pred * 3.14159265358979323846) -
            0.5 * std::log(arma::det(psi)) -
            0.5 * (df_pred + k) * std::log1p(arma::dot(e, arma::solve(psi, e)) / df_pred);

        check_close(posterior.loglik(t), expected, "predictive at period " + std::to_string(t));

        const arma::vec a = rz / q;
        m += a * e.t();
        c = r - (a * a.t()) * q;
        df = df_pred + 1.0;
        d = d_pred + (e * e.t()) / q;
    }
}

void test_no_drift_is_flat()
{
    const Sample sample = make_sample();
    const VarTvpDiscountInput input = make_input(sample, 1.0, 1.0);

    NullReporter reporter;
    const VarTvpDiscountEstimator estimator;
    const VarTvpDiscountPosterior posterior = estimator.estimate(input, reporter);

    const arma::uword last = posterior.periods() - 1;
    for (arma::uword t = 0; t < posterior.periods(); t++)
    {
        for (arma::uword i = 0; i < posterior.a.n_rows; i++)
        {
            check_close(posterior.a(i, t), posterior.a(i, last),
                        "smoothed coefficient is flat at period " + std::to_string(t));
        }
    }
}

void test_consumes_no_randomness()
{
    const Sample sample = make_sample();
    const VarTvpDiscountInput input = make_input(sample, 0.97, 0.96);

    NullReporter reporter;
    const VarTvpDiscountEstimator estimator;

    arma::arma_rng::set_seed(1);
    const VarTvpDiscountPosterior first = estimator.estimate(input, reporter);
    arma::arma_rng::set_seed(2);
    const VarTvpDiscountPosterior second = estimator.estimate(input, reporter);

    check(arma::approx_equal(first.a, second.a, "absdiff", 0.0),
          "estimate() is bit-identical under two different seeds");
}

void test_refusals()
{
    const Sample sample = make_sample();

    const auto throws = [](const VarTvpDiscountInput &input) {
        try
        {
            input.validate();
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        return false;
    };

    VarTvpDiscountInput ok = make_input(sample, 0.98, 0.96);
    check(!throws(ok), "a well specified model is accepted");

    VarTvpDiscountInput zero_discount = ok;
    zero_discount.spec.delta_beta = 0.0;
    check(throws(zero_discount), "a discount of zero is refused");

    VarTvpDiscountInput above_one = ok;
    above_one.spec.delta_sigma = 1.5;
    check(throws(above_one), "a discount above one is refused");

    VarTvpDiscountInput structural = ok;
    structural.spec.structural = true;
    check(throws(structural), "a structural model is refused");

    VarTvpDiscountInput selection = ok;
    selection.spec.varsel = bayests::VarSelection::bvs;
    check(throws(selection), "variable selection is refused");

    VarTvpDiscountInput sur = ok;
    sur.train.x.reset();
    check(throws(sur), "the SUR layout alone is refused");

    // The degrees of freedom settle at 1 / (1 - delta_sigma), which must clear
    // the width of the system.
    VarTvpDiscountInput short_memory = ok;
    short_memory.spec.delta_sigma = 0.4;
    check(throws(short_memory), "a memory too short for the system is refused");
}


/// The opt-in draws have to be draws from the posterior the filter reports,
/// and the quantity that catches a degrees of freedom mistake is the spread
/// rather than the mean: a wrong nu leaves the centre alone and scales every
/// coefficient's spread by the same factor.
///
/// A matrix t on `df` degrees of freedom has standard deviation
/// scale * sqrt(df / (df - 2)) in each element, so that ratio is what is
/// checked. Two things make the check able to see the mistake it guards
/// against, and both had to be chosen rather than assumed:
///
/// - **A short memory on the volatility.** The bias is
///   sqrt(df / (df - k + 1)), so it shrinks as the degrees of freedom grow.
///   At the package's usual discounts it is a couple of per cent and hides
///   inside Monte Carlo error. delta_sigma = 0.9 settles the degrees of
///   freedom near ten, where the bias is about 5 per cent.
/// - **The average across coefficients.** The mistake scales every one of them
///   by the same factor, so the mean ratio is the sharper statistic, and it is
///   held to half the tolerance the individual ones get.
void test_draws_match_the_posterior()
{
    const Sample sample = make_sample();
    const VarTvpDiscountInput input = make_input(sample, 0.98, 0.9);

    NullReporter reporter;
    const VarTvpDiscountEstimator estimator;
    const VarTvpDiscountPosterior posterior = estimator.estimate(input, reporter);

    const arma::uword period = posterior.periods() - 1;
    const arma::uword draws = 20000;

    arma::arma_rng::set_seed(20260920);
    const arma::mat drawn = estimator.draw_period(posterior, period, draws);

    check(drawn.n_rows == posterior.a.n_rows && drawn.n_cols == draws,
          "the draws are nparams by draws");

    const double df = posterior.df(period);
    const double inflation = std::sqrt(df / (df - 2.0));

    double ratio_sum = 0.0;
    for (arma::uword i = 0; i < drawn.n_rows; i++)
    {
        const arma::rowvec row = drawn.row(i);
        const double mean = arma::mean(row);
        const double sd = arma::stddev(row);
        const double expected_sd = posterior.a_scale(i, period) * inflation;

        // The mean is the posterior mean, to within the standard error of a
        // mean of `draws` of them.
        const double standard_error = expected_sd / std::sqrt(static_cast<double>(draws));
        check(std::abs(mean - posterior.a(i, period)) < 5.0 * standard_error,
              "draw mean matches the posterior mean, coefficient " + std::to_string(i));

        check(std::abs(sd / expected_sd - 1.0) < 0.03,
              "draw spread matches the matrix t, coefficient " + std::to_string(i));
        ratio_sum += sd / expected_sd;
    }

    const double mean_ratio = ratio_sum / static_cast<double>(drawn.n_rows);
    check(std::abs(mean_ratio - 1.0) < 0.015,
          "the average spread ratio is one, which a wrong inverse Wishart "
          "degrees of freedom moves off it uniformly");
}

} // namespace

int main()
{
    test_conjugate_identity();
    test_predictive_is_student_t();
    test_no_drift_is_flat();
    test_draws_match_the_posterior();
    test_consumes_no_randomness();
    test_refusals();

    if (failures == 0)
    {
        std::printf("unit_var_tvp_discount: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
