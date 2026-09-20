// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_vec_tvp_discount.cpp
/// @brief Checks the discounted VEC against the discounted VAR it is supposed
///        to be, over a design with the error correction term in front.
///
/// The claim the model makes is narrow and exact: conditional on a fixed
/// cointegration matrix, a VEC *is* a VAR in the regressors
/// `[beta' w_t, the short-run blocks]`. So the sharp test is not another
/// recomputation of the closed form -- unit_var_tvp_discount.cpp already pins
/// that against the textbook conjugate posterior -- but the equivalence itself:
///
/// 1. **The two models agree element for element** when the VAR is handed the
///    design the VEC assembles. The design is built here by an explicit double
///    loop over `sum_l beta(l, j) w(t, l)` rather than by the same matrix
///    product the model uses, so the check is not circular: it pins the
///    product, the orientation of `w`, and the fact that the `rank` error
///    correction columns come first.
/// 2. **A column of `a` is a column of a VecNormalWishart posterior**, which is
///    what lets the level machinery take it untouched. Asserted by converting a
///    posterior mean to the level VAR and comparing its first two lag blocks
///    with `I + alpha beta' + Gamma_1` and `-Gamma_1` written out by hand.
///
/// Beside those, rank zero has to be the VAR in differences rather than a
/// special case, estimate() has to consume no randomness, and validate() has to
/// refuse the fixed space it cannot run without rather than reading past it.
///
/// What none of this needs: a fixture, a file, or an RNG outside the forecast.

#include "bayests/var_tvp_discount.h"
#include "bayests/vec_to_var.h"
#include "bayests/vec_tvp_discount.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using bayests::MatrixNormalPrior;
using bayests::NullReporter;
using bayests::VarTvpDiscountEstimator;
using bayests::VarTvpDiscountInput;
using bayests::VarTvpDiscountPosterior;
using bayests::VecNormalWishartDraws;
using bayests::VecTvpDiscountEstimator;
using bayests::VecTvpDiscountInput;
using bayests::VecTvpDiscountPosterior;
using bayests::WishartPrior;

namespace
{

int failures = 0;

/// Exact to within an accumulation of rounding over a few thousand operations.
/// The two models run the same recursion over the same numbers, so anything
/// larger is a different design rather than a different arithmetic.
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

void check_equal(const arma::mat &actual, const arma::mat &expected, const std::string &what)
{
    if (actual.n_rows != expected.n_rows || actual.n_cols != expected.n_cols)
    {
        std::printf("FAIL: %s has shape %llu x %llu, expected %llu x %llu\n", what.c_str(),
                    static_cast<unsigned long long>(actual.n_rows),
                    static_cast<unsigned long long>(actual.n_cols),
                    static_cast<unsigned long long>(expected.n_rows),
                    static_cast<unsigned long long>(expected.n_cols));
        failures++;
        return;
    }
    for (arma::uword i = 0; i < actual.n_elem; i++)
    {
        check_close(actual(i), expected(i), what + " at " + std::to_string(i));
    }
}

/// A small deterministic sample: two levels driven by a fixed sequence, so the
/// test depends on no generator. Period i is time i + 2, which leaves the one
/// lagged difference and the error correction term room in front of it.
struct Sample
{
    arma::mat y;      ///< tt x k, the differences the VEC explains.
    arma::mat w;      ///< tt x k_beta, the error correction term.
    arma::mat x;      ///< tt x n_x_vec, one lagged difference per variable.
    arma::mat levels; ///< (tt + 2) x k, what the three above are built from.

    static constexpr arma::uword k = 2;
    static constexpr arma::uword k_beta = 2;
    static constexpr arma::uword rank = 1;
    static constexpr arma::uword tt = 40;
    static constexpr arma::uword n_x = 2; ///< k lagged differences.
};

Sample make_sample()
{
    Sample out;
    out.levels.set_size(Sample::tt + 2, Sample::k);
    for (arma::uword j = 0; j < Sample::k; j++)
    {
        out.levels(0, j) = 0.0;
    }
    for (arma::uword t = 1; t < Sample::tt + 2; t++)
    {
        for (arma::uword j = 0; j < Sample::k; j++)
        {
            const double drive =
                std::sin(0.6 * static_cast<double>(t) + 1.3 * static_cast<double>(j));
            out.levels(t, j) = 0.9 * out.levels(t - 1, j) + drive;
        }
    }

    out.y.set_size(Sample::tt, Sample::k);
    out.w.set_size(Sample::tt, Sample::k_beta);
    out.x.set_size(Sample::tt, Sample::n_x);
    for (arma::uword i = 0; i < Sample::tt; i++)
    {
        out.y.row(i) = out.levels.row(i + 2) - out.levels.row(i + 1);
        out.w.row(i) = out.levels.row(i + 1);
        out.x.row(i) = out.levels.row(i + 1) - out.levels.row(i);
    }
    return out;
}

/// The cointegration matrix the whole file conditions on: one relation, not
/// normalised to anything, because the model does not ask it to be.
arma::mat make_beta()
{
    arma::mat beta(Sample::k_beta, Sample::rank);
    beta(0, 0) = 1.0;
    beta(1, 0) = -0.8;
    return beta;
}

/// The design the VEC assembles, written out one element at a time.
///
/// Deliberately not `w * beta`: that is what the model computes, and a test
/// that recomputed it the same way would agree with a transposed beta as
/// readily as with the right one.
arma::mat make_design(const Sample &sample, const arma::mat &beta)
{
    arma::mat design(Sample::tt, Sample::rank + Sample::n_x, arma::fill::zeros);
    for (arma::uword t = 0; t < Sample::tt; t++)
    {
        for (arma::uword j = 0; j < Sample::rank; j++)
        {
            double value = 0.0;
            for (arma::uword l = 0; l < Sample::k_beta; l++)
            {
                value += beta(l, j) * sample.w(t, l);
            }
            design(t, j) = value;
        }
        for (arma::uword j = 0; j < Sample::n_x; j++)
        {
            design(t, Sample::rank + j) = sample.x(t, j);
        }
    }
    return design;
}

void fill_spec(bayests::VarSpec &spec)
{
    spec.k = static_cast<int>(Sample::k);
    spec.p = 2; ///< A level order, so one Gamma block.
    spec.n = 0;
    spec.rank = static_cast<int>(Sample::rank);
    spec.k_beta = static_cast<int>(Sample::k_beta);
}

VecTvpDiscountInput make_input(const Sample &sample, const arma::mat &beta,
                               const double delta_beta, const double delta_sigma)
{
    VecTvpDiscountInput input;
    fill_spec(input.spec);
    input.spec.delta_beta = delta_beta;
    input.spec.delta_sigma = delta_sigma;

    // Stored the way the files store it: vec(y'), period-major.
    input.train.y = arma::vectorise(sample.y.t());
    input.train.w = sample.w;
    input.train.x = sample.x;
    input.beta = beta;

    const arma::uword n_design = Sample::rank + Sample::n_x;
    input.a_prior.mean.zeros(n_design, Sample::k);
    input.a_prior.cov = 0.5 * arma::eye<arma::mat>(n_design, n_design);
    input.u_sigma_prior.df = static_cast<int>(Sample::k) + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(Sample::k, Sample::k);
    return input;
}

/// The same model stated as a VAR over the assembled design.
VarTvpDiscountInput as_var_input(const VecTvpDiscountInput &vec_input, const arma::mat &design)
{
    VarTvpDiscountInput input;
    input.spec.k = vec_input.spec.k;
    input.spec.p = 0; ///< Nothing here forecasts, so no lag blocks are walked.
    input.spec.delta_beta = vec_input.spec.delta_beta;
    input.spec.delta_sigma = vec_input.spec.delta_sigma;

    input.train.y = vec_input.train.y;
    input.train.x = design;
    input.a_prior = vec_input.a_prior;
    input.u_sigma_prior = vec_input.u_sigma_prior;
    return input;
}

void test_is_a_var_over_the_assembled_design()
{
    const Sample sample = make_sample();
    const arma::mat beta = make_beta();
    const VecTvpDiscountInput vec_input = make_input(sample, beta, 0.98, 0.96);
    const VarTvpDiscountInput var_input = as_var_input(vec_input, make_design(sample, beta));

    NullReporter reporter;
    const VecTvpDiscountPosterior vec = VecTvpDiscountEstimator{}.estimate(vec_input, reporter);
    const VarTvpDiscountPosterior var = VarTvpDiscountEstimator{}.estimate(var_input, reporter);

    check_equal(vec.a, var.a, "coefficient mean");
    check_equal(vec.a_scale, var.a_scale, "coefficient scale");
    check_equal(vec.a_cov, var.a_cov, "coefficient covariance");
    check_equal(vec.u_sigma, var.u_sigma, "error covariance");
    check_equal(arma::mat(vec.df), arma::mat(var.df), "degrees of freedom");
    check_equal(arma::mat(vec.loglik), arma::mat(var.loglik), "pointwise log likelihood");
    check_equal(vec.forecast_mean, var.forecast_mean, "one step ahead mean");

    // And the space it conditioned on is stored, without which the loadings
    // would be numbers with no relation attached.
    check(vec.has_beta(), "the posterior carries the cointegration matrix");
    check_equal(arma::mat(vec.beta), arma::mat(arma::vectorise(beta)), "stored beta");
}

void test_rank_zero_is_the_var_in_differences()
{
    const Sample sample = make_sample();
    VecTvpDiscountInput vec_input = make_input(sample, arma::mat(), 0.97, 0.95);
    vec_input.spec.rank = 0;
    vec_input.spec.k_beta = 0;
    vec_input.train.w.reset();
    vec_input.beta.reset();

    const arma::uword n_design = Sample::n_x;
    vec_input.a_prior.mean.zeros(n_design, Sample::k);
    vec_input.a_prior.cov = 0.5 * arma::eye<arma::mat>(n_design, n_design);

    const VarTvpDiscountInput var_input = as_var_input(vec_input, sample.x);

    NullReporter reporter;
    const VecTvpDiscountPosterior vec = VecTvpDiscountEstimator{}.estimate(vec_input, reporter);
    const VarTvpDiscountPosterior var = VarTvpDiscountEstimator{}.estimate(var_input, reporter);

    check_equal(vec.a, var.a, "rank zero coefficient mean");
    check_equal(arma::mat(vec.loglik), arma::mat(var.loglik), "rank zero log likelihood");
    check(!vec.has_beta(), "a rank zero posterior carries no cointegration matrix");
}

void test_coefficients_convert_to_the_level_var()
{
    const Sample sample = make_sample();
    const arma::mat beta = make_beta();
    const VecTvpDiscountInput input = make_input(sample, beta, 0.98, 0.96);

    NullReporter reporter;
    const VecTvpDiscountPosterior posterior =
        VecTvpDiscountEstimator{}.estimate(input, reporter);

    const arma::uword last = posterior.periods() - 1;
    const arma::uword n_design = Sample::rank + Sample::n_x;

    // `a` is vec(B) with B of k x n_design, so its leading `rank` columns are
    // the loadings and the k after them are Gamma_1. That is the claim.
    const arma::mat b = arma::reshape(posterior.a.col(last), Sample::k, n_design);
    const arma::mat alpha = b.cols(0, Sample::rank - 1);
    const arma::mat gamma1 = b.cols(Sample::rank, Sample::rank + Sample::k - 1);

    VecNormalWishartDraws period;
    period.a = posterior.a.col(last);
    period.beta = posterior.beta;
    period.u_sigma_inv =
        arma::vectorise(arma::inv_sympd(arma::symmatu(
            arma::reshape(posterior.u_sigma.col(last), Sample::k, Sample::k))));

    const bayests::VarNormalWishartDraws level =
        bayests::vec_to_var_coefficients(input.spec, period);

    const arma::mat blocks =
        arma::reshape(level.a.col(0), Sample::k, level.a.n_rows / Sample::k);

    const arma::mat expected_a1 =
        arma::eye<arma::mat>(Sample::k, Sample::k) + alpha * beta.t() + gamma1;
    const arma::mat expected_a2 = -gamma1;

    check_equal(arma::mat(blocks.cols(0, Sample::k - 1)), expected_a1, "level A_1");
    check_equal(arma::mat(blocks.cols(Sample::k, 2 * Sample::k - 1)), expected_a2, "level A_2");
}

void test_consumes_no_randomness()
{
    const Sample sample = make_sample();
    const VecTvpDiscountInput input = make_input(sample, make_beta(), 0.97, 0.96);

    NullReporter reporter;
    const VecTvpDiscountEstimator estimator;

    arma::arma_rng::set_seed(1);
    const VecTvpDiscountPosterior first = estimator.estimate(input, reporter);
    arma::arma_rng::set_seed(2);
    const VecTvpDiscountPosterior second = estimator.estimate(input, reporter);

    check(arma::approx_equal(first.a, second.a, "absdiff", 0.0),
          "estimate() is bit-identical under two different seeds");
}

/// The forecast and the score run through the shared level machinery, which
/// unit_predictive_score.cpp covers for the samplers. What is checked here is
/// that this model reaches it with the shapes it promises.
void test_forecast_and_score_shapes()
{
    const Sample sample = make_sample();
    const arma::mat beta = make_beta();
    VecTvpDiscountInput input = make_input(sample, beta, 0.98, 0.96);

    const arma::uword h = 3;
    const arma::uword draws = 16;
    input.spec.h = static_cast<int>(h);

    // The level layout: k lags of k variables, as vec_to_var_spec() counts them.
    input.forecast.x.set_size(h, Sample::k * 2);
    for (arma::uword i = 0; i < h; i++)
    {
        input.forecast.x.submat(i, 0, i, Sample::k - 1) = sample.levels.row(Sample::tt + 1);
        input.forecast.x.submat(i, Sample::k, i, 2 * Sample::k - 1) =
            sample.levels.row(Sample::tt);
    }

    NullReporter reporter;
    const VecTvpDiscountEstimator estimator;
    const VecTvpDiscountPosterior posterior = estimator.estimate(input, reporter);

    check(estimator.log_likelihood(input, posterior).n_rows == 1,
          "the pointwise log likelihood has one row, the parameters being integrated out");
    check(estimator.log_likelihood(input, posterior).n_cols == Sample::tt,
          "the pointwise log likelihood has one column per period");

    arma::arma_rng::set_seed(42);
    const arma::mat forecast = estimator.forecast(input, posterior, draws, reporter).values;
    check(forecast.n_rows == h * Sample::k && forecast.n_cols == draws,
          "the forecast is (h k) x draws");
    check(forecast.is_finite(), "the forecast is finite");

    input.test.y = sample.levels.rows(Sample::tt, Sample::tt + 1);
    const arma::mat score = estimator.predictive_log_density(input, posterior, draws);
    check(score.n_rows == draws && score.n_cols == 2, "the score is draws x scored periods");
    check(score.is_finite(), "the score is finite");
}

void test_refusals()
{
    const Sample sample = make_sample();
    const arma::mat beta = make_beta();

    const auto throws = [](const VecTvpDiscountInput &input) {
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

    const VecTvpDiscountInput ok = make_input(sample, beta, 0.98, 0.96);
    check(!throws(ok), "a well specified model is accepted");

    VecTvpDiscountInput no_beta = ok;
    no_beta.beta.reset();
    check(throws(no_beta), "a model with a rank and no cointegration matrix is refused");

    VecTvpDiscountInput transposed = ok;
    transposed.beta = beta.t();
    check(throws(transposed), "a cointegration matrix of the wrong shape is refused");

    VecTvpDiscountInput no_w = ok;
    no_w.train.w.reset();
    check(throws(no_w), "a missing error correction term is refused");

    VecTvpDiscountInput sur = ok;
    sur.train.x.reset();
    check(throws(sur), "the SUR layout alone is refused");

    VecTvpDiscountInput zero_discount = ok;
    zero_discount.spec.delta_beta = 0.0;
    check(throws(zero_discount), "a discount of zero is refused");

    VecTvpDiscountInput above_one = ok;
    above_one.spec.delta_sigma = 1.5;
    check(throws(above_one), "a discount above one is refused");

    VecTvpDiscountInput structural = ok;
    structural.spec.structural = true;
    check(throws(structural), "a structural model is refused");

    VecTvpDiscountInput selection = ok;
    selection.spec.varsel = bayests::VarSelection::bvs;
    check(throws(selection), "variable selection is refused");

    // The degrees of freedom settle at 1 / (1 - delta_sigma), which has to clear
    // the width of the system.
    VecTvpDiscountInput short_memory = ok;
    short_memory.spec.delta_sigma = 0.4;
    check(throws(short_memory), "a memory too short for the system is refused");
}

} // namespace

int main()
{
    test_is_a_var_over_the_assembled_design();
    test_rank_zero_is_the_var_in_differences();
    test_coefficients_convert_to_the_level_var();
    test_consumes_no_randomness();
    test_forecast_and_score_shapes();
    test_refusals();

    if (failures == 0)
    {
        std::printf("unit_vec_tvp_discount: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
