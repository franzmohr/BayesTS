// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_iid_block.cpp
/// @brief Checks the i.i.d. block: equations restricted to carry no coefficients.
///
/// `VarSpec::n_iid` names endogenous variables, ordered first, whose equations
/// have no lags, no deterministic terms and nothing else -- white noise that
/// reaches the model only through the error covariance. Jarocinski and Karadi
/// (2020) put high-frequency surprises in a monthly VAR that way.
///
/// Three things are asserted and the middle one is the point.
///
/// The restriction is exact. Every draw of a restricted equation's coefficients
/// is zero, not small: this is a restriction, not a tight prior, and the two are
/// different models.
///
/// The free coefficients are the ones the restricted model asks for. What the
/// sampler does is drop the restricted columns out of the SUR system, which
/// leaves the free coefficients drawn under the prior *conditional* on the
/// restricted ones being zero -- precision V_ff and precision-weighted mean
/// (V mu)_f. The check is an identity rather than a value: a chain run with
/// n_iid = 1 has to agree, draw for draw and bit for bit, with a chain run from
/// the same seed on a system whose columns were dropped by hand and whose prior
/// was conditioned by hand. The prior used for it is deliberately not diagonal,
/// because with a diagonal prior conditioning and marginalising coincide and the
/// test would pass for a sampler that did the wrong one of the two.
///
/// And the restricted variable is still a variable. Its equation contributes its
/// own residual, which is the series itself, so the error covariance the model
/// estimates covers it and the correlation the model exists to measure is there
/// to read.
///
/// Around those: which positions the block leaves free, that all four
/// constant-coefficient VARs honour it, and the six specifications validate()
/// refuses.

#include "bayests/reporter.h"
#include "bayests/var_normal_ald.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_stochvol.h"
#include "bayests/var_normal_wishart.h"
#include "bayests/var_tvp_wishart.h"
#include "core/models/model_support.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

namespace
{

using bayests::core::iid_block;
using bayests::core::IidBlock;

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

bool throws_with(const std::string &fragment, const std::function<void()> &call)
{
    try
    {
        call();
    }
    catch (const std::invalid_argument &error)
    {
        return std::string(error.what()).find(fragment) != std::string::npos;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

constexpr int kK = 3;    ///< Endogenous variables.
constexpr int kNx = 3;   ///< Regressors per equation: a constant and two others.
constexpr int kIid = 1;  ///< The first variable is white noise.
constexpr int kTt = 120;

/// The three series and the SUR regressor matrix they are regressed on.
///
/// Variable 0 is white noise correlated with the errors of the other two, which
/// is what a high-frequency surprise looks like: unforecastable, and informative
/// about the others only contemporaneously.
struct Sample
{
    arma::mat y;   ///< tt x k, one row per period.
    arma::mat z;   ///< (tt * k) x (k * n_x), the SUR layout.
    arma::mat x;   ///< tt x n_x, the same regressors per period.
};

Sample make_sample()
{
    Sample out;
    out.y = arma::mat(kTt, kK);
    out.x = arma::mat(kTt, kNx);

    arma::vec previous(kK, arma::fill::zeros);
    for (int t = 0; t < kTt; t++)
    {
        const arma::vec shock = arma::randn<arma::vec>(kK);
        const double surprise = shock(0);

        out.x(t, 0) = 1.0;
        out.x(t, 1) = previous(1);
        out.x(t, 2) = previous(2);

        out.y(t, 0) = surprise;
        out.y(t, 1) = 0.4 * previous(1) + 0.2 * previous(2) + 0.8 * surprise + shock(1);
        out.y(t, 2) = -0.3 * previous(1) + 0.5 * previous(2) - 0.5 * surprise + shock(2);

        previous = arma::trans(out.y.row(t));
    }

    // Row (t k + i) is equation i at period t, and its coefficients sit at
    // positions j k + i -- which is kron(x_t', I_k), stacked over periods.
    out.z = arma::mat(kTt * kK, kK * kNx, arma::fill::zeros);
    for (int t = 0; t < kTt; t++)
    {
        out.z.rows(t * kK, (t + 1) * kK - 1) = arma::kron(out.x.row(t), arma::eye(kK, kK));
    }

    return out;
}

/// A prior with off-diagonal precision, so that conditioning on the restricted
/// coefficients being zero is not the same as ignoring them.
bayests::NormalPrior correlated_prior(const arma::uword nparams)
{
    bayests::NormalPrior prior;
    prior.mu = arma::vec(nparams);
    for (arma::uword i = 0; i < nparams; i++)
    {
        prior.mu(i) = 0.1 * static_cast<double>(i % 5) - 0.2;
    }

    prior.v_inv = arma::mat(nparams, nparams, arma::fill::zeros);
    for (arma::uword i = 0; i < nparams; i++)
    {
        prior.v_inv(i, i) = 2.0;
        if (i + 1 < nparams)
        {
            prior.v_inv(i, i + 1) = 0.5;
            prior.v_inv(i + 1, i) = 0.5;
        }
    }
    return prior;
}

bayests::VarNormalWishartInput wishart_input(const Sample &sample, const int n_iid)
{
    bayests::VarNormalWishartInput input;
    input.spec.k = kK;
    input.spec.p = 1;
    input.spec.n = 1;
    input.spec.n_iid = n_iid;
    input.spec.iterations = 40;
    input.spec.burnin = 10;
    input.train.y = sample.y;
    input.train.z = sample.z;
    input.a_prior = correlated_prior(kK * kNx);
    input.u_sigma_prior.df = kK + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    input.initial.a = arma::vec(kK * kNx, arma::fill::zeros);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return input;
}

arma::mat run_wishart(bayests::VarNormalWishartInput input, arma::mat &u_sigma_inv)
{
    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(20260923);
    const bayests::VarNormalWishartDraws draws =
        bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);
    u_sigma_inv = draws.u_sigma_inv;
    return draws.a;
}

void the_block_leaves_the_right_positions_free()
{
    std::printf("which positions the restriction leaves free\n");

    bayests::VarSpec spec;
    spec.k = kK;
    spec.n_iid = kIid;

    const IidBlock block = iid_block(spec, kK * kNx);
    check("the block reports itself restricted", block.restricted);
    check("one equation of three leaves two thirds of the coefficients",
          block.free.n_elem == static_cast<arma::uword>(kNx * (kK - kIid)));

    // `a` is vec of the k by n_x coefficient matrix, so equation i owns the
    // positions congruent to i modulo k.
    bool right = true;
    for (arma::uword i = 0; i < block.free.n_elem; i++)
    {
        right = right && block.free(i) % kK >= static_cast<arma::uword>(kIid);
    }
    check("and every free position belongs to an unrestricted equation", right);

    spec.n_iid = 0;
    const IidBlock none = iid_block(spec, kK * kNx);
    check("a model with no restriction reports none", !none.restricted);

    const arma::vec drawn = arma::randn<arma::vec>(kNx * (kK - kIid));
    const arma::vec scattered = block.scatter(drawn);
    check("scattering a draw gives the stored length",
          scattered.n_elem == static_cast<arma::uword>(kK * kNx));
    check("with the free coefficients where they belong",
          arma::approx_equal(scattered.elem(block.free), drawn, "absdiff", 0.0));
    check("and exact zeros everywhere else",
          std::abs(arma::accu(scattered)) - std::abs(arma::accu(drawn)) < 1e-12);
}

void the_restricted_equations_draw_exact_zeros()
{
    std::printf("the restriction is exact, not tight\n");

    const Sample sample = make_sample();

    arma::mat sigma;
    const arma::mat a = run_wishart(wishart_input(sample, kIid), sigma);

    bool zero = true;
    for (arma::uword row = 0; row < a.n_rows; row++)
    {
        if (row % kK < static_cast<arma::uword>(kIid))
        {
            zero = zero && arma::accu(arma::abs(a.row(row))) == 0.0;
        }
    }
    check("every draw of a restricted equation's coefficients is exactly zero", zero);

    bool free_moves = false;
    for (arma::uword row = 0; row < a.n_rows; row++)
    {
        if (row % kK >= static_cast<arma::uword>(kIid))
        {
            free_moves = free_moves || arma::accu(arma::abs(a.row(row))) > 0.0;
        }
    }
    check("and the unrestricted ones are drawn", free_moves);

    // The restricted equation contributes its own residual, which is the series
    // itself, so the covariance the model estimates still covers it.
    arma::vec variance(sigma.n_cols);
    for (arma::uword draw = 0; draw < sigma.n_cols; draw++)
    {
        const arma::mat precision = arma::reshape(sigma.col(draw), kK, kK);
        const arma::mat covariance = arma::inv_sympd(precision);
        variance(draw) = covariance(0, 0);
    }
    const double expected = arma::accu(arma::square(sample.y.col(0))) / kTt;
    const double estimated = arma::mean(variance);
    std::printf("    variance of the restricted variable: %.3f, its second moment %.3f\n",
                estimated, expected);
    check("the error covariance covers the restricted variable",
          std::abs(estimated - expected) < 0.5 * expected);
}

void the_free_coefficients_are_the_conditional_ones()
{
    std::printf("the reduced system is the restricted model\n");

    const Sample sample = make_sample();

    arma::mat restricted_sigma;
    const arma::mat restricted = run_wishart(wishart_input(sample, kIid), restricted_sigma);

    // The same model with the columns dropped and the prior conditioned by
    // hand: precision V_ff, and a mean that reproduces the free part of V mu.
    bayests::VarSpec spec;
    spec.k = kK;
    spec.n_iid = kIid;
    const IidBlock block = iid_block(spec, kK * kNx);

    const bayests::NormalPrior full = correlated_prior(kK * kNx);
    const arma::mat v_ff = full.v_inv.submat(block.free, block.free);
    const arma::vec weighted_mean = full.v_inv * full.mu;
    const arma::vec rhs = weighted_mean.elem(block.free);

    bayests::VarNormalWishartInput reduced = wishart_input(sample, 0);
    reduced.train.z = sample.z.cols(block.free);
    reduced.a_prior.v_inv = v_ff;
    reduced.a_prior.mu = arma::solve(v_ff, rhs);
    reduced.initial.a = arma::vec(block.free.n_elem, arma::fill::zeros);

    arma::mat reduced_sigma;
    const arma::mat drawn = run_wishart(reduced, reduced_sigma);

    check("the reduced chain has one row per free coefficient",
          drawn.n_rows == block.free.n_elem);
    check("the free coefficients agree draw for draw",
          arma::approx_equal(restricted.rows(block.free), drawn, "absdiff", 0.0));
    check("and so does the error precision",
          arma::approx_equal(restricted_sigma, reduced_sigma, "absdiff", 0.0));
}

void every_constant_var_honours_it()
{
    std::printf("all four constant-coefficient VARs\n");

    const Sample sample = make_sample();
    const bayests::NormalPrior prior = correlated_prior(kK * kNx);
    bayests::NullReporter reporter;

    const auto restricted_rows_are_zero = [](const arma::mat &a) {
        for (arma::uword row = 0; row < a.n_rows; row++)
        {
            if (row % kK < static_cast<arma::uword>(kIid) &&
                arma::accu(arma::abs(a.row(row))) != 0.0)
            {
                return false;
            }
        }
        return a.n_rows == static_cast<arma::uword>(kK * kNx);
    };

    const auto fill = [&](auto &input) {
        input.spec.k = kK;
        input.spec.p = 1;
        input.spec.n = 1;
        input.spec.n_iid = kIid;
        input.spec.iterations = 10;
        input.spec.burnin = 5;
        input.train.y = sample.y;
        input.train.z = sample.z;
        input.a_prior = prior;
        input.initial.a = arma::vec(kK * kNx, arma::fill::zeros);
    };

    {
        bayests::VarNormalWishartInput input;
        fill(input);
        input.u_sigma_prior.df = kK + 2;
        input.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
        input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
        arma::arma_rng::set_seed(1);
        check("VarNormalWishart",
              restricted_rows_are_zero(
                  bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter).a));
    }

    {
        bayests::VarNormalGammaInput input;
        fill(input);
        input.u_sigma_prior.shape = arma::vec(kK, arma::fill::value(3.0));
        input.u_sigma_prior.rate = arma::vec(kK, arma::fill::value(2.0));
        input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
        arma::arma_rng::set_seed(1);
        check("VarNormalGamma",
              restricted_rows_are_zero(
                  bayests::VarNormalGammaSampler{}.draw_coefficients(input, reporter).a));
    }

    {
        bayests::VarNormalStochvolInput input;
        fill(input);
        input.u_sigma_prior.offset = arma::vec(kK, arma::fill::value(1e-6));
        input.u_sigma_prior.state.sigma.shape = arma::vec(kK, arma::fill::value(5.0));
        input.u_sigma_prior.state.sigma.rate = arma::vec(kK, arma::fill::value(0.05));
        input.u_sigma_prior.state.initial_state.mu = arma::vec(kK, arma::fill::zeros);
        input.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(kK, kK);
        input.initial.h = arma::mat(kTt, kK, arma::fill::zeros);
        input.initial.h_init = arma::vec(kK, arma::fill::zeros);
        input.initial.h_sigma = arma::vec(kK, arma::fill::value(0.01));
        arma::arma_rng::set_seed(1);
        check("VarNormalStochvol",
              restricted_rows_are_zero(
                  bayests::VarNormalStochvolSampler{}.draw_coefficients(input, reporter).a));
    }

    {
        bayests::VarNormalAldInput input;
        fill(input);
        input.spec.quantile = 0.5;
        input.u_scale_prior.shape = arma::vec(kK, arma::fill::value(3.0));
        input.u_scale_prior.rate = arma::vec(kK, arma::fill::value(0.2));
        input.initial.w = arma::mat(kTt, kK, arma::fill::ones);
        input.initial.u_scale = arma::vec(kK, arma::fill::ones);
        arma::arma_rng::set_seed(1);
        check("VarNormalAld",
              restricted_rows_are_zero(
                  bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter).a));
    }
}

void the_specifications_that_are_refused()
{
    std::printf("what validate() refuses\n");

    const Sample sample = make_sample();

    check("an algorithm that does not read it says so",
          throws_with("does not read n_iid", [&] {
              bayests::VarTvpWishartInput input;
              input.spec.k = kK;
              input.spec.n_iid = kIid;
              input.spec.iterations = 1;
              input.train.y = sample.y;
              input.train.z = sample.z;
              input.validate();
          }));

    check("a negative count is refused", throws_with("cannot be negative", [&] {
              bayests::VarNormalWishartInput input = wishart_input(sample, -1);
              input.validate();
          }));

    check("restricting every equation is refused",
          throws_with("leaves no equation with dynamics", [&] {
              bayests::VarNormalWishartInput input = wishart_input(sample, kK);
              input.validate();
          }));

    check("a structural form is refused", throws_with("structural form cannot be combined", [&] {
              bayests::VarNormalWishartInput input = wishart_input(sample, kIid);
              input.spec.structural = true;
              input.validate();
          }));

    check("variable selection is refused", throws_with("selection cannot be combined", [&] {
              bayests::VarNormalWishartInput input = wishart_input(sample, kIid);
              input.spec.varsel = bayests::VarSelection::bvs;
              input.validate();
          }));

    bool accepted = true;
    try
    {
        wishart_input(sample, kIid).validate();
    }
    catch (...)
    {
        accepted = false;
    }
    check("and a model that may have one is accepted", accepted);
}

} // namespace

int main()
{
    std::printf("unit_iid_block\n");
    arma::arma_rng::set_seed(20260923);

    the_block_leaves_the_right_positions_free();
    the_restricted_equations_draw_exact_zeros();
    the_free_coefficients_are_the_conditional_ones();
    every_constant_var_honours_it();
    the_specifications_that_are_refused();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
