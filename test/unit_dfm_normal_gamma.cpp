// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That DfmNormalGamma draws the model it says it does.
//
// A dynamic factor model has no closed form to check a draw against and no
// second implementation here to compare with, so this covers it from three
// directions, in decreasing order of how exact the check is:
//
//   1. The conventions, exactly. Which elements of Lambda are free, where the
//      lagged factors go, what the transition residual is, what the prior over
//      the first p factors comes to. Every one of these is a place where a
//      plausible off-by-one still runs and quietly estimates a different model
//      -- bvartools' dfmpost() has two of them, and both are noted below.
//   2. The factor block against its own definition. The path's posterior is a
//      Gaussian whose precision this test builds densely, the way the reference
//      implementation does, from which the mean and covariance follow by
//      inversion. The sampler reaches the same distribution through a banded
//      sweep over a prior-plus-transitions decomposition that looks nothing like
//      it, so agreement is evidence about the mapping and not a tautology.
//   3. Recovery. A sample simulated from known parameters, and a chain that has
//      to find them. The weakest check of the three and the only end-to-end one.
//
// Plus what validate() has to refuse. No fixture and no file: everything is
// built in memory.

#include "bayests/dfm_normal_gamma.h"
#include "core/models/dfm_support.h"

#include <cstdio>
#include <string>

namespace
{

using bayests::core::draw_factor_path;
using bayests::core::fill_lagged_factors;
using bayests::core::fill_lambda;
using bayests::core::identified_loadings;
using bayests::core::transition_residuals;

int failures = 0;

void check(const std::string &what, const bool ok, const std::string &detail = "")
{
    failures += ok ? 0 : 1;
    std::printf("  %-52s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!detail.empty())
    {
        std::printf("      %s\n", detail.c_str());
    }
}

void check_below(const std::string &what, const double got, const double bound)
{
    check(what, got <= bound,
          "got " + std::to_string(got) + ", allowed " + std::to_string(bound));
}

double max_abs_diff(const arma::mat &lhs, const arma::mat &rhs)
{
    if (lhs.n_rows != rhs.n_rows || lhs.n_cols != rhs.n_cols)
    {
        return arma::datum::inf;
    }
    return lhs.n_elem == 0 ? 0.0 : arma::abs(lhs - rhs).max();
}

//////////////////////////////////////////////////////////////////////////////
// 1. The conventions.

/// Lambda's leading N x N block is the identification and is never drawn; the
/// free elements fill in row by row, min(i, N) of them in row i.
void the_loading_matrix_is_identified()
{
    std::printf("the loading matrix carries its identification\n");

    const int k = 5, n = 3;
    arma::mat lambda = identified_loadings(k, n);

    // 3 + 3 + 3 = ... no: rows 1, 2, 3, 4 contribute 1, 2, 3, 3.
    const arma::vec free = arma::regspace<arma::vec>(1, 9);
    fill_lambda(lambda, free);

    check("the free elements are N(2M - N - 1)/2", free.n_elem == 9);

    arma::mat want(k, n, arma::fill::zeros);
    want.diag().ones();
    want(1, 0) = 1;                               // row 1: one free element
    want(2, 0) = 2; want(2, 1) = 3;               // row 2: two
    want(3, 0) = 4; want(3, 1) = 5; want(3, 2) = 6;  // row 3: three, the row is full
    want(4, 0) = 7; want(4, 1) = 8; want(4, 2) = 9;  // row 4: three
    check_below("the free elements fill in row by row", max_abs_diff(lambda, want), 0.0);

    check("the diagonal of the identifying block stays one",
          arma::approx_equal(lambda.submat(0, 0, n - 1, n - 1).diag(),
                             arma::vec(n, arma::fill::ones), "absdiff", 0.0));
    check("nothing lands above that diagonal",
          std::abs(lambda(0, 1)) + std::abs(lambda(0, 2)) + std::abs(lambda(1, 2)) == 0.0);
}

/// Block j of the transition's regressors is f_{t-j} and occupies rows
/// (j-1)N .. jN-1, with zeros before the sample. bvartools' dfmpost() writes
/// rows (j-1) .. (j-1)+N-1, which overlaps the blocks for N > 1 -- so this is
/// the check that separates the two, and it needs more than one factor to do it.
void the_lagged_factors_land_in_their_own_block()
{
    std::printf("the transition's regressors are the lagged factors\n");

    const int n = 2, p = 2, tt = 5;
    arma::mat factors(n, tt);
    for (int t = 0; t < tt; t++)
    {
        factors(0, t) = 10 + t; // 10 11 12 13 14
        factors(1, t) = 20 + t; // 20 21 22 23 24
    }

    arma::mat x_a(n * p, tt);
    fill_lagged_factors(x_a, factors, n, p);

    arma::mat want(n * p, tt, arma::fill::zeros);
    // Lag one in rows 0..1, lag two in rows 2..3, both shifted right.
    want.submat(0, 1, 1, tt - 1) = factors.cols(0, tt - 2);
    want.submat(2, 2, 3, tt - 1) = factors.cols(0, tt - 3);
    check_below("lag j occupies rows (j-1)N .. jN-1", max_abs_diff(x_a, want), 0.0);

    check("the columns before the sample are zero",
          x_a.col(0).is_zero() && x_a(2, 1) == 0.0 && x_a(3, 1) == 0.0);
    check("and the first lag is already there in period one",
          x_a(0, 1) == 10.0 && x_a(1, 1) == 20.0);

    // The residual is the same convention read the other way round.
    arma::mat a_mat(n, n * p, arma::fill::zeros);
    a_mat(0, 0) = 0.5; // A_1
    a_mat(1, 1) = 0.5;
    a_mat(0, 2) = 0.25; // A_2
    a_mat(1, 3) = 0.25;

    const arma::mat v = transition_residuals(factors, a_mat, n, p);
    check_below("the residual is f_t - sum_j A_j f_{t-j}",
                max_abs_diff(v, factors - a_mat * x_a), 0.0);
    check("period one has no lag to subtract",
          v(0, 0) == factors(0, 0) && v(1, 0) == factors(1, 0));
}

/// The factors before the sample are zero, so the first p of them are the
/// transition run from nothing, and the covariance that implies is what the
/// banded sampler is handed as its prior.
void the_prior_over_the_first_factors_is_the_transition()
{
    std::printf("the prior over the first p factors\n");

    const int n = 2;
    const arma::mat v_sigma = arma::diagmat(arma::vec{2.0, 0.5});

    // p = 1: f_1 = v_1, so the prior is V itself.
    const arma::mat one = bayests::core::initial_state_covariance(
        arma::zeros<arma::mat>(n, n), v_sigma, n, 1);
    check_below("p = 1 gives V", max_abs_diff(one, v_sigma), 1e-14);

    // p = 2: f_1 = v_1 and f_2 = A_1 v_1 + v_2, hence
    //   [[V, V A_1'], [A_1 V, A_1 V A_1' + V]].
    const arma::mat a1 = {{0.6, 0.2}, {-0.1, 0.4}};
    const arma::mat two = bayests::core::initial_state_covariance(a1, v_sigma, n, 2);

    arma::mat want(2 * n, 2 * n);
    want.submat(0, 0, n - 1, n - 1) = v_sigma;
    want.submat(0, n, n - 1, 2 * n - 1) = v_sigma * a1.t();
    want.submat(n, 0, 2 * n - 1, n - 1) = a1 * v_sigma;
    want.submat(n, n, 2 * n - 1, 2 * n - 1) = a1 * v_sigma * a1.t() + v_sigma;
    check_below("p = 2 gives the MA(1) covariance of the pair",
                max_abs_diff(two, want), 1e-13);
}

//////////////////////////////////////////////////////////////////////////////
// 2. The factor block against the dense posterior.

struct DensePosterior
{
    arma::vec mean;
    arma::mat cov;
};

/// The factor path's posterior, formed the way bvartools' dfmpost() forms it:
/// one (tt N) square precision, inverted.
///
///   K = H' (I kron V^-1) H + I kron (Lambda' U^-1 Lambda),
///   b = (I kron Lambda' U^-1) vec(X),
///
/// with H unit block lower triangular carrying -A_j on its j-th subdiagonal,
/// truncated at the top because the factors before the sample are zero. That
/// truncation is the whole of what the sampler expresses as a prior over the
/// first p states, so if the two agree, the decomposition is right.
DensePosterior dense_factor_posterior(const arma::mat &x_t, const arma::mat &lambda,
                                      const arma::mat &u_prec, const arma::mat &v_prec,
                                      const arma::mat &a_mat, const int n, const int p)
{
    const int tt = static_cast<int>(x_t.n_cols);
    const arma::mat eye_tt = arma::eye<arma::mat>(tt, tt);

    arma::mat h = arma::eye<arma::mat>(tt * n, tt * n);
    for (int i = 0; i < tt; i++)
    {
        for (int j = 1; j <= p && j <= i; j++)
        {
            h.submat(i * n, (i - j) * n, (i + 1) * n - 1, (i - j + 1) * n - 1) =
                -a_mat.cols((j - 1) * n, j * n - 1);
        }
    }

    const arma::mat precision = arma::symmatu(
        h.t() * arma::kron(eye_tt, v_prec) * h +
        arma::kron(eye_tt, lambda.t() * u_prec * lambda));
    const arma::vec rhs = arma::kron(eye_tt, lambda.t() * u_prec) * arma::vectorise(x_t);

    DensePosterior out;
    out.cov = arma::inv_sympd(precision);
    out.mean = out.cov * rhs;
    return out;
}

/// Draws the path many times and compares the first two moments with the dense
/// posterior above. Run at three transition orders, zero among them: a model
/// whose factors have no dynamics goes through the same code with a zero
/// transition, and that shortcut is worth a check of its own.
void the_factor_path_is_the_posterior_it_should_be()
{
    std::printf("the factor path matches the dense posterior\n");

    const int k = 4, n = 2, tt = 6;
    const int reps = 30000;

    arma::arma_rng::set_seed(20260827);

    arma::mat lambda = identified_loadings(k, n);
    fill_lambda(lambda, arma::vec{0.7, -0.4, 0.9, 0.3, -0.8});
    const arma::mat x_t = arma::randn<arma::mat>(k, tt);

    const arma::vec u_sigma_inv{4.0, 2.5, 6.0, 3.0};
    const arma::vec v_sigma_inv{1.5, 0.8};
    const arma::mat u_sigma = arma::diagmat(1.0 / u_sigma_inv);
    const arma::mat v_sigma = arma::diagmat(1.0 / v_sigma_inv);

    for (const int p : {0, 1, 2})
    {
        arma::mat a_mat;
        if (p == 0)
        {
            // What the sampler passes: a zero transition, which is the serially
            // independent factor written as a transition.
            a_mat = arma::zeros<arma::mat>(n, n);
        }
        else
        {
            a_mat = arma::mat(n, n * p, arma::fill::zeros);
            a_mat(0, 0) = 0.5;
            a_mat(1, 0) = 0.15;
            a_mat(1, 1) = 0.35;
            if (p == 2)
            {
                a_mat(0, 2) = -0.2;
                a_mat(1, 3) = 0.25;
            }
        }

        const DensePosterior want =
            dense_factor_posterior(x_t, lambda, arma::diagmat(u_sigma_inv),
                                   arma::diagmat(v_sigma_inv), a_mat, n, p);

        arma::mat sample(n * tt, reps);
        for (int r = 0; r < reps; r++)
        {
            sample.col(r) = arma::vectorise(
                draw_factor_path(x_t, lambda, u_sigma, v_sigma, a_mat, n, p > 0 ? p : 1));
        }

        const arma::vec got_mean = arma::mean(sample, 1);
        const arma::mat got_cov = arma::cov(sample.t());

        const std::string at = "p = " + std::to_string(p);
        check_below(at + ", the posterior mean", max_abs_diff(got_mean, want.mean), 0.03);
        check_below(at + ", the posterior covariance", max_abs_diff(got_cov, want.cov), 0.03);

        // A tolerance is only meaningful next to the scale it is measured
        // against: an all-zero draw would pass a mean check on its own.
        check(at + ", and the posterior is not trivially small",
              arma::abs(want.mean).max() > 0.3 && want.cov.diag().max() > 0.1);
    }
}

//////////////////////////////////////////////////////////////////////////////
// 3. Recovery.

struct Simulated
{
    arma::mat x;      ///< tt x k
    arma::mat lambda; ///< k x n
    arma::mat a;      ///< n x np
    arma::vec u_sigma_inv;
    arma::vec v_sigma_inv;
};

Simulated simulate(const int k, const int n, const int p, const int tt)
{
    Simulated s;
    s.lambda = identified_loadings(k, n);
    arma::vec free(static_cast<arma::uword>(n * (2 * k - n - 1) / 2));
    for (arma::uword i = 0; i < free.n_elem; i++)
    {
        free(i) = 0.9 - 0.25 * static_cast<double>(i % 5);
    }
    fill_lambda(s.lambda, free);

    s.a = arma::mat(n, n * p, arma::fill::zeros);
    s.a(0, 0) = 0.7;
    s.a(1, 1) = 0.45;
    s.a(1, 0) = 0.2;

    s.u_sigma_inv = arma::vec(k, arma::fill::value(4.0)); // idiosyncratic variance 0.25
    s.v_sigma_inv = arma::vec(n, arma::fill::ones);       // factor innovation variance 1

    arma::mat factors(n, tt, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        arma::vec f = arma::randn<arma::vec>(n) / arma::sqrt(s.v_sigma_inv);
        if (t > 0)
        {
            f += s.a.cols(0, n - 1) * factors.col(t - 1);
        }
        factors.col(t) = f;
    }

    arma::mat x = s.lambda * factors;
    x += arma::randn<arma::mat>(k, tt).each_col() / arma::sqrt(s.u_sigma_inv);
    s.x = x.t();
    return s;
}

bayests::DfmNormalGammaInput make_input(const Simulated &s, const int k, const int n, const int p,
                                        const int iterations, const int burnin)
{
    const int n_lambda = n * (2 * k - n - 1) / 2;
    const int n_a = n * n * p;

    bayests::DfmNormalGammaInput in;
    in.spec.k = k;
    in.spec.p = p;
    in.spec.n_factors = n;
    in.spec.iterations = iterations;
    in.spec.burnin = burnin;

    in.train.y = s.x;

    in.lambda_prior.mu = arma::vec(n_lambda, arma::fill::zeros);
    in.lambda_prior.v_inv = arma::eye<arma::mat>(n_lambda, n_lambda) * 0.01;
    in.a_prior.mu = arma::vec(n_a, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(n_a, n_a) * 0.01;

    in.u_sigma_prior.shape = arma::vec(k, arma::fill::value(3.0));
    in.u_sigma_prior.rate = arma::vec(k, arma::fill::value(2.0));
    in.v_sigma_prior.shape = arma::vec(n, arma::fill::value(3.0));
    in.v_sigma_prior.rate = arma::vec(n, arma::fill::value(2.0));

    in.initial.lambda = arma::vec(n_lambda, arma::fill::value(0.5));
    in.initial.a = arma::vec(n_a, arma::fill::zeros);
    in.initial.u_sigma_inv = arma::vec(k, arma::fill::ones);
    in.initial.v_sigma_inv = arma::vec(n, arma::fill::ones);
    return in;
}

/// The weakest of the three checks, and the one whose tolerances need thinking
/// about. The seed is fixed, but `arma::randn` and `arma::randg` go through
/// `std::normal_distribution` and `std::gamma_distribution`, whose sequences the
/// standard does not pin down -- so on another library this is a fresh sample
/// and a fresh chain, not a rerun of this one. What has to be tolerated is
/// therefore the sampling error of the whole experiment, roughly sd/sqrt(tt) per
/// coefficient taken at its maximum over fifteen of them, and not the last bits
/// of a reproducible number. The bounds below are several times what is observed
/// here and still an order of magnitude tighter than the errors a wrong
/// convention produces: overlapping the transition's lag blocks or transposing
/// A_1 moves these by more than one.
void it_recovers_what_it_was_given()
{
    std::printf("recovery from a simulated sample\n");

    const int k = 8, n = 2, p = 1, tt = 800;

    arma::arma_rng::set_seed(20260827);
    const Simulated s = simulate(k, n, p, tt);
    const bayests::DfmNormalGammaInput in = make_input(s, k, n, p, 600, 400);

    bayests::NullReporter reporter;
    const bayests::DfmNormalGammaDraws out =
        bayests::DfmNormalGammaSampler{}.draw_coefficients(in, reporter);

    check("the posterior has the shapes the model implies",
          out.lambda.n_rows == static_cast<arma::uword>(k * n) &&
              out.factors.n_rows == static_cast<arma::uword>(n * tt) &&
              out.a.n_rows == static_cast<arma::uword>(n * n * p) &&
              out.u_sigma_inv.n_rows == static_cast<arma::uword>(k) &&
              out.v_sigma_inv.n_rows == static_cast<arma::uword>(n) &&
              out.iterations() == 600);

    const arma::mat lambda_mean = arma::reshape(arma::mean(out.lambda, 1), k, n);
    const arma::mat a_mean = arma::reshape(arma::mean(out.a, 1), n, n * p);

    check_below("the loadings", max_abs_diff(lambda_mean, s.lambda), 0.25);
    check_below("the transition", max_abs_diff(a_mean, s.a), 0.20);
    check_below("the idiosyncratic precisions",
                arma::abs(arma::mean(out.u_sigma_inv, 1) / s.u_sigma_inv - 1).max(), 0.35);
    check_below("the factor innovation precisions",
                arma::abs(arma::mean(out.v_sigma_inv, 1) / s.v_sigma_inv - 1).max(), 0.50);

    // The identifying block is not drawn, so it has to be exactly itself in
    // every stored draw and not merely on average.
    const arma::mat first_draw = arma::reshape(out.lambda.col(0), k, n);
    const arma::mat identification = first_draw.submat(0, 0, n - 1, n - 1);
    arma::mat want = arma::eye<arma::mat>(n, n);
    want(1, 0) = identification(1, 0); // the one free element of the block
    check_below("the identifying block survives every draw",
                max_abs_diff(identification, want), 0.0);

    // The forecast and the log likelihood both read the factor path back, which
    // is the part of this posterior nothing else here has.
    bayests::DfmNormalGammaInput fcst_in = in;
    fcst_in.spec.h = 5;
    const bayests::ForecastDraws fcst =
        bayests::DfmNormalGammaSampler{}.forecast(fcst_in, out, reporter);
    check("the forecast is (h k) x draws",
          fcst.values.n_rows == static_cast<arma::uword>(5 * k) && fcst.values.n_cols == 600);
    check("the forecast is finite", fcst.values.is_finite());

    const arma::mat loglik = bayests::DfmNormalGammaSampler{}.log_likelihood(in, out);
    check("the log likelihood is draws x periods",
          loglik.n_rows == 600 && loglik.n_cols == static_cast<arma::uword>(tt));
    check("the log likelihood is finite and negative",
          loglik.is_finite() && loglik.max() < 0.0);
}

//////////////////////////////////////////////////////////////////////////////
// 4. What validate() has to refuse.

void expect_rejected(const std::string &what, const bayests::DfmNormalGammaInput &in)
{
    bool threw = false;
    std::string message;
    try
    {
        in.validate();
    }
    catch (const std::exception &e)
    {
        threw = true;
        message = e.what();
    }
    check(what, threw, message);
}

void the_impossible_inputs_are_refused()
{
    std::printf("inputs that do not describe a model\n");

    const int k = 6, n = 2, p = 1, tt = 40;
    arma::arma_rng::set_seed(1);
    const Simulated s = simulate(k, n, p, tt);
    const bayests::DfmNormalGammaInput good = make_input(s, k, n, p, 20, 10);

    bool accepted = true;
    try
    {
        good.validate();
    }
    catch (const std::exception &e)
    {
        accepted = false;
        std::printf("      unexpectedly rejected: %s\n", e.what());
    }
    check("a well-formed input is accepted", accepted);

    bayests::DfmNormalGammaInput no_factors = good;
    no_factors.spec.n_factors = 0;
    expect_rejected("no factors", no_factors);

    // More factors than series leaves no identifying block to fix the rotation.
    bayests::DfmNormalGammaInput too_many = good;
    too_many.spec.n_factors = k + 1;
    expect_rejected("more factors than observed series", too_many);

    bayests::DfmNormalGammaInput long_lag = good;
    long_lag.spec.p = tt;
    expect_rejected("a transition longer than the sample", long_lag);

    bayests::DfmNormalGammaInput selected = good;
    selected.spec.varsel = bayests::VarSelection::bvs;
    expect_rejected("variable selection", selected);

    bayests::DfmNormalGammaInput structural = good;
    structural.spec.structural = true;
    expect_rejected("a structural form", structural);

    bayests::DfmNormalGammaInput short_lambda = good;
    short_lambda.initial.lambda = arma::vec(3, arma::fill::zeros);
    expect_rejected("a starting Lambda of the wrong length", short_lambda);

    bayests::DfmNormalGammaInput bad_precision = good;
    bad_precision.initial.u_sigma_inv(2) = 0.0;
    expect_rejected("a starting precision that is not positive", bad_precision);
}

void run_group(const char *name, void (*group)())
{
    try
    {
        group();
    }
    catch (const std::exception &e)
    {
        std::printf("  %-52s %s\n", name, "FAILED");
        std::printf("      threw: %s\n", e.what());
        failures++;
    }
}

} // namespace

int main()
{
    run_group("group: the loading matrix", the_loading_matrix_is_identified);
    run_group("group: the lagged factors", the_lagged_factors_land_in_their_own_block);
    run_group("group: the prior over the first factors",
              the_prior_over_the_first_factors_is_the_transition);
    run_group("group: the factor path", the_factor_path_is_the_posterior_it_should_be);
    run_group("group: recovery", it_recovers_what_it_was_given);
    run_group("group: refused inputs", the_impossible_inputs_are_refused);

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
