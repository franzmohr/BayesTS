// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That FavarNormalWishart draws the model it says it does.
//
// A factor augmented VAR is the dynamic factor model beside it with half of its
// state observed, so this checks the half that is new and leans on
// unit_dfm_normal_gamma.cpp for the half that is not. Four directions, in
// decreasing order of how exact the check is:
//
//   1. The identification, exactly. Which cells of Lambda are free is the one
//      piece of arithmetic that differs from a DFM's, and it differs in two
//      places at once -- the leading block of the factor columns is the
//      *identity* rather than a unit lower triangle, and the observed columns of
//      those rows are zero. Getting only the second right leaves a model that
//      runs and is not identified, its loadings free to wander along a rotation
//      the free Q no longer rules out. That is checked here as an algebraic
//      statement about the transformations the state admits, not merely as a
//      table of cells.
//   2. The two identities the sampler is written on: that the SUR precision of
//      the transition collapses to a Kronecker product even with a full Q, and
//      that the pointwise log likelihood scores the panel against the whole
//      state.
//   3. Recovery, including the cross covariance of Q. That block is what a FAVAR
//      is estimated to measure, and a fixture whose observed factors did not
//      move with the factors would leave it untested.
//   4. What validate() has to refuse.
//
// The conditional path draw itself is not re-checked here. It is
// chan_jeliazkov_2009_conditional's, and unit_chan_jeliazkov.cpp puts it against
// a dense conditional built by a route that shares no index arithmetic with it.
//
// No fixture and no file: everything is built in memory.

#include "bayests/favar_normal_wishart.h"
#include "core/models/favar_support.h"

#include <cstdio>
#include <string>

namespace
{

using bayests::core::favar_identified_loadings;
using bayests::core::favar_lambda_row_width;
using bayests::core::fill_favar_lambda;
using bayests::core::fill_lagged_factors;
using bayests::core::stacked_state;

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
// 1. The identification.

/// Lambda's leading n x n block is the identity in the factor columns and zero
/// in the observed ones, and neither half is ever drawn.
void the_loading_matrix_is_identified()
{
    std::printf("the loading matrix carries both halves of its identification\n");

    const int k = 5, n = 3, n_obs = 2;
    const int ns = n + n_obs;
    arma::mat lambda = favar_identified_loadings(k, n, n_obs);

    // Rows 0, 1 and 2 are the identifying block and contribute nothing; rows 3
    // and 4 contribute the whole width, 5 each.
    const arma::vec free = arma::regspace<arma::vec>(1, 10);
    fill_favar_lambda(lambda, free, n, n_obs);

    check("the free elements are (k - n) n_state", free.n_elem == 10);

    arma::mat want(k, ns, arma::fill::zeros);
    for (int i = 0; i < n; i++)
    {
        want(i, i) = 1.0;
    }
    for (int j = 0; j < ns; j++)
    {
        want(3, j) = 1 + j;                                // row 3: the whole width
        want(4, j) = 6 + j;                                // row 4: the whole width
    }
    check_below("the free elements fill in row by row", max_abs_diff(lambda, want), 0.0);

    check("the identifying block is the identity, not a triangle",
          arma::approx_equal(lambda.submat(0, 0, n - 1, n - 1),
                             arma::eye<arma::mat>(n, n), "absdiff", 0.0));
    check_below("the observed columns of the first n rows are zero",
                arma::abs(lambda.submat(0, n, n - 1, ns - 1)).max(), 0.0);
}

/// The identification, as the algebra it is: the two transformations of the
/// state that leave the measurement unchanged are exactly the two the fixed
/// block rules out, and a weaker block would not rule out the first.
///
/// This is the check that would have caught taking the DFM's rule over. A unit
/// lower triangular block admits every unit lower triangular C, and with Q free
/// there is nothing else to reject it -- the model runs and its loadings wander.
/// The statements below are about Lambda alone, which is where the restriction
/// lives.
void the_identification_rules_out_what_it_must()
{
    std::printf("the fixed block rules out the transformations of the state\n");

    const int k = 5, n = 3, n_obs = 2;
    arma::mat lambda = favar_identified_loadings(k, n, n_obs);
    fill_favar_lambda(lambda, arma::regspace<arma::vec>(1, 10), n, n_obs);

    const arma::mat lambda_f = lambda.head_cols(n);
    const arma::mat lambda_y = lambda.tail_cols(n_obs);

    // F -> C F takes Lambda_f to Lambda_f C^-1, whose leading block is C^-1.
    // The identity there is the identity only at C = I, so any C but the
    // identity breaks the restriction -- including a unit lower triangular one,
    // which is precisely what a dynamic factor model's rule would have allowed.
    arma::mat c = arma::eye<arma::mat>(n, n);
    c(1, 0) = 0.4;
    c(2, 1) = -0.3; // unit lower triangular, and not the identity
    const arma::mat rotated = lambda_f * arma::inv(c);
    check("a unit lower triangular rotation is visible in the block",
          arma::abs(rotated.head_rows(n) - arma::eye<arma::mat>(n, n)).max() > 0.1);
    check_below("while the same rotation keeps a unit lower triangle one",
                arma::abs(arma::trimatu(rotated.head_rows(n), 1)).max() +
                    arma::abs(rotated.head_rows(n).diag() -
                              arma::vec(n, arma::fill::ones)).max(),
                1e-12);

    // F -> F + D Y takes Lambda_y to Lambda_y - Lambda_f D, whose leading block
    // goes from zero to -D. Zero there is zero only at D = 0.
    arma::mat d(n, n_obs, arma::fill::zeros);
    d(0, 0) = 0.5;
    const arma::mat shifted = lambda_y - lambda_f * d;
    check("a shift of the factors by the observed block is visible",
          arma::abs(shifted.head_rows(n)).max() > 0.1);
    check_below("and the untransformed block is zero",
                arma::abs(lambda_y.head_rows(n)).max(), 0.0);
}

/// The two counts in spec.h are different functions and must stay so.
///
/// n_favar_lambda() is not n_lambda() at any dimension -- the two restrictions
/// differ, and they differ by exactly the n(n-1)/2 loadings a dynamic factor
/// model leaves free and pays for with a diagonal V. That equality is the one
/// worth checking: it says the two identifications cost the same, which is why
/// neither model is more restricted than the other.
void the_two_loading_counts_are_different_functions()
{
    std::printf("the FAVAR loading count is not the DFM's\n");

    bool differ = true, costs_match = true;
    for (int k = 3; k <= 9; k++)
    {
        for (int n = 1; n <= 3 && n <= k; n++)
        {
            bayests::VarSpec spec;
            spec.k = k;
            spec.n_factors = n;

            // With no observed factors the two describe the same measurement,
            // and the FAVAR's is tighter by the triangle it fixes.
            differ = differ && spec.n_favar_lambda() <= spec.n_lambda();
            costs_match = costs_match &&
                          spec.n_lambda() - spec.n_favar_lambda() == n * (n - 1) / 2;
        }
    }
    check("it fixes what the DFM leaves free", differ);
    check("and the two identifications cost the same n(n-1)/2", costs_match);

    // Row zero has no free loading under either rule; every row from n on
    // carries the whole state under this one.
    bayests::VarSpec spec;
    spec.k = 6;
    spec.p = 2;
    spec.n_factors = 2;
    spec.n_obs_factors = 3;
    check("every row from n_factors on carries the whole state",
          favar_lambda_row_width(0, 2, 3) == 0 && favar_lambda_row_width(1, 2, 3) == 0 &&
              favar_lambda_row_width(2, 2, 3) == 5 && favar_lambda_row_width(5, 2, 3) == 5);
    check("and the count is (k - n_factors) n_state", spec.n_favar_lambda() == 4 * 5);

    // The transition widens from the factors to the whole state.
    check("n_favar_a() counts the whole state",
          spec.n_favar_a() == 5 * 5 * 2 && spec.n_state() == 5);
    spec.n_obs_factors = 0;
    check("and is n_factor_a() when there are none",
          spec.n_favar_a() == spec.n_factor_a());
}

//////////////////////////////////////////////////////////////////////////////
// 2. The identities the sampler is written on.

/// The ns equations of the transition share their regressors, so the SUR
/// precision sum_t Z_t' Q^-1 Z_t collapses to kron(X_a X_a', Q^-1) and no
/// (tt ns) x (ns^2 p) design is ever built.
///
/// DfmNormalGamma leans on the same identity with a diagonal V. This checks it
/// survives a *full* Q, which is the one thing that changes here -- it does,
/// because the collapse is a statement about the regressors and not about the
/// covariance, and a check is cheaper than trusting that sentence.
void the_transition_precision_collapses()
{
    std::printf("the SUR precision is a Kronecker product, full Q included\n");

    const int ns = 3, p = 2, tt = 12;
    arma::arma_rng::set_seed(20260904);

    const arma::mat state = arma::randn<arma::mat>(ns, tt);
    arma::mat x_a(ns * p, tt);
    fill_lagged_factors(x_a, state, ns, p);

    // A full precision, and deliberately far from diagonal.
    arma::mat q_root = arma::randn<arma::mat>(ns, ns);
    const arma::mat q_inv = arma::symmatu(q_root * q_root.t() + ns * arma::eye(ns, ns));
    check("the test's Q is not diagonal",
          arma::abs(q_inv - arma::diagmat(q_inv.diag())).max() > 0.1);

    const arma::mat collapsed = arma::kron(x_a * arma::trans(x_a), q_inv);

    // The direct spelling: one Kronecker product per period, summed.
    arma::mat direct(ns * ns * p, ns * ns * p, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        direct += arma::kron(x_a.col(t) * arma::trans(x_a.col(t)), q_inv);
    }
    check_below("kron(X X', Q^-1) is the sum over periods",
                max_abs_diff(collapsed, direct), 1e-10);

    // And the right hand side, which is the same identity read the other way.
    arma::vec direct_rhs(ns * ns * p, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        direct_rhs += arma::kron(x_a.col(t), q_inv * state.col(t));
    }
    const arma::vec collapsed_rhs = arma::vectorise(q_inv * (state * arma::trans(x_a)));
    check_below("and the moment vector agrees too",
                arma::abs(collapsed_rhs - direct_rhs).max(), 1e-10);
}

//////////////////////////////////////////////////////////////////////////////
// 3. Recovery, and the shapes downstream.

struct Simulated
{
    arma::mat x;       // k x tt panel
    arma::mat obs;     // tt x n_obs, one period per row
    arma::mat lambda;  // k x ns
    arma::mat a;       // ns x (ns p)
    arma::mat q;       // ns x ns
    arma::vec u_sigma_inv;
};

/// A sample from the model, with the observed factors generated by the same
/// joint transition the sampler assumes.
///
/// The cross terms of Phi and of Q are set deliberately: with an observed block
/// that did not move with the factors, the one thing separating this model from
/// a dynamic factor model would have nothing in the data to find, and the
/// recovery check below would pass on a sampler that ignored the coupling.
Simulated simulate(const int k, const int n, const int n_obs, const int p, const int tt)
{
    const int ns = n + n_obs;

    Simulated s;
    s.a = arma::mat(ns, ns * p, arma::fill::zeros);
    s.a.submat(0, 0, ns - 1, ns - 1) = 0.45 * arma::eye<arma::mat>(ns, ns);
    s.a(0, n) = 0.30;  // the factor responds to the observed variable
    s.a(n, 0) = 0.25;  // and it responds back

    // A full innovation covariance with a real cross block.
    s.q = 0.25 * arma::eye<arma::mat>(ns, ns);
    s.q(0, n) = 0.10;
    s.q(n, 0) = 0.10;

    const arma::mat q_root = arma::chol(s.q, "lower");

    arma::mat state(ns, tt, arma::fill::zeros);
    for (int t = 0; t < tt; t++)
    {
        arma::vec v = q_root * arma::randn<arma::vec>(ns);
        for (int j = 1; j <= p && t - j >= 0; j++)
        {
            v += s.a.cols((j - 1) * ns, j * ns - 1) * state.col(t - j);
        }
        state.col(t) = v;
    }

    s.lambda = favar_identified_loadings(k, n, n_obs);
    for (int i = n; i < k; i++)
    {
        for (int j = 0; j < ns; j++)
        {
            s.lambda(i, j) = 0.4 + 0.15 * static_cast<double>((i + j) % 4);
        }
    }

    s.u_sigma_inv = arma::vec(k, arma::fill::value(9.0)); // sd 1/3
    s.x = s.lambda * state;
    for (int i = 0; i < k; i++)
    {
        s.x.row(i) += arma::trans(arma::randn<arma::vec>(tt)) / std::sqrt(s.u_sigma_inv(i));
    }

    s.obs = arma::trans(state.rows(n, ns - 1));
    return s;
}

bayests::FavarNormalWishartInput make_input(const Simulated &s, const int k, const int n,
                                            const int n_obs, const int p, const int iterations,
                                            const int burnin)
{
    const int ns = n + n_obs;

    bayests::FavarNormalWishartInput in;
    in.spec.k = k;
    in.spec.p = p;
    in.spec.n_factors = n;
    in.spec.n_obs_factors = n_obs;
    in.spec.iterations = iterations;
    in.spec.burnin = burnin;

    in.train.y = arma::vectorise(s.x);
    in.train.f_obs = s.obs;

    const arma::uword n_lambda = static_cast<arma::uword>(in.spec.n_favar_lambda());
    in.lambda_prior.mu = arma::vec(n_lambda, arma::fill::zeros);
    in.lambda_prior.v_inv = arma::eye<arma::mat>(n_lambda, n_lambda) * 0.01;
    in.initial.lambda = arma::vec(n_lambda, arma::fill::value(0.5));

    const arma::uword n_a = static_cast<arma::uword>(in.spec.n_favar_a());
    in.a_prior.mu = arma::vec(n_a, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(n_a, n_a) * 0.01;
    in.initial.a = arma::vec(n_a, arma::fill::zeros);

    in.u_sigma_prior.shape = arma::vec(k, arma::fill::value(3.0));
    in.u_sigma_prior.rate = arma::vec(k, arma::fill::value(2.0));
    in.initial.u_sigma_inv = arma::vec(k, arma::fill::ones);

    in.v_sigma_prior.df = ns + 1;
    in.v_sigma_prior.scale = arma::eye<arma::mat>(ns, ns);
    in.initial.v_sigma_inv = arma::eye<arma::mat>(ns, ns);

    return in;
}

void it_recovers_what_it_was_given()
{
    std::printf("recovery from a simulated sample\n");

    const int k = 8, n = 2, n_obs = 2, p = 1, tt = 800;
    const int ns = n + n_obs;

    arma::arma_rng::set_seed(20260904);
    const Simulated s = simulate(k, n, n_obs, p, tt);
    const bayests::FavarNormalWishartInput in = make_input(s, k, n, n_obs, p, 600, 400);

    bayests::NullReporter reporter;
    const bayests::FavarNormalWishartDraws out =
        bayests::FavarNormalWishartSampler{}.draw_coefficients(in, reporter);

    check("the posterior has the shapes the model implies",
          out.lambda.n_rows == static_cast<arma::uword>(k * ns) &&
              out.factors.n_rows == static_cast<arma::uword>(n * tt) &&
              out.a.n_rows == static_cast<arma::uword>(ns * ns * p) &&
              out.u_sigma_inv.n_rows == static_cast<arma::uword>(k) &&
              out.v_sigma_inv.n_rows == static_cast<arma::uword>(ns * ns) &&
              out.iterations() == 600);

    // The factor path is the unobserved half alone. Storing the observed half
    // beside it would be a copy of the input once per draw.
    check("only the unobserved factors are stored as a path",
          out.factors.n_rows == static_cast<arma::uword>(n * tt));

    const arma::mat lambda_mean = arma::reshape(arma::mean(out.lambda, 1), k, ns);
    const arma::mat a_mean = arma::reshape(arma::mean(out.a, 1), ns, ns * p);
    const arma::mat q_inv_mean = arma::reshape(arma::mean(out.v_sigma_inv, 1), ns, ns);
    const arma::mat q_mean = arma::inv_sympd(arma::symmatu(q_inv_mean));

    check_below("the loadings", max_abs_diff(lambda_mean, s.lambda), 0.25);
    check_below("the transition", max_abs_diff(a_mean, s.a), 0.20);
    check_below("the idiosyncratic precisions",
                arma::abs(arma::mean(out.u_sigma_inv, 1) / s.u_sigma_inv - 1).max(), 0.35);

    // The state innovation covariance, and its cross block in particular. That
    // block is the correlation between the factor innovations and the shock to
    // the observed variables, and it is what this model exists to estimate.
    check_below("the state innovation covariance", max_abs_diff(q_mean, s.q), 0.10);
    check_below("its factor-to-observed cross covariance",
                std::abs(q_mean(0, n) - s.q(0, n)), 0.06);
    check("and that cross covariance is not near zero", std::abs(s.q(0, n)) > 0.05);

    // The identification is not drawn, so it has to be exactly itself in every
    // stored draw and not merely on average -- the identity in the factor
    // columns and zero in the observed ones.
    arma::mat want = arma::zeros<arma::mat>(n, ns);
    want.submat(0, 0, n - 1, n - 1) = arma::eye<arma::mat>(n, n);
    bool every_draw = true;
    for (arma::uword d = 0; d < out.iterations(); d++)
    {
        const arma::mat drawn = arma::reshape(out.lambda.col(d), k, ns);
        every_draw = every_draw && max_abs_diff(drawn.head_rows(n), want) == 0.0;
    }
    check("the identifying block survives every draw", every_draw);

    // The forecast is the one here that is wider than k: the panel of a horizon
    // followed by the observed factors of the same horizon.
    bayests::FavarNormalWishartInput fcst_in = in;
    fcst_in.spec.h = 5;
    const bayests::ForecastDraws fcst =
        bayests::FavarNormalWishartSampler{}.forecast(fcst_in, out, reporter);
    check("the forecast is (h (k + n_obs_factors)) x draws",
          fcst.values.n_rows == static_cast<arma::uword>(5 * (k + n_obs)) &&
              fcst.values.n_cols == 600);
    check("the forecast is finite", fcst.values.is_finite());

    const arma::mat loglik = bayests::FavarNormalWishartSampler{}.log_likelihood(in, out);
    check("the log likelihood is draws x periods",
          loglik.n_rows == 600 && loglik.n_cols == static_cast<arma::uword>(tt));
    // Not "negative": a log *density* may exceed zero, and with idiosyncratic
    // standard deviations of a third it does. The statement that holds is that
    // no observation scores above the peak of its own density, which is what the
    // quadratic form being non-negative comes to.
    arma::vec peak(out.iterations());
    for (arma::uword d = 0; d < out.iterations(); d++)
    {
        peak(d) = -k * std::log(2 * arma::datum::pi) / 2 +
                  arma::accu(arma::log(out.u_sigma_inv.col(d))) / 2;
    }
    check("the log likelihood is finite", loglik.is_finite());
    check("no observation scores above the peak of its density",
          arma::all(arma::max(loglik, 1) <= peak + 1e-9));

    // The identity the log likelihood is written on: the panel is scored against
    // the *whole* state, drawn half and observed half alike. Recomputed here from
    // the stored draw by the direct route, one period at a time.
    const arma::mat lambda0 = arma::reshape(out.lambda.col(0), k, ns);
    const arma::mat factors0 = arma::reshape(out.factors.col(0), n, tt);
    const arma::mat state0 = stacked_state(factors0, arma::trans(s.obs));
    const arma::vec prec0 = out.u_sigma_inv.col(0);
    const arma::mat resid = s.x - lambda0 * state0;

    double worst = 0.0;
    for (int t = 0; t < tt; t++)
    {
        double want_t = -k * std::log(2 * arma::datum::pi) / 2 +
                        arma::accu(arma::log(prec0)) / 2 -
                        arma::dot(prec0, arma::square(resid.col(t))) / 2;
        worst = std::max(worst, std::abs(loglik(0, t) - want_t));
    }
    check_below("it is the panel's density against the whole state", worst, 1e-9);
}

//////////////////////////////////////////////////////////////////////////////
// 4. What validate() has to refuse.

bool rejected(const bayests::FavarNormalWishartInput &in)
{
    try
    {
        in.validate();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

void the_impossible_inputs_are_refused()
{
    std::printf("inputs that do not describe the model are refused\n");

    const int k = 6, n = 2, n_obs = 2, p = 1, tt = 40;
    arma::arma_rng::set_seed(11);
    const Simulated s = simulate(k, n, n_obs, p, tt);
    const bayests::FavarNormalWishartInput good = make_input(s, k, n, n_obs, p, 10, 5);

    check("the well-formed input is accepted", !rejected(good));

    // A model with no observed factors is a dynamic factor model, and saying so
    // is more useful than estimating one under this name.
    bayests::FavarNormalWishartInput no_obs = good;
    no_obs.spec.n_obs_factors = 0;
    check("no observed factors is refused", rejected(no_obs));

    bayests::FavarNormalWishartInput bad_obs = good;
    bad_obs.train.f_obs = arma::randn<arma::mat>(tt - 1, n_obs);
    check("observed factors that miss a period are refused", rejected(bad_obs));

    bayests::FavarNormalWishartInput wide_obs = good;
    wide_obs.train.f_obs = arma::randn<arma::mat>(tt, n_obs + 1);
    check("observed factors of the wrong width are refused", rejected(wide_obs));

    // The two shapes a file written against a dynamic factor model gets wrong:
    // its state innovation prior and starting value are diagonals, not matrices.
    bayests::FavarNormalWishartInput diag_prior = good;
    diag_prior.v_sigma_prior.scale = arma::eye<arma::mat>(n, n);
    check("a state innovation scale of the wrong size is refused", rejected(diag_prior));

    bayests::FavarNormalWishartInput diag_initial = good;
    diag_initial.initial.v_sigma_inv = arma::eye<arma::mat>(n, n);
    check("a state innovation starting value of the wrong size is refused",
          rejected(diag_initial));

    bayests::FavarNormalWishartInput thin = good;
    thin.v_sigma_prior.df = n + n_obs - 1;
    check("too few Wishart degrees of freedom are refused", rejected(thin));

    // The FAVAR loading count, not the DFM's -- a prior sized by n_lambda()
    // would fit a model without observed loadings and has to be caught.
    bayests::FavarNormalWishartInput dfm_lambda = good;
    const arma::uword dfm_n = static_cast<arma::uword>(good.spec.n_lambda());
    dfm_lambda.lambda_prior.mu = arma::vec(dfm_n, arma::fill::zeros);
    dfm_lambda.lambda_prior.v_inv = arma::eye<arma::mat>(dfm_n, dfm_n);
    dfm_lambda.initial.lambda = arma::vec(dfm_n, arma::fill::zeros);
    check("a lambda block sized for a dynamic factor model is refused",
          rejected(dfm_lambda));

    bayests::FavarNormalWishartInput selected = good;
    selected.spec.varsel = bayests::VarSelection::ssvs;
    check("variable selection is refused", rejected(selected));

    bayests::FavarNormalWishartInput structural = good;
    structural.spec.structural = true;
    check("a structural block is refused", rejected(structural));

    bayests::FavarNormalWishartInput covar = good;
    covar.spec.covar = true;
    check("a psi block is refused", rejected(covar));

    bayests::FavarNormalWishartInput crowded = good;
    crowded.spec.n_factors = k + 1;
    check("more factors than series is refused", rejected(crowded));
}

/// Runs one group and reports a throw as a failure of that group rather than
/// letting it abort the harness.
void run_group(const char *name, void (*group)())
{
    try
    {
        group();
    }
    catch (const std::exception &e)
    {
        std::printf("  %-52s %s\n", name, "FAIL");
        std::printf("      threw: %s\n", e.what());
        failures++;
    }
}

} // namespace

int main()
{
    run_group("group: the loading matrix", the_loading_matrix_is_identified);
    run_group("group: what the identification rules out",
              the_identification_rules_out_what_it_must);
    run_group("group: the two loading counts", the_two_loading_counts_are_different_functions);
    run_group("group: the transition precision", the_transition_precision_collapses);
    run_group("group: recovery", it_recovers_what_it_was_given);
    run_group("group: refused inputs", the_impossible_inputs_are_refused);

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
