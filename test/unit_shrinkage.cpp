// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_shrinkage.cpp
/// @brief Checks the adaptive priors on the coefficients and the stationarity
///        condition of the constant-coefficient Gaussian VARs.
///
/// The hierarchical Minnesota prior draws one inverse gamma per group given the
/// coefficients, so its update can be checked against the conditional it draws
/// from: many updates at one fixed set of coefficients have to average to that
/// inverse gamma's mean. The horseshoe's precision is the base precision over
/// tau^2 lambda^2 by construction, which is an identity; and on a simulated VAR
/// whose coefficients are mostly zero it has to put those zeros closer to zero
/// than a diffuse prior does, which is what it is for. A VAR near a unit root
/// estimated under the stationarity condition has to keep only stationary
/// draws. And what is refused is refused.

#include "bayests/reporter.h"
#include "bayests/var_normal_gamma.h"
#include "bayests/var_normal_wishart.h"
#include "bayests/var_tvp_gamma.h"
#include "core/models/shrinkage_support.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{

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

using bayests::NormalPrior;
using bayests::Shrinkage;
using bayests::ShrinkagePrior;
using bayests::core::CoefficientShrinkage;

void test_minnesota_update()
{
    std::printf("the hierarchical Minnesota update\n");
    NormalPrior base;
    base.mu = arma::zeros<arma::vec>(6);
    base.v_inv = arma::diagmat(arma::vec{4.0, 4.0, 1.0, 1.0, 2.0, 0.5});
    ShrinkagePrior prior;
    prior.group = {1, 1, 2, 2, 2, 0};
    prior.shape = {3.0, 2.5};
    prior.rate = {0.5, 1.0};
    const arma::vec a = {0.3, -0.2, 0.8, 0.1, -0.4, 5.0};

    CoefficientShrinkage shrinkage(Shrinkage::minnesota, prior, base, arma::vec(), arma::vec());
    arma::arma_rng::set_seed(1);
    const int n = 40000;
    arma::mat scales(2, n);
    for (int i = 0; i < n; i++)
    {
        shrinkage.update(a);
        scales.col(i) = shrinkage.scale();
    }
    // Group g's scale given a is IG(shape + n_g/2, rate + sum v_inv (a - mu)^2 / 2),
    // with mean rate' / (shape' - 1).
    const double rate1 = 0.5 + 0.5 * (4.0 * 0.09 + 4.0 * 0.04);
    const double rate2 = 1.0 + 0.5 * (1.0 * 0.64 + 1.0 * 0.01 + 2.0 * 0.16);
    const arma::vec shape = {3.0 + 1.0, 2.5 + 1.5};
    const arma::vec mean = {rate1 / (shape(0) - 1.0), rate2 / (shape(1) - 1.0)};
    const arma::vec sd = mean / arma::sqrt(shape - 2.0);
    const arma::vec z = arma::abs(arma::mean(scales, 1) - mean) / (sd / std::sqrt(double(n)));
    check_below("each group's scale averages its inverse gamma's mean, in standard errors",
                z.max(), 5.0);

    const arma::mat precision = shrinkage.precision();
    const arma::vec s = shrinkage.scale();
    check("the precision is the base over the group's scale, and as given unshrunk",
          std::abs(precision(0, 0) - 4.0 / s(0)) < 1e-12 &&
              std::abs(precision(3, 3) - 1.0 / s(1)) < 1e-12 && precision(5, 5) == 0.5);
}

void test_horseshoe_precision()
{
    std::printf("the horseshoe\n");
    NormalPrior base;
    base.mu = arma::zeros<arma::vec>(4);
    base.v_inv = arma::diagmat(arma::vec{1.0, 2.0, 1.0, 3.0});
    ShrinkagePrior prior;
    prior.group = {1, 1, 1, 0};
    CoefficientShrinkage shrinkage(Shrinkage::horseshoe, prior, base, arma::vec{0.5},
                                   arma::vec{2.0, 0.25, 1.0, 1.0});
    const arma::mat precision = shrinkage.precision();
    check("the precision is the base over tau^2 lambda_j^2",
          std::abs(precision(0, 0) - 1.0 / (0.5 * 2.0)) < 1e-12 &&
              std::abs(precision(1, 1) - 2.0 / (0.5 * 0.25)) < 1e-12 && precision(3, 3) == 3.0);

    // A coefficient far from zero pulls its local scale up; one at zero does not.
    arma::arma_rng::set_seed(2);
    const arma::vec a = {3.0, 0.0, 0.01, 0.0};
    arma::vec local_mean(3, arma::fill::zeros);
    for (int i = 0; i < 20000; i++)
    {
        shrinkage.update(a);
        local_mean += shrinkage.local().head(3);
    }
    local_mean /= 20000.0;
    check("the local scale of a large coefficient far exceeds that of a zero one",
          local_mean(0) > 10.0 * local_mean(1));
}

constexpr arma::uword kK = 4;
constexpr arma::uword kP = 2;
constexpr arma::uword kNx = kK * kP + 1;

struct Sample
{
    arma::mat y;
    arma::mat z;
    arma::mat a; // k x n_x
};

/// A VAR(2) in four variables whose only nonzero coefficients are the own
/// first lags, and an intercept.
Sample sparse_sample(const arma::uword tt, const double own)
{
    Sample s;
    s.a = arma::zeros<arma::mat>(kK, kNx);
    for (arma::uword i = 0; i < kK; i++)
    {
        s.a(i, i) = own;
    }
    arma::arma_rng::set_seed(55);
    arma::mat path(kK, tt + kP + 50, arma::fill::zeros);
    for (arma::uword t = kP; t < path.n_cols; t++)
    {
        arma::vec next = s.a.col(kK * kP) + 0.5 * arma::randn<arma::vec>(kK);
        for (arma::uword j = 1; j <= kP; j++)
        {
            next += s.a.cols((j - 1) * kK, j * kK - 1) * path.col(t - j);
        }
        path.col(t) = next;
    }
    path = path.cols(50, path.n_cols - 1);
    s.y = arma::trans(path.cols(kP, kP + tt - 1));
    arma::mat x(tt, kNx);
    x.col(kNx - 1).ones();
    for (arma::uword t = 0; t < tt; t++)
    {
        for (arma::uword j = 1; j <= kP; j++)
        {
            x.submat(t, (j - 1) * kK, t, j * kK - 1) = arma::trans(path.col(kP + t - j));
        }
    }
    s.z = arma::kron(x, arma::eye<arma::mat>(kK, kK));
    return s;
}

bayests::VarNormalWishartInput input_for(const Sample &s, const int iterations, const int burnin)
{
    bayests::VarNormalWishartInput input;
    input.spec.k = static_cast<int>(kK);
    input.spec.p = static_cast<int>(kP);
    input.spec.n = 1;
    input.spec.iterations = iterations;
    input.spec.burnin = burnin;
    input.train.y = s.y;
    input.train.z = s.z;
    const arma::uword n = kK * kNx;
    input.a_prior.mu = arma::zeros<arma::vec>(n);
    input.a_prior.v_inv = 0.01 * arma::eye<arma::mat>(n, n);
    input.u_sigma_prior.df = static_cast<int>(kK) + 2;
    input.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    input.initial.a = arma::zeros<arma::vec>(n);
    input.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return input;
}

void test_horseshoe_shrinks_zeros()
{
    std::printf("a sparse VAR under a horseshoe\n");
    const Sample s = sparse_sample(80, 0.5);
    bayests::NullReporter reporter;

    bayests::VarNormalWishartInput diffuse = input_for(s, 2000, 500);
    bayests::VarNormalWishartInput horseshoe = diffuse;
    horseshoe.spec.shrinkage = Shrinkage::horseshoe;
    horseshoe.a_prior.v_inv = arma::eye<arma::mat>(kK * kNx, kK * kNx);
    horseshoe.a_shrinkage_prior.group = arma::ones<arma::uvec>(kK * kNx);
    horseshoe.a_shrinkage_prior.group.tail(kK).zeros(); // the intercepts as given

    arma::arma_rng::set_seed(3);
    const auto d = bayests::VarNormalWishartSampler{}.draw_coefficients(diffuse, reporter);
    arma::arma_rng::set_seed(3);
    const auto h = bayests::VarNormalWishartSampler{}.draw_coefficients(horseshoe, reporter);

    const arma::vec truth = arma::vectorise(s.a);
    const arma::uvec zeros = arma::find(truth.head(kK * kK * kP) == 0.0);
    const arma::vec mean_diffuse = arma::mean(d.a, 1);
    const arma::vec mean_horseshoe = arma::mean(h.a, 1);
    const double error_diffuse = arma::accu(arma::square(mean_diffuse.elem(zeros)));
    const double error_horseshoe = arma::accu(arma::square(mean_horseshoe.elem(zeros)));
    std::printf("    squared error over the %u zero coefficients: diffuse %.4f, horseshoe %.4f\n",
                static_cast<unsigned>(zeros.n_elem), error_diffuse, error_horseshoe);
    check_below("the horseshoe puts the zeros closer to zero", error_horseshoe / error_diffuse, 0.5);
    check("its scales are drawn and kept",
          h.a_shrinkage.n_rows == 1 && h.a_local.n_rows == kK * kNx && h.a_shrinkage.is_finite() &&
              h.a_shrinkage.min() > 0.0);
    check("and nothing of the kind without one", d.a_shrinkage.is_empty() && d.a_local.is_empty());
}

void test_stationarity()
{
    std::printf("a VAR near a unit root, estimated stationary\n");
    const Sample s = sparse_sample(60, 0.98);
    bayests::VarNormalWishartInput input = input_for(s, 400, 100);
    input.spec.stationary = true;
    bayests::NullReporter reporter;
    arma::arma_rng::set_seed(4);
    const auto draws = bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);
    bool all = true;
    for (arma::uword d = 0; d < draws.a.n_cols; d++)
    {
        all = all && bayests::core::is_stationary(draws.a.col(d), kK, kP);
    }
    check("every kept draw is stationary", all);

    input.spec.stationary = false;
    arma::arma_rng::set_seed(4);
    const auto free = bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);
    arma::uword explosive = 0;
    for (arma::uword d = 0; d < free.a.n_cols; d++)
    {
        explosive += bayests::core::is_stationary(free.a.col(d), kK, kP) ? 0 : 1;
    }
    std::printf("    without the condition, %u of %u draws are explosive\n",
                static_cast<unsigned>(explosive), static_cast<unsigned>(free.a.n_cols));
    check("the condition bites: without it some draws are explosive", explosive > 0);
}

template <typename Input>
bool refuses(Input input, const std::string &contains)
{
    try
    {
        input.validate();
    }
    catch (const std::invalid_argument &e)
    {
        const bool ok = std::string(e.what()).find(contains) != std::string::npos;
        if (!ok)
        {
            std::printf("    refused, but: %s\n", e.what());
        }
        return ok;
    }
    return false;
}

void test_refusals()
{
    std::printf("refusals\n");
    const Sample s = sparse_sample(30, 0.5);
    bayests::VarNormalWishartInput base = input_for(s, 10, 0);
    base.spec.shrinkage = Shrinkage::minnesota;
    base.a_shrinkage_prior.group = arma::ones<arma::uvec>(kK * kNx);
    base.a_shrinkage_prior.shape = {2.0};
    base.a_shrinkage_prior.rate = {1.0};

    bool ok = true;
    try
    {
        base.validate();
    }
    catch (const std::exception &e)
    {
        std::printf("    %s\n", e.what());
        ok = false;
    }
    check("a hierarchical Minnesota prior is read", ok);

    auto input = base;
    input.a_shrinkage_prior.shape.reset();
    check("without a shape per group it is refused", refuses(input, "/shape"));

    input = base;
    input.a_shrinkage_prior.group(3) = 3;
    check("groups with a gap are refused", refuses(input, "no coefficient in group"));

    input = base;
    input.a_prior.v_inv(0, 1) = input.a_prior.v_inv(1, 0) = 0.001;
    check("a base precision coupling shrunk positions is refused", refuses(input, "couples"));

    input = base;
    input.spec.varsel = bayests::VarSelection::bvs;
    check("so is shrinkage beside variable selection", refuses(input, "variable selection"));

    input = base;
    input.spec.shrinkage = Shrinkage::none;
    check("a group without a scheme is refused", refuses(input, "is none"));

    bayests::VarTvpGammaInput tvp;
    tvp.spec.k = static_cast<int>(kK);
    tvp.spec.stationary = true;
    check("VarTvpGamma refuses the stationarity condition by name",
          refuses(tvp, "VarTvpGamma does not read /model/shrinkage or /model/stationary"));
}

} // namespace

int main()
{
    try
    {
        test_minnesota_update();
        test_horseshoe_precision();
        test_horseshoe_shrinks_zeros();
        test_stationarity();
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
