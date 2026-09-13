// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_thin.cpp
/// @brief Checks that a thinned chain keeps every thin-th draw of the chain it thins.
///
/// Thinning decides which draws are stored and nothing else: it must not change
/// what is drawn. So a chain run with thin = t and n kept draws, and a chain run
/// from the same seed with thin = 1 and n * t kept draws, have to agree exactly --
/// column j of the first is column (j + 1) t - 1 of the second, the last draw of
/// each block of t. That is an identity between two chains rather than a value of
/// either, so it holds on every machine and every BLAS, which is what puts it on
/// the unit side of the line test/CMakeLists.txt draws.
///
/// One constant-coefficient and one time-varying sampler, the quantile pair,
/// because their inputs are the smallest to build in memory. Every sampler keeps
/// its draws through the same two VarSpec calls, so the other eighteen differ
/// from these only in what they store, not in which draws.
///
/// Around it: the arithmetic of those two calls at thin = 1, where they have to
/// be exactly the test every sampler made before thinning existed, and the two
/// chains validate() refuses.

#include "bayests/reporter.h"
#include "bayests/var_normal_ald.h"
#include "bayests/var_tvp_ald.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace
{

int failures = 0;

void check(const std::string &what, const bool ok)
{
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok)
    {
        failures++;
    }
}

/// Whether `thinned` is exactly every `thin`-th column of `full`, the last of
/// each block.
bool keeps_every(const arma::mat &thinned, const arma::mat &full, const int thin)
{
    if (thinned.n_rows != full.n_rows || thinned.n_cols * static_cast<arma::uword>(thin) != full.n_cols)
    {
        return false;
    }
    for (arma::uword j = 0; j < thinned.n_cols; j++)
    {
        if (!arma::approx_equal(thinned.col(j), full.col((j + 1) * thin - 1), "absdiff", 0.0))
        {
            return false;
        }
    }
    return true;
}

/// One equation, an intercept and one regressor -- unit_var_ald's sample.
struct Sample
{
    arma::mat y;
    arma::mat z;
    int tt = 0;
};

Sample make_sample(const int tt)
{
    Sample out;
    out.tt = tt;
    out.z = arma::mat(tt, 2);
    out.y = arma::mat(tt, 1);
    for (int t = 0; t < tt; t++)
    {
        const double x = std::sin(0.7 * t) + 0.3 * std::cos(2.1 * t);
        out.z(t, 0) = 1.0;
        out.z(t, 1) = x;
        out.y(t, 0) = 1.0 + 0.5 * x + arma::randn<double>();
    }
    return out;
}

bayests::VarNormalAldInput constant_input(const Sample &sample)
{
    bayests::VarNormalAldInput input;
    input.spec.k = 1;
    input.spec.p = 0;
    input.spec.n = 2;
    input.spec.quantile = 0.3;
    input.train.y = sample.y;
    input.train.z = sample.z;
    input.a_prior.mu = arma::vec(2, arma::fill::zeros);
    input.a_prior.v_inv = arma::eye<arma::mat>(2, 2) * 0.01;
    input.u_scale_prior.shape = arma::vec(1, arma::fill::value(3.0));
    input.u_scale_prior.rate = arma::vec(1, arma::fill::value(0.2));
    input.initial.a = arma::vec(2, arma::fill::zeros);
    input.initial.w = arma::mat(sample.tt, 1, arma::fill::ones);
    input.initial.u_scale = arma::vec(1, arma::fill::ones);
    return input;
}

bayests::VarTvpAldInput time_varying_input(const Sample &sample)
{
    bayests::VarTvpAldInput input;
    input.spec.k = 1;
    input.spec.p = 0;
    input.spec.n = 2;
    input.spec.quantile = 0.3;
    input.train.y = sample.y;
    input.train.z = sample.z;
    input.a_prior.sigma.shape = arma::vec(2, arma::fill::value(10.0));
    input.a_prior.sigma.rate = arma::vec(2, arma::fill::value(0.0001));
    input.a_prior.initial_state.mu = arma::vec(2, arma::fill::zeros);
    input.a_prior.initial_state.v_inv = arma::eye<arma::mat>(2, 2) * 0.01;
    input.u_scale_prior.shape = arma::vec(1, arma::fill::value(3.0));
    input.u_scale_prior.rate = arma::vec(1, arma::fill::value(0.2));
    input.initial.a = arma::mat(2, sample.tt, arma::fill::zeros);
    input.initial.a_sigma_inv = arma::diagmat(arma::vec(2, arma::fill::value(100.0)));
    input.initial.a_init = arma::vec(2, arma::fill::zeros);
    input.initial.w = arma::mat(sample.tt, 1, arma::fill::ones);
    input.initial.u_scale = arma::vec(1, arma::fill::ones);
    return input;
}

/// The same input run twice from one seed: kept draws `iterations` one in
/// `thin`, and `iterations * thin` of every one.
template <typename Sampler, typename Input>
void compare_chains(const Sampler &sampler, Input input, const int iterations, const int burnin,
                    const int thin, const char *model)
{
    bayests::NullReporter reporter;

    input.spec.burnin = burnin;
    input.spec.iterations = iterations;
    input.spec.thin = thin;
    arma::arma_rng::set_seed(20260913);
    const auto thinned = sampler.draw_coefficients(input, reporter);

    input.spec.iterations = iterations * thin;
    input.spec.thin = 1;
    arma::arma_rng::set_seed(20260913);
    const auto full = sampler.draw_coefficients(input, reporter);

    const std::string name(model);
    check(name + ": keeps `iterations` draws", thinned.iterations() == static_cast<arma::uword>(iterations));
    check(name + ": a is every thin-th draw of the unthinned chain", keeps_every(thinned.a, full.a, thin));
    check(name + ": u_scale likewise", keeps_every(thinned.u_scale, full.u_scale, thin));
    check(name + ": u_sigma_inv likewise", keeps_every(thinned.u_sigma_inv, full.u_sigma_inv, thin));
}

void thin_one_is_the_test_before_thinning()
{
    std::printf("thin = 1\n");

    bayests::VarSpec spec;
    spec.iterations = 7;
    spec.burnin = 5;

    bool same = spec.draws() == spec.iterations + spec.burnin;
    for (int draw = 0; draw < spec.draws(); draw++)
    {
        same = same && spec.keeps(draw) == (draw >= spec.burnin);
        if (draw >= spec.burnin)
        {
            same = same && spec.kept_index(draw) == draw - spec.burnin;
        }
    }
    check("draws(), keeps() and kept_index() reduce to burnin arithmetic", same);

    spec.thin = 3;
    int kept = 0;
    bool in_order = spec.draws() == 5 + 7 * 3;
    for (int draw = 0; draw < spec.draws(); draw++)
    {
        if (spec.keeps(draw))
        {
            in_order = in_order && spec.kept_index(draw) == kept;
            kept++;
        }
    }
    check("thin = 3 keeps iterations draws, in order, ending on the last", in_order && kept == 7 &&
                                                                           spec.keeps(spec.draws() - 1));
}

void a_thinned_chain_is_the_chain_it_thins()
{
    std::printf("the thinned chain\n");
    arma::arma_rng::set_seed(3);
    const Sample sample = make_sample(120);

    compare_chains(bayests::VarNormalAldSampler{}, constant_input(sample), 40, 25, 3, "VarNormalAld");
    compare_chains(bayests::VarTvpAldSampler{}, time_varying_input(sample), 30, 20, 4, "VarTvpAld");
}

void the_impossible_chains_are_refused()
{
    std::printf("refusals\n");
    arma::arma_rng::set_seed(4);
    const Sample sample = make_sample(60);
    bayests::NullReporter reporter;

    const auto refused = [&](const int iterations, const int burnin, const int thin) {
        bayests::VarNormalAldInput input = constant_input(sample);
        input.spec.iterations = iterations;
        input.spec.burnin = burnin;
        input.spec.thin = thin;
        try
        {
            bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter);
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        return false;
    };

    check("thin = 0 is refused", refused(10, 5, 0));
    check("a chain too long to count in an int is refused", refused(100000, 5, 100000));
}

} // namespace

int main()
{
    std::printf("unit_thin\n");

    thin_one_is_the_test_before_thinning();
    a_thinned_chain_is_the_chain_it_thins();
    the_impossible_chains_are_refused();

    std::printf("%s\n", failures == 0 ? "all checks passed" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
