// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_factor_score.cpp
/// @brief Checks the factor filter against the joint it is meant to factorise.
///
/// A factor model is scored by filtering: each column is the density of that
/// period's realised observation given the ones before it. The claim the whole
/// dataset rests on is that those columns sum to the joint log density of the
/// realised stretch -- the prediction error decomposition -- and that claim can
/// be checked without re-implementing the filter at all.
///
/// Everything here is linear and Gaussian, so the stacked realised vector is
/// itself normal, with a mean and a covariance that can be written down from
/// the transition and the loadings in a few lines. The test builds that normal
/// directly, takes its log density, and requires the filter's columns to add up
/// to it. A filter with the wrong gain, an update applied in the wrong order or
/// a state started in the wrong place all break that sum; none of them is
/// visible in a column on its own.

#include "core/models/factor_score.h"

#include <iostream>
#include <string>

using bayests::VarSpec;
using bayests::core::FactorPeriod;
using bayests::core::score_factor_forecast;

namespace
{

int failures = 0;

void check(bool condition, const std::string &what)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
        return;
    }
    std::cout << "ok: " << what << '\n';
}

constexpr arma::uword kK = 3;
constexpr arma::uword kN = 2;
constexpr arma::uword kPeriods = 4;
constexpr arma::uword kDraws = 2;

struct Model
{
    arma::mat lambda;     // k x n
    arma::mat transition; // n x n, one lag
    arma::vec v_var;      // n
    arma::vec u_var;      // k
    arma::vec f0;         // n, the drawn factor at the end of the sample
};

Model sample_model()
{
    Model m;
    m.lambda = {{1.0, 0.0}, {0.6, 0.8}, {-0.3, 0.5}};
    m.transition = {{0.5, 0.1}, {-0.2, 0.4}};
    m.v_var = {0.7, 1.3};
    m.u_var = {0.25, 0.4, 0.9};
    m.f0 = {0.8, -0.5};
    return m;
}

arma::mat realised_values()
{
    arma::mat y(kPeriods, kK);
    for (arma::uword i = 0; i < kPeriods; i++)
    {
        for (arma::uword j = 0; j < kK; j++)
        {
            y(i, j) = 0.4 * static_cast<double>(i + 1) - 0.7 * static_cast<double>(j) +
                      0.15 * static_cast<double>((i * kK + j) % 5);
        }
    }
    return y;
}

/// The stacked realised vector is normal. With one lag and a known factor at
/// the end of the sample,
///
///     f_i = A^i f_0 + sum_{l=0..i-1} A^l eta_{i-l},
///     y_i = Lambda f_i + e_i,
///
/// so the mean is Lambda A^i f_0 and the covariance is the block matrix below.
/// Written out rather than recursed, so that it shares nothing with the filter.
double joint_log_density(const Model &m, const arma::mat &realised)
{
    const arma::uword mk = kPeriods * kK;

    arma::vec mean(mk);
    arma::mat power = arma::eye<arma::mat>(kN, kN);
    std::vector<arma::mat> powers;  // A^0 .. A^{periods-1}
    for (arma::uword i = 0; i < kPeriods; i++)
    {
        powers.push_back(power);
        power = m.transition * power;
    }
    for (arma::uword i = 0; i < kPeriods; i++)
    {
        // A^{i+1} f_0, the factor of period i + 1 counted from the sample's end.
        mean.subvec(i * kK, (i + 1) * kK - 1) =
            m.lambda * (m.transition * powers[i] * m.f0);
    }

    const arma::mat q = arma::diagmat(m.v_var);
    arma::mat covariance(mk, mk, arma::fill::zeros);
    for (arma::uword i = 0; i < kPeriods; i++)
    {
        for (arma::uword j = 0; j < kPeriods; j++)
        {
            arma::mat factor_cov(kN, kN, arma::fill::zeros);
            // The innovations the two periods share: eta_1 .. eta_{min(i,j)+1}.
            const arma::uword shared = std::min(i, j) + 1;
            for (arma::uword l = 0; l < shared; l++)
            {
                factor_cov += powers[i - l] * q * arma::trans(powers[j - l]);
            }
            arma::mat block = m.lambda * factor_cov * arma::trans(m.lambda);
            if (i == j)
            {
                block += arma::diagmat(m.u_var);
            }
            covariance.submat(i * kK, j * kK, (i + 1) * kK - 1, (j + 1) * kK - 1) = block;
        }
    }

    arma::vec stacked(mk);
    for (arma::uword i = 0; i < kPeriods; i++)
    {
        stacked.subvec(i * kK, (i + 1) * kK - 1) = arma::trans(realised.row(i));
    }

    const arma::vec deviation = stacked - mean;
    double log_det = 0.0;
    double sign = 0.0;
    arma::log_det(log_det, sign, covariance);
    return -static_cast<double>(mk) * std::log(2 * arma::datum::pi) / 2 - log_det / 2 -
           arma::dot(deviation, arma::solve(covariance, deviation)) / 2;
}

VarSpec spec_of(int p)
{
    VarSpec spec;
    spec.k = static_cast<int>(kK);
    spec.n_factors = static_cast<int>(kN);
    spec.p = p;
    spec.h = static_cast<int>(kPeriods);
    return spec;
}

void test_columns_sum_to_the_joint()
{
    const Model m = sample_model();
    const arma::mat realised = realised_values();

    const auto step = [&](const arma::uword, const int i, FactorPeriod &out) {
        if (i == 0)
        {
            out.lambda = m.lambda;
            out.transition = m.transition;
            out.v_var = m.v_var;
            out.u_var = m.u_var;
            out.start = arma::mat(m.f0);
        }
    };

    const arma::mat score = score_factor_forecast(spec_of(1), realised, kDraws, step);
    check(score.n_rows == kDraws && score.n_cols == kPeriods,
          "the score is draws by scored periods");

    const double joint = joint_log_density(m, realised);
    check(std::abs(arma::accu(score.row(0)) - joint) < 1e-10,
          "and its columns sum to the joint log density of the realised stretch");
    check(std::abs(arma::accu(score.row(1)) - joint) < 1e-10,
          "for every draw, the filter drawing nothing of its own");
}

/// With no dynamics the factors are white noise, every period is independent of
/// the others, and each column is the density of that period on its own. The
/// filter has to arrive at that without being told.
void test_no_dynamics()
{
    Model m = sample_model();
    m.transition.zeros();
    const arma::mat realised = realised_values();

    const auto step = [&](const arma::uword, const int i, FactorPeriod &out) {
        if (i == 0)
        {
            out.lambda = m.lambda;
            out.transition = arma::mat();
            out.v_var = m.v_var;
            out.u_var = m.u_var;
        }
    };

    const arma::mat score = score_factor_forecast(spec_of(0), realised, 1, step);

    const arma::mat variance =
        m.lambda * arma::diagmat(m.v_var) * arma::trans(m.lambda) + arma::diagmat(m.u_var);
    double log_det = 0.0;
    double sign = 0.0;
    arma::log_det(log_det, sign, variance);

    bool all_match = true;
    for (arma::uword i = 0; i < kPeriods; i++)
    {
        const arma::vec y = arma::trans(realised.row(i));
        const double expected = -static_cast<double>(kK) * std::log(2 * arma::datum::pi) / 2 -
                                log_det / 2 - arma::dot(y, arma::solve(variance, y)) / 2;
        all_match = all_match && std::abs(score(0, i) - expected) < 1e-10;
    }
    check(all_match, "with no dynamics each period is scored on its own");
}

/// The realised values have to reach the density. Score the same model against
/// two different stretches and the numbers have to differ -- a filter that
/// dropped the update would still produce a column per period.
void test_the_realised_values_are_used()
{
    const Model m = sample_model();
    const auto step = [&](const arma::uword, const int i, FactorPeriod &out) {
        if (i == 0)
        {
            out.lambda = m.lambda;
            out.transition = m.transition;
            out.v_var = m.v_var;
            out.u_var = m.u_var;
            out.start = arma::mat(m.f0);
        }
    };

    const arma::mat first = realised_values();
    arma::mat second = first;
    second.row(0) += 1.0;

    const arma::mat a = score_factor_forecast(spec_of(1), first, 1, step);
    const arma::mat b = score_factor_forecast(spec_of(1), second, 1, step);

    check(std::abs(a(0, 0) - b(0, 0)) > 1e-8,
          "changing the first realised period changes its own column");
    check(std::abs(a(0, 1) - b(0, 1)) > 1e-8,
          "and the column after it, which conditions on it");
    check(std::abs(a(0, 2) - b(0, 2)) > 1e-8, "and the one after that");
}

} // namespace

int main()
{
    try
    {
        test_columns_sum_to_the_joint();
        test_no_dynamics();
        test_the_realised_values_are_used();
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: threw: " << e.what() << '\n';
        ++failures;
    }

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "all checks passed\n";
    return 0;
}
