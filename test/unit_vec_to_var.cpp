// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_vec_to_var.cpp
/// @brief Checks the VEC-to-VAR transformation against the identity that
///        defines it.
///
/// Unlike the golden harness next door, this is a numerical test with expected
/// values, and it can be: nothing here is a posterior draw. The transformation
/// is a change of basis, so it has an answer that does not depend on the
/// compiler, the BLAS or the CPU -- the level VAR must predict, to the last bit
/// an exact identity allows, whatever the VEC it came from predicts for the same
/// data. Every case below builds coefficients directly, converts them, and
/// compares the two predictions.
///
/// What that pins, and what a fingerprint could not: the block counts. A VEC's
/// `p` is the *level* lag order, so the model carries p - 1 lagged differences
/// and not p, and the same offset applies to `s`. Reading it the other way round
/// produces a matrix of the wrong width, which is a shape error the first case
/// catches, and coefficients assigned to the wrong lag, which only the
/// prediction cases do. The counts came out of
/// bvartools-vec-normal-wishart.h5 -- 16 coefficients for k = 2, p = 2, rank = 2
/// and 4 deterministic terms -- and the first case restates them so that a file
/// is not needed to run this.

#include "bayests/vec_to_var.h"

#include <cmath>
#include <cstdio>

using bayests::VarSpec;
using bayests::VecNormalWishartDraws;
using bayests::vec_to_var_coefficients;
using bayests::vec_to_var_spec;

namespace
{

int failures = 0;

/// Exact to within an accumulation of rounding: both sides are the same sums of
/// the same products, so anything above this is a wrong coefficient rather than
/// a wrong last digit.
constexpr double tolerance = 1e-12;

void check_close(const char *what, const double got, const double want)
{
    const bool ok = std::abs(got - want) < tolerance;
    std::printf("  %-44s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
    {
        std::printf("      got %.17g, want %.17g\n", got, want);
        failures++;
    }
}

void check_equal(const char *what, const arma::uword got, const arma::uword want)
{
    const bool ok = got == want;
    std::printf("  %-44s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
    {
        std::printf("      got %llu, want %llu\n", static_cast<unsigned long long>(got),
                    static_cast<unsigned long long>(want));
        failures++;
    }
}

/// One draw's coefficient matrix, [A_1 .. A_p, B_0 .. B_s, C, Pi_d], recovered
/// from the vectorised posterior the transformation returns.
arma::mat level_coefficients(const arma::mat &a, const int k)
{
    return arma::reshape(a.col(0), k, a.n_rows / static_cast<arma::uword>(k));
}

/// The shapes the recorded model file has, restated so this needs no file.
void fixture_shape()
{
    std::printf("shape of bvartools-vec-normal-wishart.h5 (k=2, p=2, rank=2, n=4)\n");

    VarSpec spec;
    spec.k = 2;
    spec.p = 2;
    spec.n = 4;
    spec.rank = 2;
    spec.k_beta = 2; // y_{t-1} alone: no unmodelled variables, no restricted terms
    spec.iterations = 1;

    const VarSpec var = vec_to_var_spec(spec);
    check_equal("level lag order is p, not p + 1", static_cast<arma::uword>(var.p), 2);
    check_equal("level exogenous order stays 0", static_cast<arma::uword>(var.s), 0);
    check_equal("cointegration rank cleared", static_cast<arma::uword>(var.rank), 0);

    // p = 2 means one Gamma block, so the VEC holds
    // k*rank + k*(k*1 + 4) = 4 + 12 coefficients. This is the number the file's
    // /initial/a carries and the number of columns in its /data/train/z.
    const arma::uword nparams_vec = 16;
    check_equal("VEC nparams", 2 * 2 + 2 * (2 * 1 + 4), nparams_vec);
    check_equal("level nparams", static_cast<arma::uword>(var.nparams_per_period()), 16);

    // The accessors have to agree with the file, and the VAR one has to differ
    // from them -- it counts k^2 p where a VEC has k^2 (p - 1), which is the
    // confusion the two names exist to keep apart.
    check_equal("spec.n_gamma()", static_cast<arma::uword>(spec.n_gamma()), 1);
    check_equal("spec.n_upsilon()", static_cast<arma::uword>(spec.n_upsilon()), 0);
    check_equal("spec.nparams_per_period_vec()",
                static_cast<arma::uword>(spec.nparams_per_period_vec()), nparams_vec);
    check_equal("spec.nparams_per_period() is the VAR count",
                static_cast<arma::uword>(spec.nparams_per_period()), 20);

    VecNormalWishartDraws draws;
    draws.a = arma::randn<arma::mat>(nparams_vec, 3);
    draws.beta = arma::randn<arma::mat>(4, 3);
    draws.u_sigma_inv = arma::randn<arma::mat>(4, 3);

    const auto out = vec_to_var_coefficients(spec, draws);
    check_equal("converted a rows", out.a.n_rows, 16);
    check_equal("chain length preserved", out.a.n_cols, 3);
    check_equal("u_sigma_inv shared unchanged", out.u_sigma_inv.n_elem, draws.u_sigma_inv.n_elem);

    // Selection does not survive the transformation: A_i is Gamma_i - Gamma_{i-1},
    // and one indicator cannot say which half was in.
    check_equal("inclusion indicators dropped", out.a_lambda.n_elem, 0);
}

/// dy_t = Pi y_{t-1} + Gamma_1 dy_{t-1} + C d_t, whose level form is
/// A_1 = I + Pi + Gamma_1, A_2 = -Gamma_1.
void endogenous_only()
{
    std::printf("\nprediction with k=2, p=2, rank=2, n=1\n");

    const int k = 2, rank = 2, n = 1;
    VarSpec spec;
    spec.k = k;
    spec.p = 2;
    spec.n = n;
    spec.rank = rank;
    spec.k_beta = k; // k_ect == k here
    spec.iterations = 1;

    const arma::mat alpha = arma::randn<arma::mat>(k, rank);
    const arma::mat beta_mat = arma::randn<arma::mat>(k, rank);
    const arma::mat gamma_1 = arma::randn<arma::mat>(k, k);
    const arma::mat c = arma::randn<arma::mat>(k, n);
    const arma::mat pi = alpha * beta_mat.t();

    VecNormalWishartDraws draws;
    draws.a = arma::join_cols(arma::vectorise(alpha),
                              arma::vectorise(arma::join_rows(gamma_1, c)));
    draws.beta = arma::vectorise(beta_mat);
    draws.u_sigma_inv = arma::vectorise(arma::eye<arma::mat>(k, k));

    const arma::mat level = level_coefficients(vec_to_var_coefficients(spec, draws).a, k);
    check_equal("level coefficient columns", level.n_cols,
                static_cast<arma::uword>(k * 2 + n));

    const arma::vec y_1 = arma::randn<arma::vec>(k);
    const arma::vec y_2 = arma::randn<arma::vec>(k);
    const arma::vec d = arma::randn<arma::vec>(n);

    const arma::vec from_vec = y_1 + pi * y_1 + gamma_1 * (y_1 - y_2) + c * d;
    const arma::vec from_var = level.cols(0, k - 1) * y_1 +
                               level.cols(k, 2 * k - 1) * y_2 +
                               level.cols(2 * k, 2 * k + n - 1) * d;

    for (int i = 0; i < k; i++)
    {
        check_close("prediction agrees", from_var(i), from_vec(i));
    }
}

/// The full shape: unmodelled variables, whose current difference puts a
/// coefficient on x_t and shifts Pi_x onto x_{t-1}, and deterministic terms
/// restricted to the cointegration space, which become ordinary regressors.
void with_exogenous_and_restricted()
{
    std::printf("\nprediction with k=2, p=2, rank=1, m=1, s=1, n=1, n_restricted=1\n");

    const int k = 2, rank = 1, m = 1, n = 1, n_r = 1;
    VarSpec spec;
    spec.k = k;
    spec.p = 2;
    spec.m = m;
    spec.s = 1;
    spec.n = n;
    spec.rank = rank;
    spec.n_restricted = n_r;
    spec.k_beta = k + m + n_r;
    spec.iterations = 1;

    const int k_ect = spec.k_beta;
    const arma::mat alpha = arma::randn<arma::mat>(k, rank);
    const arma::mat beta_mat = arma::randn<arma::mat>(k_ect, rank);
    const arma::mat gamma_1 = arma::randn<arma::mat>(k, k);
    const arma::mat upsilon_0 = arma::randn<arma::mat>(k, m);
    const arma::mat c = arma::randn<arma::mat>(k, n);

    const arma::mat pi = alpha * beta_mat.t();
    const arma::mat pi_y = pi.cols(0, k - 1);
    const arma::mat pi_x = pi.cols(k, k + m - 1);
    const arma::mat pi_d = pi.cols(k + m, k_ect - 1);

    VecNormalWishartDraws draws;
    draws.a = arma::join_cols(
        arma::vectorise(alpha),
        arma::vectorise(arma::join_rows(arma::join_rows(gamma_1, upsilon_0), c)));
    draws.beta = arma::vectorise(beta_mat);
    draws.u_sigma_inv = arma::vectorise(arma::eye<arma::mat>(k, k));

    // s = 1 means one Upsilon block, as p = 2 means one Gamma block.
    check_equal("VEC nparams", draws.a.n_rows,
                static_cast<arma::uword>(k * rank + k * (k * 1 + m * 1 + n)));

    const arma::mat level = level_coefficients(vec_to_var_coefficients(spec, draws).a, k);
    check_equal("level coefficient columns", level.n_cols,
                static_cast<arma::uword>(k * 2 + m * 2 + n + n_r));

    const arma::vec y_1 = arma::randn<arma::vec>(k);
    const arma::vec y_2 = arma::randn<arma::vec>(k);
    const arma::vec x_0 = arma::randn<arma::vec>(m);
    const arma::vec x_1 = arma::randn<arma::vec>(m);
    const arma::vec d = arma::randn<arma::vec>(n);
    const arma::vec d_r = arma::randn<arma::vec>(n_r);

    const arma::vec from_vec = y_1 + pi_y * y_1 + pi_x * x_1 + pi_d * d_r +
                               gamma_1 * (y_1 - y_2) + upsilon_0 * (x_0 - x_1) + c * d;

    // [A_1 A_2 | B_0 B_1 | C | Pi_d]
    arma::uword at = 0;
    const arma::mat a_1 = level.cols(at, at + k - 1);
    at += k;
    const arma::mat a_2 = level.cols(at, at + k - 1);
    at += k;
    const arma::mat b_0 = level.cols(at, at + m - 1);
    at += m;
    const arma::mat b_1 = level.cols(at, at + m - 1);
    at += m;
    const arma::mat c_level = level.cols(at, at + n - 1);
    at += n;
    const arma::mat pi_d_level = level.cols(at, at + n_r - 1);

    const arma::vec from_var =
        a_1 * y_1 + a_2 * y_2 + b_0 * x_0 + b_1 * x_1 + c_level * d + pi_d_level * d_r;

    for (int i = 0; i < k; i++)
    {
        check_close("prediction agrees", from_var(i), from_vec(i));
    }
}

/// p = 1 leaves the VEC with no Gamma at all, and its level form is the single
/// block A_1 = I + Pi. The level order cannot be zero even though the
/// differenced one is, which is the floor vec_to_var_spec() applies.
void no_gamma()
{
    std::printf("\nprediction with k=2, p=1, rank=1 (no Gamma)\n");

    const int k = 2, rank = 1;
    VarSpec spec;
    spec.k = k;
    spec.p = 1;
    spec.rank = rank;
    spec.k_beta = k;
    spec.iterations = 1;

    check_equal("level lag order floored at one",
                static_cast<arma::uword>(vec_to_var_spec(spec).p), 1);

    const arma::mat alpha = arma::randn<arma::mat>(k, rank);
    const arma::mat beta_mat = arma::randn<arma::mat>(k, rank);
    const arma::mat pi = alpha * beta_mat.t();

    VecNormalWishartDraws draws;
    draws.a = arma::vectorise(alpha);
    draws.beta = arma::vectorise(beta_mat);
    draws.u_sigma_inv = arma::vectorise(arma::eye<arma::mat>(k, k));

    const auto out = vec_to_var_coefficients(spec, draws);
    check_equal("one level block", out.a.n_rows, static_cast<arma::uword>(k * k));

    const arma::mat a_1 = arma::reshape(out.a.col(0), k, k);
    const arma::vec y_1 = arma::randn<arma::vec>(k);
    const arma::vec from_vec = y_1 + pi * y_1;
    const arma::vec from_var = a_1 * y_1;

    for (int i = 0; i < k; i++)
    {
        check_close("prediction agrees", from_var(i), from_vec(i));
    }
}

/// A VEC without a cointegration relation still has a level form: the loadings
/// are absent rather than zero, so A_1 picks up the identity alone.
void without_cointegration()
{
    std::printf("\nprediction with k=2, p=2, rank=0\n");

    const int k = 2;
    VarSpec spec;
    spec.k = k;
    spec.p = 2;
    spec.rank = 0;
    spec.iterations = 1;

    const arma::mat gamma_1 = arma::randn<arma::mat>(k, k);

    VecNormalWishartDraws draws;
    draws.a = arma::vectorise(gamma_1);
    draws.u_sigma_inv = arma::vectorise(arma::eye<arma::mat>(k, k));

    const arma::mat level = level_coefficients(vec_to_var_coefficients(spec, draws).a, k);
    check_equal("two level blocks", level.n_cols, static_cast<arma::uword>(2 * k));

    const arma::vec y_1 = arma::randn<arma::vec>(k);
    const arma::vec y_2 = arma::randn<arma::vec>(k);

    const arma::vec from_vec = y_1 + gamma_1 * (y_1 - y_2);
    const arma::vec from_var =
        level.cols(0, k - 1) * y_1 + level.cols(k, 2 * k - 1) * y_2;

    for (int i = 0; i < k; i++)
    {
        check_close("prediction agrees", from_var(i), from_vec(i));
    }
}

/// A structural VEC. A_0 stands to the left of dy_t, hence to the left of y_t as
/// well, and the y_{t-1} that differencing gives back carries it into the first
/// lag: A_1 = A_0 + Pi + Gamma_1. Using the identity there instead is wrong by
/// A_0 - I, which is strictly lower triangular and so leaves the first equation
/// of the model right and every other one wrong -- k = 3 is the smallest size
/// that shows more than one way to be wrong.
void structural()
{
    std::printf("\nprediction with k=3, p=2, rank=1, n=1, structural\n");

    const int k = 3, rank = 1, n = 1;
    const int n_structural = k * (k - 1) / 2;
    VarSpec spec;
    spec.k = k;
    spec.p = 2;
    spec.n = n;
    spec.rank = rank;
    spec.k_beta = k;
    spec.structural = true;
    spec.iterations = 1;

    check_equal("spec.n_structural()", static_cast<arma::uword>(spec.n_structural()),
                static_cast<arma::uword>(n_structural));

    const arma::mat alpha = arma::randn<arma::mat>(k, rank);
    const arma::mat beta_mat = arma::randn<arma::mat>(k, rank);
    const arma::mat gamma_1 = arma::randn<arma::mat>(k, k);
    const arma::mat c = arma::randn<arma::mat>(k, n);
    const arma::mat pi = alpha * beta_mat.t();
    const arma::vec a0_packed = arma::randn<arma::vec>(n_structural);

    // The same matrix the transformation is expected to reconstruct: unit
    // diagonal, the free elements filling the strict lower triangle by column.
    arma::mat a_0 = arma::eye<arma::mat>(k, k);
    a_0(1, 0) = a0_packed(0);
    a_0(2, 0) = a0_packed(1);
    a_0(2, 1) = a0_packed(2);

    VecNormalWishartDraws draws;
    draws.a = arma::join_cols(arma::join_cols(arma::vectorise(alpha),
                                              arma::vectorise(arma::join_rows(gamma_1, c))),
                              a0_packed);
    draws.beta = arma::vectorise(beta_mat);
    draws.u_sigma_inv = arma::vectorise(arma::eye<arma::mat>(k, k));

    const auto out = vec_to_var_coefficients(spec, draws);
    check_equal("level nparams", out.a.n_rows,
                static_cast<arma::uword>(k * (k * 2 + n) + n_structural));

    // The contemporaneous coefficients pass through as they are, still last.
    for (int i = 0; i < n_structural; i++)
    {
        check_close("contemporaneous coefficient unchanged",
                    out.a(out.a.n_rows - n_structural + i, 0), a0_packed(i));
    }

    const arma::mat level =
        arma::reshape(out.a.rows(0, k * (k * 2 + n) - 1), k, k * 2 + n);

    const arma::vec y_1 = arma::randn<arma::vec>(k);
    const arma::vec y_2 = arma::randn<arma::vec>(k);
    const arma::vec d = arma::randn<arma::vec>(n);

    // A_0 dy_t = Pi y_{t-1} + Gamma_1 dy_{t-1} + C d_t against
    // A_0 y_t = A_1 y_{t-1} + A_2 y_{t-2} + C d_t: the same left-hand side, so
    // the right-hand sides have to agree once the level one is shifted back by
    // A_0 y_{t-1}.
    const arma::vec from_vec = a_0 * y_1 + pi * y_1 + gamma_1 * (y_1 - y_2) + c * d;
    const arma::vec from_var = level.cols(0, k - 1) * y_1 +
                               level.cols(k, 2 * k - 1) * y_2 +
                               level.cols(2 * k, 2 * k + n - 1) * d;

    for (int i = 0; i < k; i++)
    {
        check_close("prediction agrees", from_var(i), from_vec(i));
    }

    // Not implied by the above: with the identity in place of A_0 the first
    // equation still agrees, because A_0 - I has nothing in its first row.
    const arma::vec wrong = y_1 + pi * y_1 + gamma_1 * (y_1 - y_2) + c * d;
    check_equal("the identity would not do",
                static_cast<arma::uword>(std::abs(wrong(k - 1) - from_vec(k - 1)) > 1e-8), 1);
}

/// The packing of the contemporaneous coefficients, which only k >= 4 pins
/// down: by column, the order kron(-y, I_k) leaves behind once the diagonal and
/// everything above it is dropped. Read row by row instead -- the order Psi uses
/// -- and the third and fourth elements swap places.
void structural_packing()
{
    std::printf("\npacking of A_0 with k=4\n");

    const int k = 4, n = 1;
    const int n_structural = k * (k - 1) / 2;
    VarSpec spec;
    spec.k = k;
    spec.p = 1;
    spec.n = n;
    spec.structural = true;
    spec.iterations = 1;

    // 1..6, so that a misplaced element is visible as its own index.
    arma::vec a0_packed(n_structural);
    for (int i = 0; i < n_structural; i++)
    {
        a0_packed(i) = i + 1;
    }

    const arma::mat c = arma::zeros<arma::mat>(k, n);

    VecNormalWishartDraws draws;
    draws.a = arma::join_cols(arma::vectorise(c), a0_packed);
    draws.u_sigma_inv = arma::vectorise(arma::eye<arma::mat>(k, k));

    // p = 1 and no cointegration relation, so A_1 is A_0 alone.
    const arma::mat a_1 =
        arma::reshape(vec_to_var_coefficients(spec, draws).a.rows(0, k * k - 1), k, k);

    check_close("A_0(1,0)", a_1(1, 0), 1);
    check_close("A_0(2,0)", a_1(2, 0), 2);
    check_close("A_0(3,0)", a_1(3, 0), 3);
    check_close("A_0(2,1)", a_1(2, 1), 4);
    check_close("A_0(3,1)", a_1(3, 1), 5);
    check_close("A_0(3,2)", a_1(3, 2), 6);
    check_close("unit diagonal", a_1(2, 2), 1);
    check_close("nothing above the diagonal", a_1(0, 3), 0);
}

} // namespace

int main()
{
    // The data is random but the assertions are exact identities, so the seed
    // only decides which numbers a failure reports. Pinned so that it reports
    // the same ones twice.
    arma::arma_rng::set_seed(20260815);

    fixture_shape();
    endogenous_only();
    with_exogenous_and_restricted();
    no_gamma();
    without_cointegration();
    structural();
    structural_packing();

    std::printf("\n%s\n", failures == 0 ? "all checks passed"
                                        : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
