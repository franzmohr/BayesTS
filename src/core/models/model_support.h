// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CORE_MODELS_MODEL_SUPPORT_H
#define BAYESTS_CORE_MODELS_MODEL_SUPPORT_H

#include "bayests/data.h"

namespace bayests::core
{

/// The response the samplers actually work with: the observations stacked
/// period by period, vec(y'). Storing `y` period-per-row and stacking here
/// keeps the caller's matrix in the orientation everyone else writes it in.
inline arma::vec stacked_response(const TrainData &train)
{
    return arma::vectorise(arma::trans(train.y));
}

/// One draw from the normal posterior written in precision form: returns a
/// sample from N(V^-1 b, V^-1), where `precision` is V and `rhs` is b, the
/// precision-weighted mean term.
///
/// Both the posterior mean and the draw need V factorised, and the obvious
/// spelling factorises it twice:
///
///     mu = arma::solve(V, b);                              // LU,   2n^3/3
///     x  = mu + arma::solve(arma::chol(V), arma::randn(n)); // chol,  n^3/3
///
/// `arma::solve()` on a general square matrix runs an LU with a reciprocal
/// condition estimate; it does not detect that V is symmetric positive
/// definite. Taking one Cholesky and reusing it for both the mean and the draw
/// costs n^3/3 in total -- a third of the work -- with the three triangular
/// solves being O(n^2) each.
///
/// V must be symmetric positive definite, which every posterior precision here
/// is by construction (prior precision plus a Gram matrix). `arma::chol()`
/// throws if it is not, which fails louder than the LU path did: an indefinite
/// V used to survive the mean solve and only fail on the draw.
inline arma::vec draw_normal_precision(const arma::mat &precision, const arma::vec &rhs)
{
    // precision = r.t() * r, with r upper triangular.
    const arma::mat r = arma::chol(precision);

    // mean = precision^-1 rhs, by forward then back substitution.
    const arma::vec mean = arma::solve(arma::trimatu(r),
                                       arma::solve(arma::trimatl(r.t()), rhs));

    // Cov(r^-1 z) = r^-1 r^-T = (r.t() * r)^-1 = precision^-1.
    return mean + arma::solve(arma::trimatu(r), arma::randn<arma::vec>(precision.n_rows));
}

} // namespace bayests::core

#endif // BAYESTS_CORE_MODELS_MODEL_SUPPORT_H
