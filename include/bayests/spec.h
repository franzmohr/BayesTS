// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_SPEC_H
#define BAYESTS_SPEC_H

#include <string>

namespace bayests
{

/// Which variable selection scheme, if any, the sampler applies to the
/// coefficients in `a`.
enum class VarSelection
{
    none, ///< All regressors stay in the model.
    ssvs, ///< Stochastic search variable selection.
    bvs   ///< Bayesian variable selection.
};

/// Parses the spelling used on disk and at the R level. Throws
/// std::invalid_argument on anything else, rather than silently disabling
/// selection because a name was misspelled.
VarSelection var_selection_from_string(const std::string &name);

/// Inverse of var_selection_from_string.
const char *to_string(VarSelection selection);

/// The shape of the model, independent of the data and the priors.
struct VarSpec
{
    int k = 0; ///< Endogenous variables.
    int p = 0; ///< Lags of the endogenous variables.
    int m = 0; ///< Exogenous variables.
    int s = 0; ///< Lags of the exogenous variables.
    int n = 0; ///< Deterministic terms.
    int h = 0; ///< Forecast horizon; 0 when no forecast was requested.

    int iterations = 0; ///< Draws kept.
    int burnin = 0;     ///< Draws discarded before the first kept one.

    VarSelection varsel = VarSelection::none;

    /// Whether the error specification asks for a covariance block -- the
    /// "+covar" suffix the files carry. Which prefix it had, "gamma" or "sv",
    /// identifies the model and so is the reader's business, not the sampler's:
    /// by the time a sampler sees this, it already knows which one it is.
    bool covar = false;

    /// Structural model: the last k(k-1)/2 entries of `a` are contemporaneous
    /// coefficients rather than lag or deterministic terms, and are split off
    /// before a forecast path is simulated.
    bool structural = false;

    /// Total length of the chain.
    int draws() const { return iterations + burnin; }

    bool uses_varsel() const { return varsel != VarSelection::none; }

    /// Whether the model carries a Psi block. One variable has no off-diagonal
    /// covariance to model, so the flag on its own is not enough.
    bool uses_covar() const { return covar && k > 1; }

    /// Free elements of the lower triangle of Psi: k(k-1)/2, and zero unless
    /// the model has a covariance block.
    int n_psi() const { return uses_covar() ? k * (k - 1) / 2 : 0; }

    /// Contemporaneous coefficients carried at the end of `a`.
    int n_structural() const { return structural ? k * (k - 1) / 2 : 0; }

    /// Coefficients that are not contemporaneous terms. Declared from the
    /// model's dimensions rather than counted off `z`, because a forecast runs
    /// from the posterior alone and has no `z` to count.
    int n_non_structural() const { return k * (k * p + m * (s + 1) + n); }

    /// Coefficients a draw carries for one period, contemporaneous terms
    /// included. This is the width a stored coefficient path is cut into, so a
    /// host slicing one needs exactly this number and cannot recover it from
    /// the stored width alone.
    int nparams_per_period() const { return n_non_structural() + n_structural(); }

    /// Throws std::invalid_argument if the counts cannot describe a model.
    /// Cheap, and the message is far better than the failure it prevents --
    /// a zero k turns the first reshape into a division by zero.
    void validate() const;
};

} // namespace bayests

#endif // BAYESTS_SPEC_H
