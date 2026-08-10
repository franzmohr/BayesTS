// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "bayests/inputs.h"

#include <stdexcept>
#include <string>

namespace
{

std::string dims(const arma::mat &m)
{
    return std::to_string(m.n_rows) + "x" + std::to_string(m.n_cols);
}

void require_square(const arma::mat &m, arma::uword side, const char *what)
{
    if (m.n_rows != side || m.n_cols != side)
    {
        throw std::invalid_argument(std::string(what) + " must be " + std::to_string(side) +
                                    "x" + std::to_string(side) + ", got " + dims(m));
    }
}

void require_length(const arma::vec &v, arma::uword n, const char *what)
{
    if (v.n_elem != n)
    {
        throw std::invalid_argument(std::string(what) + " must have " + std::to_string(n) +
                                    " elements, got " + std::to_string(v.n_elem));
    }
}

void require_shape(const arma::mat &m, arma::uword rows, arma::uword cols, const char *what)
{
    if (m.n_rows != rows || m.n_cols != cols)
    {
        throw std::invalid_argument(std::string(what) + " must be " + std::to_string(rows) + "x" +
                                    std::to_string(cols) + ", got " + dims(m));
    }
}

} // namespace

namespace bayests
{
namespace
{

/// Number of periods, after checking that the response divides into them.
/// Every validator starts here: a zero or ragged `y` turns the first reshape
/// into a division by zero or a silently misaligned sample.
arma::uword checked_periods(const VarSpec &spec, const TrainData &train)
{
    spec.validate();

    const arma::uword k = static_cast<arma::uword>(spec.k);

    if (train.y.n_elem == 0)
    {
        throw std::invalid_argument("no training observations");
    }
    if (train.y.n_elem % k != 0)
    {
        throw std::invalid_argument("training observations (" + std::to_string(train.y.n_elem) +
                                    ") are not a multiple of k (" + std::to_string(k) + ")");
    }
    return train.y.n_elem / k;
}

void require_stacked_regressors(const TrainData &train, arma::uword tt, arma::uword k)
{
    if (train.z.n_rows != tt * k)
    {
        throw std::invalid_argument("z must have " + std::to_string(tt * k) +
                                    " rows to match the stacked response, got " +
                                    std::to_string(train.z.n_rows));
    }
}

/// The checks a selection block needs whichever coefficient vector it applies
/// to. `n` is the length of that vector, and the labels name it so the message
/// says whether it was the coefficients or the covariance block that was wrong.
void validate_varsel(const VarSelPrior &prior, const arma::vec &initial_lambda,
                     arma::uword n, VarSelection scheme, const char *block)
{
    const std::string what(block);

    require_length(prior.inprior, n, (what + " prior inclusion probabilities").c_str());
    require_length(initial_lambda, n, (what + " initial inclusion indicators").c_str());

    if (prior.include.n_elem == 0)
    {
        throw std::invalid_argument("variable selection is enabled for " + what +
                                    " but no positions were marked for selection");
    }
    if (prior.include.max() >= n)
    {
        throw std::invalid_argument(what + " variable selection position " +
                                    std::to_string(prior.include.max() + 1) +
                                    " is out of range for " + std::to_string(n) + " elements");
    }
    if (scheme == VarSelection::ssvs)
    {
        require_length(prior.ssvs.tau0, n, (what + " SSVS tau0").c_str());
        require_length(prior.ssvs.tau1, n, (what + " SSVS tau1").c_str());
    }
}

/// The normal prior plus starting value that both the coefficient block and the
/// covariance block of the constant-coefficient models carry.
void validate_normal_block(const NormalPrior &prior, const arma::vec &initial,
                           arma::uword n, const char *block)
{
    const std::string what(block);
    require_length(prior.mu, n, ("prior mean of " + what).c_str());
    require_square(prior.v_inv, n, ("prior precision of " + what).c_str());
    require_length(initial, n, ("initial value of " + what).c_str());
}

} // namespace
} // namespace bayests

namespace bayests
{

void VarNormalWishartInput::validate() const
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = checked_periods(spec, train);
    const arma::uword nparams = train.nparams();

    if (use_a())
    {
        require_stacked_regressors(train, tt, k);
        validate_normal_block(a_prior, initial.a, nparams, "a");

        if (spec.uses_varsel())
        {
            validate_varsel(varsel_prior, initial.a_lambda, nparams, spec.varsel, "a");
        }
    }

    if (u_sigma_prior.df <= 0)
    {
        throw std::invalid_argument("Wishart prior degrees of freedom must be positive");
    }
    require_square(u_sigma_prior.scale, k, "Wishart prior scale");
    require_square(initial.u_sigma_inv, k, "initial error precision");
}

void VarNormalGammaInput::validate() const
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = checked_periods(spec, train);
    const arma::uword nparams = train.nparams();

    if (use_a())
    {
        require_stacked_regressors(train, tt, k);
        validate_normal_block(a_prior, initial.a, nparams, "a");

        if (spec.uses_varsel())
        {
            validate_varsel(a_varsel_prior, initial.a_lambda, nparams, spec.varsel, "a");
        }
    }

    if (use_psi())
    {
        const arma::uword n_psi = static_cast<arma::uword>(spec.n_psi());
        validate_normal_block(psi_prior, initial.psi, n_psi, "psi");

        if (spec.uses_varsel())
        {
            validate_varsel(psi_varsel_prior, initial.psi_lambda, n_psi, spec.varsel, "psi");
        }
    }

    require_length(u_sigma_prior.shape, k, "gamma prior shape of the error precision");
    require_length(u_sigma_prior.rate, k, "gamma prior rate of the error precision");
    require_square(initial.u_sigma_inv, k, "initial error precision");
}

void VarNormalStochvolInput::validate() const
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = checked_periods(spec, train);
    const arma::uword nparams = train.nparams();

    if (spec.varsel == VarSelection::ssvs)
    {
        throw std::invalid_argument("SSVS is not implemented for a stochastic volatility model; "
                                    "expected one of none, bvs");
    }

    if (use_a())
    {
        require_stacked_regressors(train, tt, k);
        validate_normal_block(a_prior, initial.a, nparams, "a");

        if (spec.uses_varsel())
        {
            validate_varsel(a_varsel_prior, initial.a_lambda, nparams, spec.varsel, "a");
        }
    }

    if (use_psi())
    {
        const arma::uword n_psi = static_cast<arma::uword>(spec.n_psi());
        validate_normal_block(psi_prior, initial.psi, n_psi, "psi");

        if (spec.uses_varsel())
        {
            validate_varsel(psi_varsel_prior, initial.psi_lambda, n_psi, spec.varsel, "psi");
        }
    }

    require_length(u_sigma_prior.offset, k, "log-volatility offset");
    require_length(u_sigma_prior.state.sigma.shape, k, "prior shape of the log-volatility variance");
    require_length(u_sigma_prior.state.sigma.rate, k, "prior rate of the log-volatility variance");
    require_length(u_sigma_prior.state.initial_state.mu, k, "prior mean of the initial log-volatility");
    require_square(u_sigma_prior.state.initial_state.v_inv, k,
                   "prior precision of the initial log-volatility");

    require_shape(initial.h, tt, k, "initial log-volatility");
    require_length(initial.h_init, k, "initial value of the log-volatility before the sample");
    require_length(initial.h_sigma, k, "initial variance of the log-volatility innovations");

    // The random walk differences h against its own lag, so a single period
    // leaves nothing to difference.
    if (tt < 2)
    {
        throw std::invalid_argument("a stochastic volatility model needs at least two periods");
    }
}

void VarTvpGammaInput::validate() const
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = checked_periods(spec, train);
    const arma::uword nparams = train.nparams();

    if (spec.varsel == VarSelection::ssvs || psi_varsel == VarSelection::ssvs)
    {
        throw std::invalid_argument("SSVS is not implemented for a time-varying parameter model; "
                                    "expected one of none, bvs");
    }

    // The state equation lags the path against itself, and the smoother is
    // handed columns 1..tt of a t+1 wide result.
    if (tt < 2)
    {
        throw std::invalid_argument("a time-varying parameter model needs at least two periods");
    }

    if (use_a())
    {
        require_stacked_regressors(train, tt, k);
        require_shape(initial.a, nparams, tt, "initial coefficient path");
        require_square(initial.a_sigma_inv, nparams, "initial precision of the coefficient innovations");
        require_length(initial.a_init, nparams, "initial value of a before the sample");

        require_length(a_prior.sigma.shape, nparams, "prior shape of the coefficient innovations");
        require_length(a_prior.sigma.rate, nparams, "prior rate of the coefficient innovations");
        require_length(a_prior.initial_state.mu, nparams, "prior mean of a before the sample");
        require_square(a_prior.initial_state.v_inv, nparams, "prior precision of a before the sample");

        if (spec.uses_varsel())
        {
            validate_varsel(a_varsel_prior, initial.a_lambda, nparams, spec.varsel, "a");
        }
    }

    if (use_psi())
    {
        const arma::uword n_psi = static_cast<arma::uword>(spec.n_psi());
        require_shape(initial.psi, n_psi, tt, "initial psi path");
        require_square(initial.psi_sigma_inv, n_psi, "initial precision of the psi innovations");
        require_length(initial.psi_init, n_psi, "initial value of psi before the sample");

        require_length(psi_prior.sigma.shape, n_psi, "prior shape of the psi innovations");
        require_length(psi_prior.sigma.rate, n_psi, "prior rate of the psi innovations");
        require_length(psi_prior.initial_state.mu, n_psi, "prior mean of psi before the sample");
        require_square(psi_prior.initial_state.v_inv, n_psi, "prior precision of psi before the sample");

        if (uses_psi_varsel())
        {
            validate_varsel(psi_varsel_prior, initial.psi_lambda, n_psi, psi_varsel, "psi");
        }
    }

    require_length(u_sigma_prior.shape, k, "gamma prior shape of the error precision");
    require_length(u_sigma_prior.rate, k, "gamma prior rate of the error precision");
    require_square(initial.u_omega_inv, k, "initial error precision");
}

void VarTvpWishartInput::validate() const
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = checked_periods(spec, train);
    const arma::uword nparams = train.nparams();

    if (spec.varsel == VarSelection::ssvs)
    {
        throw std::invalid_argument("SSVS is not implemented for a time-varying parameter model; "
                                    "expected one of none, bvs");
    }

    // The state equation lags the path against itself, and the smoother is
    // handed columns 1..tt of a t+1 wide result.
    if (tt < 2)
    {
        throw std::invalid_argument("a time-varying parameter model needs at least two periods");
    }

    if (use_a())
    {
        require_stacked_regressors(train, tt, k);
        require_shape(initial.a, nparams, tt, "initial coefficient path");
        require_square(initial.a_sigma_inv, nparams, "initial precision of the coefficient innovations");
        require_length(initial.a_init, nparams, "initial value of a before the sample");

        require_length(a_prior.sigma.shape, nparams, "prior shape of the coefficient innovations");
        require_length(a_prior.sigma.rate, nparams, "prior rate of the coefficient innovations");
        require_length(a_prior.initial_state.mu, nparams, "prior mean of a before the sample");
        require_square(a_prior.initial_state.v_inv, nparams, "prior precision of a before the sample");

        if (spec.uses_varsel())
        {
            validate_varsel(a_varsel_prior, initial.a_lambda, nparams, spec.varsel, "a");
        }
    }

    // Unlike VarTvpGamma this model has no psi block: the covariance is carried
    // by the Wishart precision alone, so there is nothing here to match against
    // spec.n_psi().
    if (u_sigma_prior.df <= 0)
    {
        throw std::invalid_argument("Wishart prior degrees of freedom must be positive");
    }
    require_square(u_sigma_prior.scale, k, "Wishart prior scale");
    require_square(initial.u_sigma_inv, k, "initial error precision");
}


void VarTvpStochvolInput::validate() const
{
    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword tt = checked_periods(spec, train);
    const arma::uword nparams = train.nparams();

    if (spec.varsel == VarSelection::ssvs || psi_varsel == VarSelection::ssvs)
    {
        throw std::invalid_argument("SSVS is not implemented for a time-varying parameter model "
                                    "with stochastic volatility; expected one of none, bvs");
    }

    // The state equation lags the path against itself, and the smoother is
    // handed columns 1..tt of a t+1 wide result.
    if (tt < 2)
    {
        throw std::invalid_argument("a time-varying parameter model needs at least two periods");
    }

    if (use_a())
    {
        require_stacked_regressors(train, tt, k);
        require_shape(initial.a, nparams, tt, "initial coefficient path");
        require_square(initial.a_sigma_inv, nparams, "initial precision of the coefficient innovations");
        require_length(initial.a_init, nparams, "initial value of a before the sample");

        require_length(a_prior.sigma.shape, nparams, "prior shape of the coefficient innovations");
        require_length(a_prior.sigma.rate, nparams, "prior rate of the coefficient innovations");
        require_length(a_prior.initial_state.mu, nparams, "prior mean of a before the sample");
        require_square(a_prior.initial_state.v_inv, nparams, "prior precision of a before the sample");

        if (spec.uses_varsel())
        {
            validate_varsel(a_varsel_prior, initial.a_lambda, nparams, spec.varsel, "a");
        }
    }

    if (use_psi())
    {
        const arma::uword n_psi = static_cast<arma::uword>(spec.n_psi());
        require_shape(initial.psi, n_psi, tt, "initial psi path");
        require_square(initial.psi_sigma_inv, n_psi, "initial precision of the psi innovations");
        require_length(initial.psi_init, n_psi, "initial value of psi before the sample");

        require_length(psi_prior.sigma.shape, n_psi, "prior shape of the psi innovations");
        require_length(psi_prior.sigma.rate, n_psi, "prior rate of the psi innovations");
        require_length(psi_prior.initial_state.mu, n_psi, "prior mean of psi before the sample");
        require_square(psi_prior.initial_state.v_inv, n_psi, "prior precision of psi before the sample");

        if (uses_psi_varsel())
        {
            validate_varsel(psi_varsel_prior, initial.psi_lambda, n_psi, psi_varsel, "psi");
        }
    }

    require_length(u_sigma_prior.offset, k, "offset of the log-volatility measurement equation");
    require_length(u_sigma_prior.state.sigma.shape, k, "prior shape of the log-volatility innovations");
    require_length(u_sigma_prior.state.sigma.rate, k, "prior rate of the log-volatility innovations");
    require_length(u_sigma_prior.state.initial_state.mu, k, "prior mean of the log-volatility before the sample");
    require_square(u_sigma_prior.state.initial_state.v_inv, k, "prior precision of the log-volatility before the sample");

    require_length(initial.h_sigma, k, "initial variance of the log-volatility innovations");
    require_length(initial.h_init, k, "initial log-volatility before the sample");
    require_shape(initial.h, tt, k, "initial log-volatility path");
}

} // namespace bayests
