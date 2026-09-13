// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That validate() refuses a cointegration prior matrix which is not symmetric,
// and accepts one which is symmetric only up to rounding.
//
// Nothing downstream of validate() would refuse one. The samplers read these
// matrices partly through one triangle and partly whole, so an asymmetric one
// runs, exits 0, and stands for a prior that is neither the matrix nor its
// transpose. The matrices, and the VECs that read each:
//
//   /priors/beta/p_tau_inv   the four constant VECs
//   /priors/beta/v_inv       the three time-varying VECs, on the state before
//                            the sample
//   /priors/beta/p_tau       the three time-varying VECs, the transition
//
// For each, on every model that reads it: a symmetric one is accepted; one a
// few ulps off and one 1e-10 off -- what an outer product or a general LU
// inverse can leave behind -- are accepted too; and one 1e-6 off and one 0.1
// off are refused with a message that names the matrix and says it must be
// symmetric. Every model rather than one per block, so that a validate() which
// stops going through the shared block fails here.
//
// No draw and no RNG, validate() alone. The tolerance, and why it is what it
// is, is at require_symmetric() in src/core/inputs.cpp.

#include "bayests/inputs.h"

#include <iostream>
#include <limits>
#include <string>

namespace
{

constexpr int kK = 3;           // endogenous variables
constexpr int kP = 2;           // level lags, so one lagged difference
constexpr int kRank = 1;        // cointegration rank
constexpr int kNRestricted = 1; // a constant inside the cointegration space
constexpr int kTT = 12;         // periods
constexpr int kKBeta = kK + kNRestricted;
constexpr int kNAlpha = kK * kRank;
constexpr int kNBeta = kKBeta * kRank;
constexpr int kNX = kK * (kP - 1);      // compact regressors besides the ect
constexpr int kNA = kNAlpha + kK * kNX; // coefficients in `a`

constexpr double kRho = 0.99;

int failures = 0;

void check(const std::string &label, bool condition, const std::string &detail = "")
{
    failures += condition ? 0 : 1;
    std::cout << (condition ? "  ok   " : "  FAIL ") << label << "\n";
    if (!detail.empty())
    {
        std::cout << "         " << detail << "\n";
    }
}

bayests::VarSpec make_spec()
{
    bayests::VarSpec spec;
    spec.k = kK;
    spec.p = kP;
    spec.n = 0;
    spec.n_restricted = kNRestricted;
    spec.rank = kRank;
    spec.k_beta = kKBeta;
    spec.iterations = 10;
    spec.burnin = 5;
    return spec;
}

/// Data of the right shape. validate() reads shapes, not values. `compact` is
/// VecKlgs2010's layout, the regressors in `x`; the others take them in SUR form
/// in `z`.
bayests::TrainData make_train(bool compact)
{
    bayests::TrainData train;
    train.y = arma::mat(kTT, kK, arma::fill::ones);
    train.w = arma::mat(kTT, kKBeta, arma::fill::ones);
    if (compact)
    {
        train.x = arma::mat(kTT, kNX, arma::fill::ones);
    }
    else
    {
        train.z = arma::mat(kTT * kK, kNA, arma::fill::ones);
    }
    return train;
}

arma::vec initial_beta()
{
    arma::vec beta(kNBeta, arma::fill::zeros);
    beta(0) = 1.0;
    beta(1) = -0.5;
    return beta;
}

/// H H' + c (I - H H') with H the direction the chain starts in: dense, so every
/// off-diagonal element has a mirror image worth comparing, and symmetric to the
/// bit. c = tau in (0, 1] is Koop, Leon-Gonzalez and Strachan's P_tau, and
/// c = 1 / tau its inverse.
arma::mat koop_matrix(double c)
{
    const arma::vec h = arma::normalise(initial_beta());
    const arma::mat along = h * arma::trans(h);
    return along + c * (arma::eye<arma::mat>(kKBeta, kKBeta) - along);
}

/// `m` with element (1, 0) moved by `relative` times its largest absolute
/// element. The lower triangle, so that the eigenvalues P_tau is also checked
/// for -- computed from the upper one -- do not move with it, and the message
/// can only be about symmetry.
arma::mat perturbed(arma::mat m, double relative)
{
    m(1, 0) += relative * arma::abs(m).max();
    return m;
}

template <class Input>
void fill_constant_vec(Input &in, bool compact)
{
    in.spec = make_spec();
    in.train = make_train(compact);

    in.a_prior.mu = arma::vec(kNA, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(kNA, kNA);
    in.beta_prior.v_inv = 0.1;
    in.beta_prior.p_tau_inv = arma::eye<arma::mat>(kKBeta, kKBeta);

    in.initial.a = arma::vec(kNA, arma::fill::zeros);
    in.initial.beta = initial_beta();
}

template <class Input>
void fill_tvp_vec(Input &in)
{
    in.spec = make_spec();
    in.train = make_train(false);

    in.a_prior.sigma.shape = arma::vec(kNA, arma::fill::value(3.0));
    in.a_prior.sigma.rate = arma::vec(kNA, arma::fill::value(1e-4));
    in.a_prior.initial_state.mu = arma::vec(kNA, arma::fill::zeros);
    in.a_prior.initial_state.v_inv = arma::eye<arma::mat>(kNA, kNA);

    in.beta_prior.rho = kRho;
    in.beta_prior.initial_state.mu = arma::vec(kNBeta, arma::fill::zeros);
    in.beta_prior.initial_state.v_inv = arma::eye<arma::mat>(kNBeta, kNBeta);

    in.initial.a = arma::mat(kNA, kTT, arma::fill::zeros);
    in.initial.a_sigma_inv = arma::eye<arma::mat>(kNA, kNA);
    in.initial.a_init = arma::vec(kNA, arma::fill::zeros);
    in.initial.beta = arma::repmat(initial_beta(), 1, kTT);
    in.initial.beta_init = initial_beta();
}

template <class Input>
void fill_wishart(Input &in)
{
    in.u_sigma_prior.df = kK;
    in.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
}

template <class Input>
void fill_gamma_prior(Input &in)
{
    in.u_sigma_prior.shape = arma::vec(kK, arma::fill::value(3.0));
    in.u_sigma_prior.rate = arma::vec(kK, arma::fill::value(2.0));
}

template <class Input>
void fill_stochvol(Input &in)
{
    in.u_sigma_prior.offset = arma::vec(kK, arma::fill::value(1e-4));
    in.u_sigma_prior.state.sigma.shape = arma::vec(kK, arma::fill::value(3.0));
    in.u_sigma_prior.state.sigma.rate = arma::vec(kK, arma::fill::value(0.1));
    in.u_sigma_prior.state.initial_state.mu = arma::vec(kK, arma::fill::zeros);
    in.u_sigma_prior.state.initial_state.v_inv = arma::eye<arma::mat>(kK, kK);

    in.initial.h = arma::mat(kTT, kK, arma::fill::zeros);
    in.initial.h_init = arma::vec(kK, arma::fill::zeros);
    in.initial.h_sigma = arma::vec(kK, arma::fill::value(0.1));
}

/// What validate() says about `in`: empty when it accepts it.
template <class Input>
std::string refusal(const Input &in)
{
    try
    {
        in.validate();
    }
    catch (const std::exception &e)
    {
        return e.what();
    }
    return "";
}

/// The five cases above for one matrix on one model. `set` puts a candidate
/// into a copy of `base`; `names` is what the message has to mention for a user
/// to know which dataset it is about.
template <class Input, class Set>
void expect_symmetry_required(const std::string &title, const Input &base, Set set,
                              const arma::mat &symmetric, const std::string &names)
{
    std::cout << title << ":\n";

    const auto verdict = [&](const arma::mat &m) {
        Input in = base;
        set(in, m);
        return refusal(in);
    };

    const std::string exact = verdict(symmetric);
    check("accepts a symmetric one", exact.empty(), exact);

    const struct
    {
        const char *label;
        double relative;
    } rounding[] = {{"accepts one a few ulps off", 16 * std::numeric_limits<double>::epsilon()},
                    {"accepts one 1e-10 off", 1e-10}};

    for (const auto &c : rounding)
    {
        const arma::mat m = perturbed(symmetric, c.relative);
        const std::string message = verdict(m);
        // Guards the test rather than the code: a perturbation lost to rounding
        // would pass this without ever reaching the check.
        const bool really_asymmetric = arma::abs(m - arma::trans(m)).max() > 0.0;
        check(c.label, message.empty() && really_asymmetric,
              really_asymmetric ? message : "the perturbation did not survive rounding");
    }

    const struct
    {
        const char *label;
        double relative;
    } mistakes[] = {{"refuses one 1e-6 off", 1e-6}, {"refuses one 0.1 off", 0.1}};

    for (const auto &c : mistakes)
    {
        const std::string message = verdict(perturbed(symmetric, c.relative));
        check(c.label,
              message.find("must be symmetric") != std::string::npos &&
                  message.find(names) != std::string::npos,
              message.empty() ? "validate() accepted it" : message);
    }
}

} // namespace

int main()
{
    // Dense and far from the identity, so that the relative tolerance is being
    // measured against something other than a one. v_inv is the stationary
    // precision the paper puts on the state before the sample, scaled by
    // (1 - rho^2); P_tau's eigenvalues are 1 and tau = 0.5.
    const arma::mat p_tau_inv = koop_matrix(2.0);
    const arma::mat beta0_v_inv = (1.0 - kRho * kRho) * koop_matrix(2.0);
    const arma::mat p_tau = koop_matrix(0.5);

    const auto set_p_tau_inv = [](auto &in, const arma::mat &m) { in.beta_prior.p_tau_inv = m; };
    const auto set_beta0_v_inv = [](auto &in, const arma::mat &m) {
        in.beta_prior.initial_state.v_inv = m;
    };
    const auto set_p_tau = [](auto &in, const arma::mat &m) { in.beta_prior.p_tau = m; };

    bayests::VecNormalWishartInput normal_wishart;
    fill_constant_vec(normal_wishart, false);
    fill_wishart(normal_wishart);

    bayests::VecKlgs2010Input klgs;
    fill_constant_vec(klgs, true);
    fill_wishart(klgs);

    bayests::VecNormalGammaInput normal_gamma;
    fill_constant_vec(normal_gamma, false);
    fill_gamma_prior(normal_gamma);
    normal_gamma.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);

    bayests::VecNormalStochvolInput normal_stochvol;
    fill_constant_vec(normal_stochvol, false);
    fill_stochvol(normal_stochvol);

    bayests::VecTvpWishartInput tvp_wishart;
    fill_tvp_vec(tvp_wishart);
    fill_wishart(tvp_wishart);

    bayests::VecTvpGammaInput tvp_gamma;
    fill_tvp_vec(tvp_gamma);
    fill_gamma_prior(tvp_gamma);
    tvp_gamma.initial.u_omega_inv = arma::eye<arma::mat>(kK, kK);

    bayests::VecTvpStochvolInput tvp_stochvol;
    fill_tvp_vec(tvp_stochvol);
    fill_stochvol(tvp_stochvol);

    const std::string coint_space = "prior precision of the cointegration space";
    expect_symmetry_required("VecNormalWishart, p_tau_inv", normal_wishart, set_p_tau_inv,
                             p_tau_inv, coint_space);
    expect_symmetry_required("VecKlgs2010, p_tau_inv", klgs, set_p_tau_inv, p_tau_inv,
                             coint_space);
    expect_symmetry_required("VecNormalGamma, p_tau_inv", normal_gamma, set_p_tau_inv, p_tau_inv,
                             coint_space);
    expect_symmetry_required("VecNormalStochvol, p_tau_inv", normal_stochvol, set_p_tau_inv,
                             p_tau_inv, coint_space);

    const std::string beta0 = "prior precision of beta before the sample";
    expect_symmetry_required("VecTvpWishart, v_inv of beta", tvp_wishart, set_beta0_v_inv,
                             beta0_v_inv, beta0);
    expect_symmetry_required("VecTvpGamma, v_inv of beta", tvp_gamma, set_beta0_v_inv,
                             beta0_v_inv, beta0);
    expect_symmetry_required("VecTvpStochvol, v_inv of beta", tvp_stochvol, set_beta0_v_inv,
                             beta0_v_inv, beta0);

    const std::string transition = "transition P_tau of the cointegration state equation";
    expect_symmetry_required("VecTvpWishart, p_tau", tvp_wishart, set_p_tau, p_tau, transition);
    expect_symmetry_required("VecTvpGamma, p_tau", tvp_gamma, set_p_tau, p_tau, transition);
    expect_symmetry_required("VecTvpStochvol, p_tau", tvp_stochvol, set_p_tau, p_tau, transition);

    std::cout << (failures == 0 ? "all checks passed\n" : "some checks FAILED\n");
    return failures == 0 ? 0 : 1;
}
