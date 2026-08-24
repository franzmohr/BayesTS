// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That validate() rejects a structural model whose error covariance is
// unrestricted, and accepts every combination that is identified.
//
// A unit test rather than a fixture because the golden harness cannot express
// this: it runs a sampler and prints fingerprints, and the BaseModel front-ends
// swallow exceptions, so a rejected input reads there as a test that passed
// while writing nothing. What is asserted here is a property of the input, not
// of a draw -- no RNG, no toolchain sensitivity, the same answer everywhere --
// so it can assert and return non-zero the way the other unit tests do.
//
// The reasoning behind the rule is at require_identified_structural() in
// src/core/inputs.cpp. In short: the data determine A_0^-1 Sigma A_0^-T, and
// A_0 plus an unrestricted Sigma carries k(k-1)/2 parameters more than that.
// Both a Wishart prior and a covariance block leave Sigma unrestricted, so both
// are covered here -- the second being the one easily missed, since Psi is a
// second contemporaneous matrix doing A_0's job.

#include "bayests/inputs.h"

#include <iostream>
#include <string>

namespace
{
constexpr int kK = 3;
constexpr int kTT = 12;

/// A minimal but complete VarNormalGamma input, so that anything thrown is the
/// identification check rather than a missing prior.
bayests::VarNormalGammaInput make_gamma(bool covar, bool structural)
{
    const int n_struct = structural ? kK * (kK - 1) / 2 : 0;
    const int nparams = kK * kK + n_struct;
    const int n_psi = kK * (kK - 1) / 2;

    bayests::VarNormalGammaInput in;
    in.spec.k = kK;
    in.spec.p = 1;
    in.spec.iterations = 10;
    in.spec.burnin = 5;
    in.spec.covar = covar;
    in.spec.structural = structural;

    in.train.y = arma::mat(kTT, kK, arma::fill::ones);
    in.train.z = arma::mat(kTT * kK, nparams, arma::fill::ones);

    in.a_prior.mu = arma::vec(nparams, arma::fill::zeros);
    in.a_prior.v_inv = arma::eye<arma::mat>(nparams, nparams);
    in.initial.a = arma::vec(nparams, arma::fill::zeros);

    in.psi_prior.mu = arma::vec(n_psi, arma::fill::zeros);
    in.psi_prior.v_inv = arma::eye<arma::mat>(n_psi, n_psi);
    in.initial.psi = arma::vec(n_psi, arma::fill::zeros);

    in.u_sigma_prior.shape = arma::vec(kK, arma::fill::value(3.0));
    in.u_sigma_prior.rate = arma::vec(kK, arma::fill::value(2.0));
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return in;
}

/// The same for VarTvpWishart, whose Sigma is unrestricted whatever covar says.
bayests::VarTvpWishartInput make_wishart(bool structural)
{
    const int n_struct = structural ? kK * (kK - 1) / 2 : 0;
    const int nparams = kK * kK + n_struct;

    bayests::VarTvpWishartInput in;
    in.spec.k = kK;
    in.spec.p = 1;
    in.spec.iterations = 10;
    in.spec.burnin = 5;
    in.spec.structural = structural;

    in.train.y = arma::mat(kTT, kK, arma::fill::ones);
    in.train.z = arma::mat(kTT * kK, nparams, arma::fill::ones);

    in.initial.a = arma::mat(nparams, kTT, arma::fill::zeros);
    in.initial.a_sigma_inv = arma::eye<arma::mat>(nparams, nparams);
    in.initial.a_init = arma::vec(nparams, arma::fill::zeros);
    in.a_prior.sigma.shape = arma::vec(nparams, arma::fill::value(3.0));
    in.a_prior.sigma.rate = arma::vec(nparams, arma::fill::value(0.01));
    in.a_prior.initial_state.mu = arma::vec(nparams, arma::fill::zeros);
    in.a_prior.initial_state.v_inv = arma::eye<arma::mat>(nparams, nparams);

    in.u_sigma_prior.df = kK;
    in.u_sigma_prior.scale = arma::eye<arma::mat>(kK, kK);
    in.initial.u_sigma_inv = arma::eye<arma::mat>(kK, kK);
    return in;
}

int failures = 0;

template <class Input>
void expect(const char *label, const Input &in, bool should_throw)
{
    std::string what;
    bool threw = false;
    try
    {
        in.validate();
    }
    catch (const std::exception &e)
    {
        threw = true;
        what = e.what();
    }

    const bool ok = threw == should_throw;
    failures += ok ? 0 : 1;
    std::cout << (ok ? "  ok   " : "  FAIL ") << label << (threw ? " -> rejected" : " -> accepted")
              << "\n";
    if (threw)
    {
        std::cout << "         " << what << "\n";
    }
}

} // namespace

int main()
{
    std::cout << "sound combinations (diagonal Sigma) must be accepted:\n";
    expect("gamma, covar=0, structural=1", make_gamma(false, true), false);
    expect("gamma, covar=1, structural=0", make_gamma(true, false), false);
    expect("wishart,        structural=0", make_wishart(false), false);

    std::cout << "unidentified combinations must be rejected:\n";
    expect("gamma, covar=1, structural=1", make_gamma(true, true), true);
    expect("wishart,        structural=1", make_wishart(true), true);

    std::cout << (failures == 0 ? "all as expected\n" : "SOMETHING IS WRONG\n");
    return failures == 0 ? 0 : 1;
}
