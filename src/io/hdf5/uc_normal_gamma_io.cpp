// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/uc_normal_gamma_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::uc_normal_gamma
{

UcNormalGammaInput read_input(const ModelFile &file)
{
    UcNormalGammaInput input;

    // "gamma+covar" turns the covariance block over the 2k shocks on, the same
    // spelling the gamma VARs use for theirs.
    input.spec = read_spec(file, "gamma+covar");

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_test_observations(file, input.spec, input.test.y);
    read_constraints(file, "/data/train/constraints", input.train.constraints);
    read_constraints(file, "/data/test/constraints", input.test.constraints);
    refuse_constraints(file, "/data/forecast/constraints", "UcNormalGamma");

    // The drift's deterministic terms, in the layout a factor model reads its
    // own from: tt x n under /data/train/x and h x n under /data/forecast/x.
    read_factor_deterministic(file, input.train, input.forecast);
    read_normal_block_if_present(file, "c", input.c_prior, input.initial.c);

    // Only draw_coefficients needs the priors and the starting values. A file
    // that holds nothing but a fitted posterior can still be forecast from, so a
    // missing one is left for validate() to complain about if it turns out to
    // matter.
    if (input.use_a())
    {
        read_normal_block_if_present(file, "a", input.a_prior, input.initial.a);
    }
    read_normal_block_if_present(file, "tau0", input.tau0_prior, input.initial.tau0);
    if (input.use_psi())
    {
        read_normal_block_if_present(file, "psi", input.psi_prior, input.initial.psi);
    }

    if (file.exist("/priors/u_sigma/shape"))
    {
        input.u_sigma_prior = read_gamma_prior(file, "/priors/u_sigma");
    }
    // The diagonal of Omega, a vector of 2k, as the factor models store theirs.
    read_vec_if_present(file, "/initial/u_omega_inv", input.initial.u_omega_inv);

    return input;
}

UcNormalGammaDraws read_coefficients(const ModelFile &file)
{
    UcNormalGammaDraws draws;

    read_draws_if_present(file, "/posterior/trend/coeffs", draws.trend);
    read_draws_if_present(file, "/posterior/a/coeffs", draws.a);
    read_draws_if_present(file, "/posterior/c/coeffs", draws.c);
    read_draws_if_present(file, "/posterior/tau0/coeffs", draws.tau0);
    read_draws_if_present(file, "/posterior/psi/coeffs", draws.psi);
    read_draws_if_present(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    read_draws_if_present(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);

    return draws;
}

void write_coefficients(const ModelFile &file, const UcNormalGammaDraws &draws)
{
    ensure_group(file, "/posterior");

    // The trend path, which is part of the posterior rather than a by-product:
    // the forecast starts from its end, and nothing recovers it from the
    // parameters alone. The cycle is y minus it and is not written.
    write_draws(file, "/posterior/trend/coeffs", draws.trend);

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
    }
    if (draws.has_c())
    {
        write_draws(file, "/posterior/c/coeffs", draws.c);
    }
    write_draws(file, "/posterior/tau0/coeffs", draws.tau0);
    if (draws.has_psi())
    {
        write_draws(file, "/posterior/psi/coeffs", draws.psi);
    }
    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::uc_normal_gamma
