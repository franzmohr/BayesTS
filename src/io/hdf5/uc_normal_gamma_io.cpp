// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/uc_normal_gamma_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/uc_io_common.h"

namespace bayests::hdf5_io::uc_normal_gamma
{

UcNormalGammaInput read_input(const ModelFile &file)
{
    UcNormalGammaInput input;

    // "gamma+covar" turns the covariance block over the 2k shocks on, the same
    // spelling the gamma VARs use for theirs.
    input.spec = read_spec(file, "gamma+covar");
    uc_common::read_input(file, input, "UcNormalGamma");
    uc_common::read_extension(file, input);

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
    uc_common::read_coefficients(file, draws);
    read_draws_if_present(file, "/posterior/psi/coeffs", draws.psi);
    read_draws_if_present(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    read_draws_if_present(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    read_draws_if_present(file, "/posterior/cycle0/coeffs", draws.cycle0);
    uc_common::read_extension_draws(file, draws);
    return draws;
}

void write_coefficients(const ModelFile &file, const UcNormalGammaDraws &draws)
{
    ensure_group(file, "/posterior");
    uc_common::write_coefficients(file, draws);
    if (draws.has_psi())
    {
        write_draws(file, "/posterior/psi/coeffs", draws.psi);
    }
    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    // The cycle before the sample, under /model/cycle_init stationary only.
    if (draws.cycle0.n_elem > 0)
    {
        write_draws(file, "/posterior/cycle0/coeffs", draws.cycle0);
    }
    uc_common::write_extension_draws(file, draws);
}

} // namespace bayests::hdf5_io::uc_normal_gamma
