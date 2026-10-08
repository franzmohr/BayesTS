// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/uc_normal_stochvol_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/uc_io_common.h"

namespace bayests::hdf5_io::uc_normal_stochvol
{

UcNormalStochvolInput read_input(const ModelFile &file)
{
    UcNormalStochvolInput input;

    // "sv+covar" turns the covariance block over the 2k shocks on, the spelling
    // every stochastic volatility model uses for its own.
    input.spec = read_spec(file, "sv+covar");
    uc_common::read_input(file, input, "UcNormalStochvol");

    if (input.use_psi())
    {
        read_normal_block_if_present(file, "psi", input.psi_prior, input.initial.psi);
    }

    // Only draw_coefficients needs these, as for the other stochvol models.
    read_vec_if_present(file, "/priors/u_sigma/offset", input.u_sigma_prior.offset);
    read_vec_if_present(file, "/priors/u_sigma/shape", input.u_sigma_prior.state.sigma.shape);
    read_vec_if_present(file, "/priors/u_sigma/rate", input.u_sigma_prior.state.sigma.rate);
    read_vec_if_present(file, "/priors/u_sigma/omega_v", input.u_sigma_prior.state.omega_v);
    read_vec_if_present(file, "/priors/u_sigma/mu", input.u_sigma_prior.state.initial_state.mu);
    read_mat_if_present(file, "/priors/u_sigma/v_inv",
                        input.u_sigma_prior.state.initial_state.v_inv);

    // A state the sampler redraws every iteration, even though the file keeps
    // it next to the prior it is drawn under.
    read_vec_if_present(file, "/priors/u_sigma/sigma", input.initial.h_sigma);
    read_mat_if_present(file, "/initial/h", input.initial.h);
    read_vec_if_present(file, "/initial/h_init", input.initial.h_init);

    return input;
}

UcNormalStochvolDraws read_coefficients(const ModelFile &file)
{
    UcNormalStochvolDraws draws;
    uc_common::read_coefficients(file, draws);
    read_draws_if_present(file, "/posterior/psi/coeffs", draws.psi);
    read_draws_if_present(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    read_draws_if_present(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    read_draws_if_present(file, "/posterior/u_sigma_inv/sigma", draws.h_sigma);
    return draws;
}

void write_coefficients(const ModelFile &file, const UcNormalStochvolDraws &draws)
{
    ensure_group(file, "/posterior");
    uc_common::write_coefficients(file, draws);
    if (draws.has_psi())
    {
        write_draws(file, "/posterior/psi/coeffs", draws.psi);
    }
    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    write_draws(file, "/posterior/u_sigma_inv/sigma", draws.h_sigma);
    write_noncentred(file, "/posterior/u_sigma_inv", draws.h_noncentred);
}

} // namespace bayests::hdf5_io::uc_normal_stochvol
