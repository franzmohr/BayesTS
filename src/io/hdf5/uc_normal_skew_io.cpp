// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/uc_normal_skew_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/uc_io_common.h"

namespace bayests::hdf5_io::uc_normal_skew
{

UcNormalSkewInput read_input(const ModelFile &file)
{
    UcNormalSkewInput input;

    // No covariance block: the skewed shocks are independent by construction.
    input.spec = read_spec(file, nullptr);
    uc_common::read_input(file, input, "UcNormalSkew");

    read_mat_if_present(file, "/data/train/skew_x", input.train_skew_x);
    read_mat_if_present(file, "/data/forecast/skew_x", input.forecast_skew_x);
    read_normal_block_if_present(file, "skew", input.skew_prior, input.initial.skew);

    if (file.exist("/priors/u_sigma/shape"))
    {
        input.u_sigma_prior = read_gamma_prior(file, "/priors/u_sigma");
    }
    read_vec_if_present(file, "/initial/u_omega_inv", input.initial.u_omega_inv);

    return input;
}

UcNormalSkewDraws read_coefficients(const ModelFile &file)
{
    UcNormalSkewDraws draws;
    uc_common::read_coefficients(file, draws);
    read_draws_if_present(file, "/posterior/skew/coeffs", draws.skew);
    read_draws_if_present(file, "/posterior/skew_latent/coeffs", draws.skew_latent);
    read_draws_if_present(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
    return draws;
}

void write_coefficients(const ModelFile &file, const UcNormalSkewDraws &draws)
{
    ensure_group(file, "/posterior");
    uc_common::write_coefficients(file, draws);
    write_draws(file, "/posterior/skew/coeffs", draws.skew);
    // The latents are part of the posterior rather than a by-product: the log
    // likelihood conditions on them.
    write_draws(file, "/posterior/skew_latent/coeffs", draws.skew_latent);
    write_draws(file, "/posterior/u_omega_inv/coeffs", draws.u_omega_inv);
}

} // namespace bayests::hdf5_io::uc_normal_skew
