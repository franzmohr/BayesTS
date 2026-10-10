// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/uc_normal_wishart_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/uc_io_common.h"

namespace bayests::hdf5_io::uc_normal_wishart
{

UcNormalWishartInput read_input(const ModelFile &file)
{
    UcNormalWishartInput input;

    // No covariance block in this model -- the Wishart leaves the shock
    // covariance unrestricted -- so the error specification says nothing it
    // needs to know.
    input.spec = read_spec(file, nullptr);
    uc_common::read_input(file, input, "UcNormalWishart");
    uc_common::read_extension(file, input);

    if (file.exist("/priors/u_sigma/df"))
    {
        input.u_sigma_prior.df = get_dataset_int(file, "/priors/u_sigma/df");
    }
    read_mat_if_present(file, "/priors/u_sigma/scale", input.u_sigma_prior.scale);
    read_mat_if_present(file, "/initial/u_sigma_inv", input.initial.u_sigma_inv);

    return input;
}

UcNormalWishartDraws read_coefficients(const ModelFile &file)
{
    UcNormalWishartDraws draws;
    uc_common::read_coefficients(file, draws);
    read_draws_if_present(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    read_draws_if_present(file, "/posterior/cycle0/coeffs", draws.cycle0);
    uc_common::read_extension_draws(file, draws);
    return draws;
}

void write_coefficients(const ModelFile &file, const UcNormalWishartDraws &draws)
{
    ensure_group(file, "/posterior");
    uc_common::write_coefficients(file, draws);
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    // The cycle before the sample, under /model/cycle_init stationary only.
    if (draws.cycle0.n_elem > 0)
    {
        write_draws(file, "/posterior/cycle0/coeffs", draws.cycle0);
    }
    uc_common::write_extension_draws(file, draws);
}

} // namespace bayests::hdf5_io::uc_normal_wishart
