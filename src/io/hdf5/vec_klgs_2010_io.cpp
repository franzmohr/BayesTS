// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/vec_klgs_2010_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::vec_klgs_2010
{

VecKlgs2010Input read_input(const ModelFile &file)
{
    VecKlgs2010Input input;

    // No covariance block in this model, so the error specification says
    // nothing it needs to know.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/w", input.train.w);
    read_mat_if_present(file, "/data/train/x", input.train.x);
    read_mat_if_present(file, "/data/forecast/z", input.forecast.z);

    if (input.use_a())
    {
        input.a_prior = read_normal_prior(file, "/priors/a");
        input.initial.a = read_vec(file, "/initial/a");
    }

    if (input.use_beta())
    {
        input.beta_prior = read_coint_space_prior_constant(file, "/priors/beta");
        input.initial.beta = read_vec(file, "/initial/beta");
    }

    // Only draw_coefficients needs these. A file that holds nothing but a
    // fitted posterior can still be forecast from, so a missing prior is left
    // for validate() to complain about if it turns out to matter.
    if (file.exist("/priors/u_sigma/df"))
    {
        input.u_sigma_prior.df = get_dataset_int(file, "/priors/u_sigma/df");
    }
    read_mat_if_present(file, "/priors/u_sigma/scale", input.u_sigma_prior.scale);
    read_mat_if_present(file, "/initial/u_sigma_inv", input.initial.u_sigma_inv);

    return input;
}

VecKlgs2010Draws read_coefficients(const ModelFile &file)
{
    VecKlgs2010Draws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        draws.beta = read_draws(file, "/posterior/beta/coeffs");
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

void write_coefficients(const ModelFile &file, const VecKlgs2010Draws &draws)
{
    ensure_group(file, "/posterior");

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
    }

    // The cointegration matrix, at the path read_coefficients() reads it back
    // from. `a` alone does not carry the cointegration relation -- only the
    // loadings on it -- so without this nothing downstream could reconstruct Pi.
    if (draws.has_beta())
    {
        write_draws(file, "/posterior/beta/coeffs", draws.beta);
    }

    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::vec_klgs_2010
