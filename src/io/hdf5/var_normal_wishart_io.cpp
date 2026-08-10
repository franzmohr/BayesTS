// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/var_normal_wishart_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::var_normal_wishart
{

VarNormalWishartInput read_input(const HighFive::File &file)
{
    VarNormalWishartInput input;

    // No covariance block in this model, so the error specification says
    // nothing it needs to know.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/z", input.train.z);
    read_mat_if_present(file, "/data/forecast/z", input.forecast.z);

    if (input.use_a())
    {
        input.a_prior = read_normal_prior(file, "/priors/a");
        input.initial.a = read_vec(file, "/initial/a");

        if (input.spec.uses_varsel())
        {
            input.initial.a_lambda = read_vec(file, "/initial/a_lambda");
            input.varsel_prior = read_varsel_prior(file, "/priors/a", input.spec.varsel);
        }
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

VarNormalWishartDraws read_coefficients(const HighFive::File &file)
{
    VarNormalWishartDraws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/a/lambda"))
    {
        draws.a_lambda = read_draws(file, "/posterior/a/lambda");
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

void write_coefficients(HighFive::File &file, const VarNormalWishartDraws &draws)
{
    ensure_group(file, "/posterior");

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
        if (draws.has_lambda())
        {
            write_draws(file, "/posterior/a/lambda", draws.a_lambda);
        }
    }
    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::var_normal_wishart
