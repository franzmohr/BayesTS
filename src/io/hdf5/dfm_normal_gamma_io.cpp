// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/dfm_normal_gamma_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::dfm_normal_gamma
{

DfmNormalGammaInput read_input(const ModelFile &file)
{
    DfmNormalGammaInput input;

    // No covariance block in this model, so the error specification says
    // nothing it needs to know.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);

    // Only draw_coefficients needs the priors and the starting values. A file
    // that holds nothing but a fitted posterior can still be forecast from, so a
    // missing one is left for validate() to complain about if it turns out to
    // matter.
    if (file.exist("/priors/lambda/mu"))
    {
        input.lambda_prior = read_normal_prior(file, "/priors/lambda");
    }
    read_vec_if_present(file, "/initial/lambda", input.initial.lambda);

    if (input.use_a())
    {
        if (file.exist("/priors/a/mu"))
        {
            input.a_prior = read_normal_prior(file, "/priors/a");
        }
        read_vec_if_present(file, "/initial/a", input.initial.a);
    }

    if (file.exist("/priors/u_sigma/shape"))
    {
        input.u_sigma_prior = read_gamma_prior(file, "/priors/u_sigma");
    }
    if (file.exist("/priors/v_sigma/shape"))
    {
        input.v_sigma_prior = read_gamma_prior(file, "/priors/v_sigma");
    }

    // Both precisions are diagonal and are stored as the diagonal, which is
    // where this parts company with every other model here: the constant
    // variance VARs write /initial/u_sigma_inv as a k x k matrix.
    read_vec_if_present(file, "/initial/u_sigma_inv", input.initial.u_sigma_inv);
    read_vec_if_present(file, "/initial/v_sigma_inv", input.initial.v_sigma_inv);

    return input;
}

DfmNormalGammaDraws read_coefficients(const ModelFile &file)
{
    DfmNormalGammaDraws draws;

    if (dataset_has_data(file, "/posterior/lambda/coeffs"))
    {
        draws.lambda = read_draws(file, "/posterior/lambda/coeffs");
    }
    if (dataset_has_data(file, "/posterior/factors/coeffs"))
    {
        draws.factors = read_draws(file, "/posterior/factors/coeffs");
    }
    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }
    if (dataset_has_data(file, "/posterior/v_sigma_inv/coeffs"))
    {
        draws.v_sigma_inv = read_draws(file, "/posterior/v_sigma_inv/coeffs");
    }

    return draws;
}

void write_coefficients(const ModelFile &file, const DfmNormalGammaDraws &draws)
{
    ensure_group(file, "/posterior");

    write_draws(file, "/posterior/lambda/coeffs", draws.lambda);

    // The factor path, which is part of the posterior rather than a by-product:
    // the factors are unobserved, so nothing downstream -- not the forecast, not
    // the log likelihood -- can be recomputed from the parameters alone.
    if (draws.has_factors())
    {
        write_draws(file, "/posterior/factors/coeffs", draws.factors);
    }

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
    }

    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
    write_draws(file, "/posterior/v_sigma_inv/coeffs", draws.v_sigma_inv);
}

} // namespace bayests::hdf5_io::dfm_normal_gamma
