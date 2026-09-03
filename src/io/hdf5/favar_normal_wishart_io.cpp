// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/favar_normal_wishart_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::favar_normal_wishart
{

FavarNormalWishartInput read_input(const ModelFile &file)
{
    FavarNormalWishartInput input;

    // No psi block in this model, so the error specification says nothing it
    // needs to know: the state innovation covariance is the Wishart precision
    // alone, the arrangement VarTvpWishart is in.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);

    // The observed half of the state. Not `/data/train/z`, and not read into
    // `train.x` either: these are not regressors but the observed block of the
    // state vector, and TrainData keeps them in a member of their own for that
    // reason.
    read_mat_if_present(file, "/data/train/f_obs", input.train.f_obs);

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

    // The one prior that parts company with every DFM: a Wishart on a matrix
    // rather than independent gammas on a diagonal.
    if (file.exist("/priors/v_sigma/df"))
    {
        input.v_sigma_prior.df = get_dataset_int(file, "/priors/v_sigma/df");
    }
    read_mat_if_present(file, "/priors/v_sigma/scale", input.v_sigma_prior.scale);

    // The idiosyncratic precision is diagonal and is stored as the diagonal, as
    // in every DFM; the state innovation precision is a whole n_state square
    // matrix, as in every Wishart VAR. The two starting values therefore read
    // through different functions, and that is the difference a file written
    // against a DFM gets wrong.
    read_vec_if_present(file, "/initial/u_sigma_inv", input.initial.u_sigma_inv);
    read_mat_if_present(file, "/initial/v_sigma_inv", input.initial.v_sigma_inv);

    return input;
}

FavarNormalWishartDraws read_coefficients(const ModelFile &file)
{
    FavarNormalWishartDraws draws;

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

void write_coefficients(const ModelFile &file, const FavarNormalWishartDraws &draws)
{
    ensure_group(file, "/posterior");

    write_draws(file, "/posterior/lambda/coeffs", draws.lambda);

    // The path of the unobserved factors, which is part of the posterior rather
    // than a by-product: nothing downstream can be recomputed from the
    // parameters alone. The observed factors are not written beside them --
    // they are the caller's own input, and a copy per draw would be the largest
    // thing in the file.
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

} // namespace bayests::hdf5_io::favar_normal_wishart
