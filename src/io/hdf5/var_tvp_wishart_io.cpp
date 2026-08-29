// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/var_tvp_wishart_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"

namespace bayests::hdf5_io::var_tvp_wishart
{

VarTvpWishartInput read_input(const ModelFile &file)
{
    VarTvpWishartInput input;

    input.spec = read_spec(file, "wishart");

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/z", input.train.z);
    read_mat_if_present(file, "/data/forecast/z", input.forecast.z);

    const arma::uword tt = input.train.y.n_elem > 0 && input.spec.k > 0
                               ? input.train.periods(input.spec.k)
                               : 0;

    if (input.use_a())
    {
        const arma::uword nparams = input.train.nparams();

        input.initial.a = read_path(file, "/initial/a", nparams, tt);
        input.initial.a_sigma_inv = read_mat(file, "/initial/a_sigma_inv");
        input.initial.a_init = read_vec(file, "/initial/a_init");

        input.a_prior.sigma = read_gamma_prior(file, "/priors/a");
        input.a_prior.initial_state = read_normal_prior(file, "/priors/a");

        // BVS is the only scheme this model implements. An SSVS file is left
        // unread here and rejected by validate(), which can say why.
        if (input.spec.varsel == VarSelection::bvs)
        {
            input.initial.a_lambda = read_vec(file, "/initial/a_lambda");
            input.a_varsel_prior = read_varsel_prior(file, "/priors/a", input.spec.varsel);
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

// `input` is unused here and kept for the uniform signature the front-ends call
// every model's readers through. Unlike VarTvpGamma, whose precision is stored
// per period and has to be read against the spec, this model's u_sigma_inv is
// one k x k matrix per draw, so the dataset is read as it stands.
VarTvpWishartDraws read_loglik_coefficients(const ModelFile &file,
                                            [[maybe_unused]] const VarTvpWishartInput &input)
{
    VarTvpWishartDraws draws;

    if (dataset_has_data(file, "/posterior/a/coeffs"))
    {
        draws.a = read_draws(file, "/posterior/a/coeffs");
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

VarTvpWishartDraws read_forecast_coefficients(const ModelFile &file,
                                              const VarTvpWishartInput &input)
{
    VarTvpWishartDraws draws;

    // The width the path was stored at, taken from the spec as VarTvpGamma and
    // VarTvpStochvol take it -- not from `z.n_cols`, which is what this used to
    // do. A structural model's contemporaneous coefficients sit at the end of
    // every period of `a` and have no column in `z`, so counting off `z` slices
    // the path at the wrong stride: it silently returns a matrix one block
    // narrow, cut across period boundaries.
    const arma::uword nparams = static_cast<arma::uword>(input.spec.nparams_per_period());

    if (nparams > 0 && dataset_has_data(file, "/posterior/a/coeffs"))
    {
        const arma::uword tt = input.train.periods(input.spec.k);
        draws.a = read_draws_at_period(file, "/posterior/a/coeffs", tt - 1, nparams);
    }
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        draws.u_sigma_inv = read_draws(file, "/posterior/u_sigma_inv/coeffs");
    }

    return draws;
}

void write_coefficients(const ModelFile &file, const VarTvpWishartDraws &draws)
{
    ensure_group(file, "/posterior");

    if (draws.has_a())
    {
        write_draws(file, "/posterior/a/coeffs", draws.a);
        write_draws(file, "/posterior/a/sigma", draws.a_sigma);
        if (draws.a_lambda.n_elem > 0)
        {
            write_draws(file, "/posterior/a/lambda", draws.a_lambda);
        }
    }

    write_draws(file, "/posterior/u_sigma_inv/coeffs", draws.u_sigma_inv);
}

} // namespace bayests::hdf5_io::var_tvp_wishart
