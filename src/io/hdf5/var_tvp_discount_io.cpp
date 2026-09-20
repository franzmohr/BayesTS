// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/var_tvp_discount_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::var_tvp_discount
{

VarTvpDiscountInput read_input(const ModelFile &file)
{
    VarTvpDiscountInput input;

    // No covariance block in this model: the error covariance is the inverse
    // Wishart whole, so the error specification says nothing it needs to know.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/x", input.train.x);
    read_forecast_regressors(file, input.spec.k, input.forecast.x);
    read_test_observations(file, input.spec, input.test.y);

    if (input.use_a())
    {
        input.a_prior.mean = read_mat(file, "/priors/a/mean");
        input.a_prior.cov = read_mat(file, "/priors/a/cov");
    }

    // Only estimate() needs these. A file that holds nothing but a fitted
    // posterior can still be forecast from, so a missing prior is left for
    // validate() to complain about if it turns out to matter.
    if (file.exist("/priors/u_sigma/df"))
    {
        input.u_sigma_prior.df = get_dataset_int(file, "/priors/u_sigma/df");
    }
    read_mat_if_present(file, "/priors/u_sigma/scale", input.u_sigma_prior.scale);

    return input;
}

VarTvpDiscountPosterior read_posterior(const ModelFile &file)
{
    VarTvpDiscountPosterior posterior;

    read_posterior_path_if_present(file, "/posterior/a/mean", posterior.a);
    read_posterior_path_if_present(file, "/posterior/a/scale", posterior.a_scale);
    read_posterior_path_if_present(file, "/posterior/a/cov", posterior.a_cov);
    read_posterior_path_if_present(file, "/posterior/u_sigma/scale", posterior.u_sigma);

    arma::mat df;
    if (read_posterior_path_if_present(file, "/posterior/df", df))
    {
        posterior.df = arma::vectorise(df);
    }

    return posterior;
}

void write_posterior(const ModelFile &file, const VarTvpDiscountPosterior &posterior)
{
    ensure_group(file, "/posterior");
    ensure_group(file, "/posterior/a");
    ensure_group(file, "/posterior/u_sigma");

    write_posterior_path(file, "/posterior/a/mean", posterior.a);
    write_posterior_path(file, "/posterior/a/scale", posterior.a_scale);

    // The regressor side of the coefficient covariance, n_reg squared per
    // period. Written whole rather than reduced to `scale`: the two together
    // are the joint posterior, and a forecast or an impulse response needs the
    // joint, not the marginal bands.
    write_posterior_path(file, "/posterior/a/cov", posterior.a_cov);

    // A covariance, which is why it is not at `/posterior/u_sigma_inv` where
    // every sampler here puts a precision.
    write_posterior_path(file, "/posterior/u_sigma/scale", posterior.u_sigma);

    write_posterior_path(file, "/posterior/df", arma::mat(posterior.df.t()));
}

} // namespace bayests::hdf5_io::var_tvp_discount
