// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "io/hdf5/vec_tvp_discount_io.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/model_io_common.h"

namespace bayests::hdf5_io::vec_tvp_discount
{

VecTvpDiscountInput read_input(const ModelFile &file)
{
    VecTvpDiscountInput input;

    // No covariance block in this model, as for the three Wishart-precision
    // VECs: the error covariance is the inverse Wishart whole.
    input.spec = read_spec(file, nullptr);

    read_mat_if_present(file, "/data/train/y", input.train.y);
    read_mat_if_present(file, "/data/train/w", input.train.w);
    read_mat_if_present(file, "/data/train/x", input.train.x);
    read_forecast_regressors(file, input.spec.k, input.forecast.x);
    read_test_observations(file, input.spec, input.test.y);

    if (input.use_a())
    {
        input.a_prior.mean = read_mat(file, "/priors/a/mean");
        input.a_prior.cov = read_mat(file, "/priors/a/cov");
    }

    if (input.use_beta())
    {
        // Stored as vec of a k_beta x rank matrix, the layout every VEC here
        // stores beta in, and reshaped rather than read as a matrix so that one
        // spelling serves both. See the header for why the path is /initial.
        const arma::vec flat = read_vec(file, "/initial/beta");
        const arma::uword k_beta = static_cast<arma::uword>(input.spec.k_beta);
        const arma::uword rank = static_cast<arma::uword>(input.spec.rank);
        if (flat.n_elem == k_beta * rank && k_beta > 0)
        {
            input.beta = arma::reshape(flat, k_beta, rank);
        }
        else
        {
            // Left in whatever shape the file implies, for validate() to refuse
            // with the message that names both counts. Reshaping a dataset of
            // the wrong length here would pad it with zeros and hand the filter
            // a cointegration matrix the file never described.
            input.beta = flat;
        }
    }

    if (file.exist("/priors/u_sigma/df"))
    {
        input.u_sigma_prior.df = get_dataset_int(file, "/priors/u_sigma/df");
    }
    read_mat_if_present(file, "/priors/u_sigma/scale", input.u_sigma_prior.scale);

    return input;
}

VecTvpDiscountPosterior read_posterior(const ModelFile &file)
{
    VecTvpDiscountPosterior posterior;

    read_posterior_path_if_present(file, "/posterior/a/mean", posterior.a);
    read_posterior_path_if_present(file, "/posterior/a/scale", posterior.a_scale);
    read_posterior_path_if_present(file, "/posterior/a/cov", posterior.a_cov);
    read_posterior_path_if_present(file, "/posterior/u_sigma/scale", posterior.u_sigma);

    arma::mat df;
    if (read_posterior_path_if_present(file, "/posterior/df", df))
    {
        posterior.df = arma::vectorise(df);
    }

    if (dataset_has_data(file, "/posterior/beta/coeffs"))
    {
        posterior.beta = arma::vectorise(read_draws(file, "/posterior/beta/coeffs"));
    }

    return posterior;
}

void write_posterior(const ModelFile &file, const VecTvpDiscountPosterior &posterior)
{
    ensure_group(file, "/posterior");
    ensure_group(file, "/posterior/a");
    ensure_group(file, "/posterior/u_sigma");

    write_posterior_path(file, "/posterior/a/mean", posterior.a);
    write_posterior_path(file, "/posterior/a/scale", posterior.a_scale);
    write_posterior_path(file, "/posterior/a/cov", posterior.a_cov);
    write_posterior_path(file, "/posterior/u_sigma/scale", posterior.u_sigma);
    write_posterior_path(file, "/posterior/df", arma::mat(posterior.df.t()));

    // The space the run conditioned on, at the path read_posterior() reads it
    // back from and every other VEC writes its draws of it to. One column: it
    // did not move and was not estimated.
    if (posterior.has_beta())
    {
        ensure_group(file, "/posterior/beta");
        write_draws(file, "/posterior/beta/coeffs", arma::mat(posterior.beta));
    }
}

} // namespace bayests::hdf5_io::vec_tvp_discount
