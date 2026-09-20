// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VarTvpDiscount estimator.
//
// The numerics live in src/core/models/var_tvp_discount.cpp and know nothing
// about files. What is left here is the part that is specific to driving the
// model from a file on disk: deciding whether the work has already been done,
// reading the input, and putting the results back.
//
// Three things differ from the eighteen samplers beside it, and all three come
// from the same fact -- this model has an answer rather than a chain:
//
// - The stage that would draw the coefficients *estimates* them. It is still
//   `bayests coefficients`, and what it writes is still the model's parameters,
//   so the command line needs no new verb; but it consumes no random numbers,
//   so `/model/seed` has nothing to repeat and two runs agree to the bit.
// - `/posterior/a/mean` is what says the model has been run, since there is no
//   `/posterior/u_sigma_inv/coeffs` to probe.
// - The log likelihood stage re-runs the filter rather than reading a stored
//   score. Nothing is lost by it: the filter is one deterministic pass over the
//   sample, and the alternative is keeping the same numbers in two datasets
//   that can then disagree.

#include "models/model_check.h"
#include "models/models.h"

#include "bayests/var_tvp_discount.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/model_io_common.h"
#include "io/hdf5/var_tvp_discount_io.h"
#include "reporters/console_reporter.h"

#include <iostream>
#include <stdexcept>

namespace
{
namespace io = bayests::hdf5_io::var_tvp_discount;

/// How many i.i.d. draws a stage that needs them takes. `iterations` keeps its
/// name and loses its chain: there is no sweep, so the number is simply how
/// many draws of the posterior to take.
arma::uword forecast_draws(const bayests::VarSpec &spec)
{
    if (spec.iterations <= 0)
    {
        throw std::runtime_error(
            "a forecast of the discounted model is drawn from its posterior, and iterations "
            "says how many paths to draw; it must be positive");
    }
    return static_cast<arma::uword>(spec.iterations);
}
} // namespace

VarTvpDiscount::VarTvpDiscount()
{
}

VarTvpDiscount::~VarTvpDiscount()
{
}

void VarTvpDiscount::draw_coefficients(const ModelLocation &location_arg)
{
    this->location = location_arg;

    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    if (dataset_has_data(file, io::kPosteriorProbe))
    {
        std::cout << "Posterior data already exists in file. Skipping estimation." << std::endl;
        return;
    }

    const bayests::VarTvpDiscountInput input = io::read_input(file);

    bayests::ConsoleReporter reporter;
    const bayests::VarTvpDiscountPosterior posterior =
        bayests::VarTvpDiscountEstimator{}.estimate(input, reporter);

    io::write_posterior(file, posterior);
}

void VarTvpDiscount::forecast(const ModelLocation &location_arg)
{
    this->location = location_arg;

    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    if (!dataset_has_data(file, io::kPosteriorProbe))
    {
        throw std::runtime_error("The estimated posterior is missing.");
    }

    // No horizon, so no forecast was asked for: a skip, not a failure.
    if (bayests::hdf5_io::optional_attribute_int(file, "/model", "h", 0) <= 0)
    {
        return;
    }

    const bool scorable = dataset_has_data(file, "/data/test/y");
    const bool scored = dataset_has_data(file, "/posterior/forecast/loglik");
    if (dataset_has_data(file, "/posterior/forecast/forecasts") && (!scorable || scored))
    {
        return;
    }

    const bayests::VarTvpDiscountInput input = io::read_input(file);
    const bayests::VarTvpDiscountPosterior posterior = io::read_posterior(file);
    const arma::uword draws = forecast_draws(input.spec);

    bayests::NullReporter reporter;
    if (!dataset_has_data(file, "/posterior/forecast/forecasts"))
    {
        const bayests::ForecastDraws fcst =
            bayests::VarTvpDiscountEstimator{}.forecast(input, posterior, draws, reporter);
        bayests::hdf5_io::write_forecast(file, fcst);
    }

    // One row rather than `draws` of them: the score past the end of the sample
    // is exact here, the filter carrying itself through the realised values, so
    // there is nothing to average over. See
    // VarTvpDiscountEstimator::predictive_log_density().
    if (scorable && !scored)
    {
        const arma::mat score =
            bayests::VarTvpDiscountEstimator{}.predictive_log_density(input, posterior);
        bayests::hdf5_io::write_forecast_loglik(file, score);
    }
}

void VarTvpDiscount::log_likelihood(const ModelLocation &location_arg)
{
    this->location = location_arg;

    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    if (dataset_has_data(file, "/posterior/loglik"))
    {
        return;
    }

    if (!dataset_has_data(file, io::kPosteriorProbe))
    {
        throw std::runtime_error("The estimated posterior is missing.");
    }

    // Re-run rather than read back: the pointwise score is what the filter
    // produces on its way through, it costs one pass, and it consumes no random
    // numbers, so this is the same arithmetic on the same input and not a
    // second estimate of it.
    const bayests::VarTvpDiscountInput input = io::read_input(file);

    bayests::NullReporter reporter;
    const bayests::VarTvpDiscountPosterior posterior =
        bayests::VarTvpDiscountEstimator{}.estimate(input, reporter);

    const arma::mat loglik =
        bayests::VarTvpDiscountEstimator{}.log_likelihood(input, posterior);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}

ModelCheck VarTvpDiscount::check(const ModelLocation &location_arg)
{
    return check_var_discount_model(
        location_arg, [](const ModelFile &file) { return io::read_input(file); },
        io::kPosteriorProbe);
}
