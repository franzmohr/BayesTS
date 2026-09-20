// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VecTvpDiscount estimator.
//
// VarTvpDiscount.cpp says what the three stages of a model with an answer
// rather than a chain look like, and all of it holds here. Two things are this
// model's own:
//
// - The forecast runs in levels, as every VEC's does, so `/data/forecast/x` is
//   in the level layout and not the differenced one `/data/train/x` uses.
// - The score past the end of the sample is drawn rather than exact. The VAR
//   carries its filter through the realised values; a VEC cannot, because the
//   error correction term of a scored period is not recoverable from the level
//   regressors -- see VecTvpDiscountEstimator::predictive_log_density().

#include "models/model_check.h"
#include "models/models.h"

#include "bayests/vec_tvp_discount.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/model_io_common.h"
#include "io/hdf5/vec_tvp_discount_io.h"
#include "reporters/console_reporter.h"

#include <iostream>
#include <stdexcept>

namespace
{
namespace io = bayests::hdf5_io::vec_tvp_discount;

/// How many i.i.d. draws a stage that needs them takes; see VarTvpDiscount.cpp.
arma::uword posterior_draws(const bayests::VarSpec &spec)
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

VecTvpDiscount::VecTvpDiscount()
{
}

VecTvpDiscount::~VecTvpDiscount()
{
}

void VecTvpDiscount::draw_coefficients(const ModelLocation &location_arg)
{
    this->location = location_arg;

    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    if (dataset_has_data(file, io::kPosteriorProbe))
    {
        std::cout << "Posterior data already exists in file. Skipping estimation." << std::endl;
        return;
    }

    const bayests::VecTvpDiscountInput input = io::read_input(file);

    bayests::ConsoleReporter reporter;
    const bayests::VecTvpDiscountPosterior posterior =
        bayests::VecTvpDiscountEstimator{}.estimate(input, reporter);

    io::write_posterior(file, posterior);
}

void VecTvpDiscount::forecast(const ModelLocation &location_arg)
{
    this->location = location_arg;

    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    if (!dataset_has_data(file, io::kPosteriorProbe))
    {
        throw std::runtime_error("The estimated posterior is missing.");
    }

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

    const bayests::VecTvpDiscountInput input = io::read_input(file);
    const bayests::VecTvpDiscountPosterior posterior = io::read_posterior(file);
    const arma::uword draws = posterior_draws(input.spec);

    bayests::NullReporter reporter;
    if (!dataset_has_data(file, "/posterior/forecast/forecasts"))
    {
        const bayests::ForecastDraws fcst =
            bayests::VecTvpDiscountEstimator{}.forecast(input, posterior, draws, reporter);
        bayests::hdf5_io::write_forecast(file, fcst);
    }

    // Scored in the level parameterisation the forecast runs in, against the
    // realised levels /data/test/y holds.
    if (scorable && !scored)
    {
        const arma::mat score = bayests::VecTvpDiscountEstimator{}.predictive_log_density(
            input, posterior, draws);
        bayests::hdf5_io::write_forecast_loglik(file, score);
    }
}

void VecTvpDiscount::log_likelihood(const ModelLocation &location_arg)
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

    // Re-run rather than read back; see VarTvpDiscount.cpp for why that is the
    // same arithmetic and not a second estimate.
    const bayests::VecTvpDiscountInput input = io::read_input(file);

    bayests::NullReporter reporter;
    const bayests::VecTvpDiscountPosterior posterior =
        bayests::VecTvpDiscountEstimator{}.estimate(input, reporter);

    const arma::mat loglik =
        bayests::VecTvpDiscountEstimator{}.log_likelihood(input, posterior);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}

ModelCheck VecTvpDiscount::check(const ModelLocation &location_arg)
{
    return check_vec_discount_model(
        location_arg, [](const ModelFile &file) { return io::read_input(file); },
        io::kPosteriorProbe);
}
