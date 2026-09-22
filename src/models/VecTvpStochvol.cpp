// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VecTvpStochvol sampler.
//
// The numerics live in src/core/models/vec_tvp_stochvol.cpp and know nothing
// about files. What is left here is the part that is specific to driving the
// model from a file on disk: deciding whether the work has already been done,
// reading the input, and putting the results back.

#include "models/models.h"
#include "models/model_check.h"
#include "models/coefficients_stage.h"

#include "bayests/vec_tvp_stochvol.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/vec_tvp_stochvol_io.h"
#include "reporters/console_reporter.h"

#include <iostream>
#include <stdexcept>

namespace
{
namespace io = bayests::hdf5_io::vec_tvp_stochvol;
}

VecTvpStochvol::VecTvpStochvol()
{
}

VecTvpStochvol::~VecTvpStochvol()
{
}

void VecTvpStochvol::draw_coefficients(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    // Skipped if already estimated; estimated again if a run was stopped while
    // writing -- see coefficients_needed().
    if (!coefficients_needed(file, "/posterior/u_sigma_inv/coeffs"))
    {
        return;
    }

    const bayests::VecTvpStochvolInput input = io::read_input(file);

    bayests::ConsoleReporter reporter;
    const bayests::VecTvpStochvolDraws draws =
        bayests::VecTvpStochvolSampler{}.draw_coefficients(input, reporter);

    bayests::hdf5_io::mark_coefficients_writing(file);
    io::write_coefficients(file, draws);
    bayests::hdf5_io::mark_coefficients_complete(file);
}

void VecTvpStochvol::forecast(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        throw std::runtime_error("Posterior draws of u_sigma_inv are missing.");
    }

    // No horizon, so no forecast was asked for: a skip, not a failure. That is
    // an absent attribute -- every -nofcst fixture leaves it out -- and just as
    // much a written h <= 0, which is how R and Python callers spell "no
    // forecast". Five of these front-ends used to print "Error processing ..."
    // here and the other thirteen returned in silence; all of them are silent
    // now, because none of them has failed.
    if (bayests::hdf5_io::optional_attribute_int(file, "/model", "h", 0) <= 0)
    {
        return;
    }

    // Stop if forecasts are already available in the object -- unless the file
    // carries what the horizon realised and has not been scored against it.
    // The two members of the group are asked for separately so that adding
    // /data/test/y to a file that was already forecast is enough to score it,
    // rather than needing the paths thrown away first.
    const bool scorable = dataset_has_data(file, "/data/test/y");
    const bool scored = dataset_has_data(file, "/posterior/forecast/loglik");
    if (dataset_has_data(file, "/posterior/forecast/forecasts") && (!scorable || scored))
    {
        return;
    }

    const bayests::VecTvpStochvolInput input = io::read_input(file);
    bayests::hdf5_io::require_log_volatility_variances(file, input.spec);

    // The coefficients, the cointegration vectors and the precision all move
    // with time, so the forecast starts from the last in-sample period of
    // each.
    const bayests::VecTvpStochvolDraws draws = io::read_forecast_coefficients(file, input);

    bayests::NullReporter reporter;
    if (!dataset_has_data(file, "/posterior/forecast/forecasts"))
    {
        const bayests::ForecastDraws fcst =
            bayests::VecTvpStochvolSampler{}.forecast(input, draws, reporter);
        bayests::hdf5_io::write_forecast(file, fcst);
    }

    // Scored in the level parameterisation the forecast runs in, against the
    // realised levels /data/test/y holds, under the same walk.
    if (scorable && !scored)
    {
        const arma::mat score = bayests::VecTvpStochvolSampler{}.predictive_log_density(input, draws);
        bayests::hdf5_io::write_forecast_loglik(file, score);
    }
}

void VecTvpStochvol::log_likelihood(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    // Stop if the log likelihood is already available
    if (dataset_has_data(file, "/posterior/loglik"))
    {
        return;
    }

    if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        throw std::runtime_error("Posterior draws of u_sigma_inv are missing.");
    }

    const bayests::VecTvpStochvolInput input = io::read_input(file);
    const bayests::VecTvpStochvolDraws draws = io::read_loglik_coefficients(file, input);

    const arma::mat loglik =
        bayests::VecTvpStochvolSampler{}.log_likelihood(input, draws);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}

ModelCheck VecTvpStochvol::check(const ModelLocation &location_arg)
{
    return check_vec_model(location_arg, [](const ModelFile &file) { return io::read_input(file); });
}
