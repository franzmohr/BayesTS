// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VecNormalStochvol sampler.
//
// The numerics live in src/core/models/vec_normal_stochvol.cpp and know nothing
// about files. What is left here is the part that is specific to driving the
// model from a file on disk: deciding whether the work has already been done,
// reading the input, and putting the results back.

#include "models/models.h"

#include "bayests/vec_normal_stochvol.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/vec_normal_stochvol_io.h"
#include "reporters/console_reporter.h"

#include <iostream>
#include <stdexcept>

namespace
{
namespace io = bayests::hdf5_io::vec_normal_stochvol;
}

VecNormalStochvol::VecNormalStochvol()
{
}

VecNormalStochvol::~VecNormalStochvol()
{
}

void VecNormalStochvol::draw_coefficients(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    // Check if posterior data already exists
    if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        std::cout << "Posterior data already exists in file. Skipping simulation." << std::endl;
        return;
    }

    const bayests::VecNormalStochvolInput input = io::read_input(file);

    bayests::ConsoleReporter reporter;
    const bayests::VecNormalStochvolDraws draws =
        bayests::VecNormalStochvolSampler{}.draw_coefficients(input, reporter);

    io::write_coefficients(file, draws);
}

void VecNormalStochvol::forecast(const ModelLocation &location_arg)
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

    // No horizon, so no forecast was asked for: a skip, not a failure. A file
    // written with h = 0 carries no attribute at all, which is what every
    // -nofcst fixture looks like. Five of these front-ends used to print
    // "Error processing ..." here and the other thirteen returned in silence;
    // all eighteen are silent now, because none of them has failed.
    if (!attribute_exists(file, "/model", "h"))
    {
        return;
    }

    // Stop if forecasts are already available in the object
    if (dataset_has_data(file, "/posterior/forecast"))
    {
        return;
    }

    const bayests::VecNormalStochvolInput input = io::read_input(file);
    const bayests::VecNormalStochvolDraws draws = io::read_forecast_coefficients(file, input);

    bayests::NullReporter reporter;
    const bayests::ForecastDraws fcst =
        bayests::VecNormalStochvolSampler{}.forecast(input, draws, reporter);

    bayests::hdf5_io::write_forecast(file, fcst);
}

void VecNormalStochvol::log_likelihood(const ModelLocation &location_arg)
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

    const bayests::VecNormalStochvolInput input = io::read_input(file);
    const bayests::VecNormalStochvolDraws draws = io::read_coefficients(file);

    const arma::mat loglik =
        bayests::VecNormalStochvolSampler{}.log_likelihood(input, draws);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}
