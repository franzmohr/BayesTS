// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the DfmTvpGamma sampler.
//
// The numerics live in src/core/models/dfm_tvp_gamma.cpp and know nothing
// about files. What is left here is the part that is specific to driving the
// model from a file on disk: deciding whether the work has already been done,
// reading the input, and putting the results back.

#include "models/models.h"

#include "bayests/dfm_tvp_gamma.h"
#include "io/hdf5/dfm_tvp_gamma_io.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "reporters/console_reporter.h"

#include <iostream>

namespace
{
namespace io = bayests::hdf5_io::dfm_tvp_gamma;
}

DfmTvpGamma::DfmTvpGamma()
{
}

DfmTvpGamma::~DfmTvpGamma()
{
}

void DfmTvpGamma::draw_coefficients(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    try
    {
        // Check if posterior data already exists
        if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
        {
            std::cout << "Posterior data already exists in file. Skipping simulation." << std::endl;
            return;
        }

        const bayests::DfmTvpGammaInput input = io::read_input(file);

        bayests::ConsoleReporter reporter;
        const bayests::DfmTvpGammaDraws draws =
            bayests::DfmTvpGammaSampler{}.draw_coefficients(input, reporter);

        io::write_coefficients(file, draws);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void DfmTvpGamma::forecast(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    try
    {
        if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
        {
            std::cerr << "Error processing " << location.describe() << ": Posterior draws of u_sigma_inv are missing." << std::endl;
            return;
        }

        // Stop if h (the forecast horizon) does not exist
        if (!attribute_exists(file, "/model", "h"))
        {
            return;
        }

        // Stop if forecasts are already available in the object
        if (dataset_has_data(file, "/posterior/forecast"))
        {
            return;
        }

        const bayests::DfmTvpGammaInput input = io::read_input(file);

        // The loadings and the transition move with time, so the forecast
        // starts from the last in-sample period of each.
        const bayests::DfmTvpGammaDraws draws = io::read_forecast_coefficients(file, input);

        bayests::NullReporter reporter;
        const bayests::ForecastDraws fcst =
            bayests::DfmTvpGammaSampler{}.forecast(input, draws, reporter);

        bayests::hdf5_io::write_forecast(file, fcst);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void DfmTvpGamma::log_likelihood(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    try
    {
        // Stop if the log likelihood is already available
        if (dataset_has_data(file, "/posterior/loglik"))
        {
            return;
        }

        if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
        {
            std::cerr << "Error processing " << location.describe() << ": Posterior draws of u_sigma_inv are missing." << std::endl;
            return;
        }

        const bayests::DfmTvpGammaInput input = io::read_input(file);

        // The whole loading path: every period is scored under its own Lambda_t.
        const bayests::DfmTvpGammaDraws draws = io::read_loglik_coefficients(file);

        const arma::mat loglik = bayests::DfmTvpGammaSampler{}.log_likelihood(input, draws);

        bayests::hdf5_io::write_log_likelihood(file, loglik);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}
