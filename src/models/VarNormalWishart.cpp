// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VarNormalWishart sampler.
//
// The numerics live in src/core/models/var_normal_wishart.cpp and know nothing
// about files. What is left here is the part that is specific to driving the
// model from a file on disk: deciding whether the work has already been done,
// reading the input, and putting the results back.

#include "models/models.h"

#include "bayests/var_normal_wishart.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/var_normal_wishart_io.h"
#include "reporters/console_reporter.h"

#include <iostream>

namespace
{
namespace io = bayests::hdf5_io::var_normal_wishart;
}

VarNormalWishart::VarNormalWishart()
{
}

VarNormalWishart::~VarNormalWishart()
{
}

void VarNormalWishart::draw_coefficients(const std::filesystem::path &filepath_arg)
{
    // Store filepath for later use
    this->filepath = filepath_arg;

    // Open file
    HighFive::File file = open_hdf5_file_readwrite(filepath);

    try
    {
        // Check if posterior data already exists
        if (dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
        {
            std::cout << "Posterior data already exists in file. Skipping simulation." << std::endl;
            return;
        }

        const bayests::VarNormalWishartInput input = io::read_input(file);

        bayests::ConsoleReporter reporter;
        const bayests::VarNormalWishartDraws draws =
            bayests::VarNormalWishartSampler{}.draw_coefficients(input, reporter);

        io::write_coefficients(file, draws);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void VarNormalWishart::forecast(const std::filesystem::path &filepath_arg)
{
    // Store filepath for later use
    this->filepath = filepath_arg;

    // Open file
    HighFive::File file = open_hdf5_file_readwrite(filepath);

    try
    {
        if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
        {
            std::cerr << "Error processing " << filepath << ": Posterior draws of u_sigma_inv are missing." << std::endl;
            return;
        }

        // Stop if h (the forecast horizon) does not exist
        if (!attribute_exists(file, "/model", "h"))
        {
            return;
        }

        // Stop if forecasts are already available in the obeject
        if (dataset_has_data(file, "/posterior/forecast"))
        {
            return;
        }

        const bayests::VarNormalWishartInput input = io::read_input(file);
        const bayests::VarNormalWishartDraws draws = io::read_coefficients(file);

        bayests::NullReporter reporter;
        const bayests::ForecastDraws fcst =
            bayests::VarNormalWishartSampler{}.forecast(input, draws, reporter);

        bayests::hdf5_io::write_forecast(file, fcst);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void VarNormalWishart::log_likelihood(const std::filesystem::path &filepath_arg)
{
    // Store filepath for later use
    this->filepath = filepath_arg;

    // Open file
    HighFive::File file = open_hdf5_file_readwrite(filepath);

    try
    {
        // Stop of log likelihood is already available
        if (dataset_has_data(file, "/posterior/loglik"))
        {
            return;
        }

        if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
        {
            std::cerr << "Error processing " << filepath << ": Posterior draws of u_sigma_inv are missing." << std::endl;
            return;
        }

        const bayests::VarNormalWishartInput input = io::read_input(file);
        const bayests::VarNormalWishartDraws draws = io::read_coefficients(file);

        const arma::mat loglik =
            bayests::VarNormalWishartSampler{}.log_likelihood(input, draws);

        bayests::hdf5_io::write_log_likelihood(file, loglik);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}
