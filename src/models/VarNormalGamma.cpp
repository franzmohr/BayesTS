// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VarNormalGamma sampler.
//
// The numerics live in src/core/models/var_normal_gamma.cpp and know nothing
// about files. What is left here is the part that is specific to driving the
// model from a file on disk: deciding whether the work has already been done,
// reading the input, and putting the results back.

#include "models/models.h"
#include "models/model_check.h"

#include "bayests/var_normal_gamma.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/var_normal_gamma_io.h"
#include "reporters/console_reporter.h"

#include <iostream>
#include <stdexcept>

namespace
{
namespace io = bayests::hdf5_io::var_normal_gamma;
}

VarNormalGamma::VarNormalGamma()
{
}

VarNormalGamma::~VarNormalGamma()
{
}

void VarNormalGamma::draw_coefficients(const ModelLocation &location_arg)
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

    const bayests::VarNormalGammaInput input = io::read_input(file);

    bayests::ConsoleReporter reporter;
    const bayests::VarNormalGammaDraws draws =
        bayests::VarNormalGammaSampler{}.draw_coefficients(input, reporter);

    io::write_coefficients(file, draws);
}

void VarNormalGamma::forecast(const ModelLocation &location_arg)
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
    // here and the other thirteen returned in silence; all eighteen are silent
    // now, because none of them has failed.
    if (bayests::hdf5_io::optional_attribute_int(file, "/model", "h", 0) <= 0)
    {
        return;
    }

    // Stop if forecasts are already available in the object
    if (dataset_has_data(file, "/posterior/forecast"))
    {
        return;
    }

    const bayests::VarNormalGammaInput input = io::read_input(file);
    const bayests::VarNormalGammaDraws draws = io::read_coefficients(file);

    bayests::NullReporter reporter;
    const bayests::ForecastDraws fcst =
        bayests::VarNormalGammaSampler{}.forecast(input, draws, reporter);

    bayests::hdf5_io::write_forecast(file, fcst);
}

void VarNormalGamma::log_likelihood(const ModelLocation &location_arg)
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

    const bayests::VarNormalGammaInput input = io::read_input(file);
    const bayests::VarNormalGammaDraws draws = io::read_coefficients(file);

    const arma::mat loglik =
        bayests::VarNormalGammaSampler{}.log_likelihood(input, draws);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}

ModelCheck VarNormalGamma::check(const ModelLocation &location_arg)
{
    return check_var_model(location_arg, [](const ModelFile &file) { return io::read_input(file); });
}
