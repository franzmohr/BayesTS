// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VarNormalAld sampler.
//
// The numerics live in src/core/models/var_normal_ald.cpp and know nothing about
// files. What is left here is the part that is specific to driving the model
// from a file on disk: deciding whether the work has already been done, reading
// the input, and putting the results back.

#include "models/models.h"

#include "bayests/var_normal_ald.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/var_normal_ald_io.h"
#include "reporters/console_reporter.h"

#include <iostream>
#include <stdexcept>

namespace
{
namespace io = bayests::hdf5_io::var_normal_ald;
}

VarNormalAld::VarNormalAld()
{
}

VarNormalAld::~VarNormalAld()
{
}

void VarNormalAld::draw_coefficients(const ModelLocation &location_arg)
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

    const bayests::VarNormalAldInput input = io::read_input(file);

    bayests::ConsoleReporter reporter;
    const bayests::VarNormalAldDraws draws =
        bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter);

    io::write_coefficients(file, draws);
}

void VarNormalAld::forecast(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // A quantile regression model has no forecast, and says so rather than
    // leaving the subcommand to do nothing quietly.
    //
    // Not an error, and it does not throw: validate() refuses a non-zero horizon,
    // so a valid file here asks for nothing, exactly as a file written with h = 0
    // does for every other model. `bayests forecasts` over a directory should not
    // fail because some of the models in it are quantile models. What would be an
    // error is a horizon reaching a sampler, and VarNormalAldSampler::forecast()
    // throws on that.
    std::cerr << location.describe()
              << ": a quantile regression model does not forecast. The h step ahead quantile is "
                 "not the quantile of the iterated one step ahead quantiles, so there is no path "
                 "to simulate that could be read as one."
              << std::endl;
}

void VarNormalAld::log_likelihood(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    // Stop if log likelihood is already available
    if (dataset_has_data(file, "/posterior/loglik"))
    {
        return;
    }

    if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
    {
        throw std::runtime_error("Posterior draws of u_sigma_inv are missing.");
    }

    const bayests::VarNormalAldInput input = io::read_input(file);
    const bayests::VarNormalAldDraws draws = io::read_coefficients(file);

    const arma::mat loglik = bayests::VarNormalAldSampler{}.log_likelihood(input, draws);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}
