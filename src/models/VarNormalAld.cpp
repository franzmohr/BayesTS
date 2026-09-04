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

    try
    {
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
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void VarNormalAld::forecast(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // A quantile regression model has no forecast, and says so rather than
    // leaving the subcommand to do nothing quietly. The file cannot carry a
    // horizon either -- validate() refuses one -- so a run that reaches here has
    // asked for something the model cannot give, and the reason belongs on the
    // console rather than in a comment.
    (void)location_arg;

    std::cerr << "Error processing " << location.describe()
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

    try
    {
        // Stop if log likelihood is already available
        if (dataset_has_data(file, "/posterior/loglik"))
        {
            return;
        }

        if (!dataset_has_data(file, "/posterior/u_sigma_inv/coeffs"))
        {
            std::cerr << "Error processing " << location.describe() << ": Posterior draws of u_sigma_inv are missing." << std::endl;
            return;
        }

        const bayests::VarNormalAldInput input = io::read_input(file);
        const bayests::VarNormalAldDraws draws = io::read_coefficients(file);

        const arma::mat loglik = bayests::VarNormalAldSampler{}.log_likelihood(input, draws);

        bayests::hdf5_io::write_log_likelihood(file, loglik);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}
