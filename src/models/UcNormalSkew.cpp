// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the UcNormalSkew sampler.
//
// The numerics live in src/core/models/uc_normal_skew.cpp and know nothing
// about files. What is left here is the part that is specific to driving the
// model from a file on disk: deciding whether the work has already been done,
// reading the input, and putting the results back.

#include "models/models.h"
#include "models/model_check.h"
#include "models/coefficients_stage.h"

#include "bayests/uc_normal_skew.h"
#include "io/hdf5/uc_normal_skew_io.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "reporters/console_reporter.h"

#include <iostream>
#include <stdexcept>

namespace
{
namespace io = bayests::hdf5_io::uc_normal_skew;
}

UcNormalSkew::UcNormalSkew()
{
}

UcNormalSkew::~UcNormalSkew()
{
}

void UcNormalSkew::draw_coefficients(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    // Skipped if already estimated; estimated again if a run was stopped while
    // writing -- see coefficients_needed().
    if (!coefficients_needed(file, "/posterior/u_omega_inv/coeffs"))
    {
        return;
    }

    const bayests::UcNormalSkewInput input = io::read_input(file);

    bayests::ConsoleReporter reporter;
    const bayests::UcNormalSkewDraws draws =
        bayests::UcNormalSkewSampler{}.draw_coefficients(input, reporter);

    bayests::hdf5_io::mark_coefficients_writing(file);
    io::write_coefficients(file, draws);
    bayests::hdf5_io::mark_coefficients_complete(file);
}

void UcNormalSkew::forecast(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    // Open the file and name the model inside it
    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    if (!dataset_has_data(file, "/posterior/u_omega_inv/coeffs"))
    {
        throw std::runtime_error("Posterior draws of u_omega_inv are missing.");
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

    // Stop if forecasts are already available in the object. There is no score
    // to add to them: the model's predictive density is not written, and
    // validate() refuses a file that carries /data/test/y.
    if (dataset_has_data(file, "/posterior/forecast/forecasts"))
    {
        return;
    }

    const bayests::UcNormalSkewInput input = io::read_input(file);
    const bayests::UcNormalSkewDraws draws = io::read_coefficients(file);

    bayests::NullReporter reporter;
    const bayests::ForecastDraws fcst = bayests::UcNormalSkewSampler{}.forecast(input, draws, reporter);
    bayests::hdf5_io::write_forecast(file, fcst);
}

void UcNormalSkew::log_likelihood(const ModelLocation &location_arg)
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

    if (!dataset_has_data(file, "/posterior/u_omega_inv/coeffs"))
    {
        throw std::runtime_error("Posterior draws of u_omega_inv are missing.");
    }

    const bayests::UcNormalSkewInput input = io::read_input(file);
    const bayests::UcNormalSkewDraws draws = io::read_coefficients(file);

    const arma::mat loglik = bayests::UcNormalSkewSampler{}.log_likelihood(input, draws);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}

ModelCheck UcNormalSkew::check(const ModelLocation &location_arg)
{
    return check_uc_model(location_arg, [](const ModelFile &file) { return io::read_input(file); },
                          "/posterior/u_omega_inv/coeffs");
}
