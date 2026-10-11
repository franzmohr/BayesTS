// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// HDF5 front-end for the VarNormalAld sampler.
//
// The numerics live in src/core/models/var_normal_ald.cpp and know nothing about
// files. What is left here is the part that is specific to driving the model
// from a file on disk: deciding whether the work has already been done, reading
// the input, and putting the results back.

#include "models/models.h"
#include "models/model_check.h"
#include "models/coefficients_stage.h"

#include "bayests/var_normal_ald.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/var_normal_ald_io.h"

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

    // Skipped if already estimated; estimated again if a run was stopped while
    // writing -- see coefficients_needed(). Probed at the scale, which a single
    // quantile and a grid of them both write; a grid keeps no precision path.
    if (!coefficients_needed(file, "/posterior/u_scale/coeffs"))
    {
        return;
    }

    const bayests::VarNormalAldInput input = io::read_input(file);

    bayests::Reporter &reporter = stage_reporter();
    const bayests::VarNormalAldDraws draws =
        bayests::VarNormalAldSampler{}.draw_coefficients(input, reporter);

    bayests::hdf5_io::mark_coefficients_writing(file);
    io::write_coefficients(file, draws);
    bayests::hdf5_io::mark_coefficients_complete(file);
}

void VarNormalAld::forecast(const ModelLocation &location_arg)
{
    // Store the location for later use
    this->location = location_arg;

    HighFive::File h5 = open_hdf5_file_readwrite(location.file);
    const ModelFile file(h5, location.group);

    // A single quantile has no forecast: validate() refuses a horizon for it, so
    // a valid file of one asks for none, exactly as a file written with h = 0
    // does for every other model, and `bayests forecasts` over a directory of
    // them does nothing rather than failing. A grid of quantiles describes the
    // whole distribution and does forecast, by simulation.
    if (bayests::hdf5_io::optional_attribute_int(file, "/model", "h", 0) <= 0)
    {
        return;
    }
    if (dataset_has_data(file, "/posterior/forecast/forecasts"))
    {
        return;
    }
    if (!dataset_has_data(file, "/posterior/u_scale/coeffs"))
    {
        throw std::runtime_error("Posterior draws of u_scale are missing.");
    }

    const bayests::VarNormalAldInput input = io::read_input(file);
    const bayests::VarNormalAldDraws draws = io::read_coefficients(file);
    bayests::Reporter &reporter = quiet_reporter();
    const bayests::ForecastDraws fcst =
        bayests::VarNormalAldSampler{}.forecast(input, draws, reporter);
    bayests::hdf5_io::write_forecast(file, fcst);
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

    if (!dataset_has_data(file, "/posterior/u_scale/coeffs"))
    {
        throw std::runtime_error("Posterior draws of u_scale are missing.");
    }

    const bayests::VarNormalAldInput input = io::read_input(file);
    const bayests::VarNormalAldDraws draws = io::read_coefficients(file);

    const arma::mat loglik = bayests::VarNormalAldSampler{}.log_likelihood(input, draws);

    bayests::hdf5_io::write_log_likelihood(file, loglik);
}

ModelCheck VarNormalAld::check(const ModelLocation &location_arg)
{
    return check_quantile_model(
        location_arg, [](const ModelFile &file) { return io::read_input(file); },
        "/posterior/u_scale/coeffs");
}
