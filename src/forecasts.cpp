// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr


#include "cli_options.h"
#include "model_locations.h"
#include "model_seed.h"
#include "models/models.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/model_io_common.h"
#include <cstdint>
#include <iostream>
#include <optional>

// Helper function to process a single model
static int process_single_file_forecasts(const ModelLocation &location)
{
	try
	{
		std::string model_type;
		std::optional<std::uint64_t> seed;

		{
			// Open HDF5 file (will be closed when scope ends)
			HighFive::File h5 = open_hdf5_file(location.file);

			// A --group that names nothing is reported here rather than as a
			// missing dataset further in.
			require_group(h5, location.group);

			// Get model type from the model's own /model group
			model_type = get_algorithm_type(ModelFile(h5, location.group));
			seed = bayests::hdf5_io::read_model_seed(ModelFile(h5, location.group));

			// File is automatically closed here when 'h5' goes out of scope
		}

		// Initialize model
		auto model = create_model(model_type);

		// Perform the forecast
		seed_model_stage(seed, ModelStage::forecast);
		model->forecast(location);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error processing " << location.describe() << ": " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

int forecasts(int argc, char *argv[])
{
	CommandOptions options;
	if (!parse_command_options(argc, argv, "forecasts", false, options))
	{
		return 2;
	}

	return run_over_models(options, process_single_file_forecasts);
}
