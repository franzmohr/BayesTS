// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr


#include "cli_options.h"
#include "model_locations.h"
#include "models/models.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include <iostream>
#include <string>

// Helper function to process a single model
static int process_single_file_evaluation(const ModelLocation &location, bool run_coefficients,
                                         bool run_forecasts, bool run_loglik)
{
	try
	{
		std::string model_type;

		{
			// Open HDF5 file (will be closed when scope ends)
			HighFive::File h5 = open_hdf5_file(location.file);

			// A --group that names nothing is reported here rather than as a
			// missing dataset further in.
			require_group(h5, location.group);

			// Get model type from the model's own /model group
			model_type = get_algorithm_type(ModelFile(h5, location.group));

			// File is automatically closed here when 'h5' goes out of scope
		}

		// Initialize model
		auto model = create_model(model_type);

		// Posterior draws
		if (run_coefficients)
		{
			model->draw_coefficients(location);
		}

		// Information criteria
		if (run_loglik)
		{
			model->log_likelihood(location);
		}

		// Forecasts
		if (run_forecasts)
		{
			model->forecast(location);
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error processing " << location.describe() << ": " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

int posterior(int argc, char *argv[])
{
	CommandOptions options;
	if (!parse_command_options(argc, argv, "posterior", true, options))
	{
		return 2;
	}

	return run_over_models(options, [&options](const ModelLocation &location) {
		return process_single_file_evaluation(location, options.run_coefficients,
		                                      options.run_forecasts, options.run_loglik);
	});
}
