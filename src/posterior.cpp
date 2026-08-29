// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr


#include "cli_options.h"
#include "models/models.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include <iostream>
#include <filesystem>
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

	const std::filesystem::path filepath = options.path;

	// Check if path exists
	if (!std::filesystem::exists(filepath))
	{
		std::cerr << "Error: Path does not exist: " << filepath << std::endl;
		return 1;
	}

	// Check if path is a directory
	if (std::filesystem::is_directory(filepath))
	{

		// A file that fails is reported and the walk continues, but the exit
		// status has to say that something failed: a caller looping over model
		// directories cannot see stderr per file.
		int failures = 0;

		// Loop over all files in the directory and subdirectories recursively
		for (const auto& entry : std::filesystem::recursive_directory_iterator(filepath))
		{
			if (entry.is_regular_file() && is_hdf5_file(entry.path()))
			{
				const ModelLocation location{entry.path(), options.group};
				std::cout << "Processing: " << location.describe() << std::endl;
				failures += process_single_file_evaluation(location, options.run_coefficients,
				                                          options.run_forecasts,
				                                          options.run_loglik);
			}
		}
		return failures == 0 ? 0 : 1;
	}
	else if (std::filesystem::is_regular_file(filepath))
	{
		// Process single file, but only if it is an hdf5 file
		if (!is_hdf5_file(filepath))
		{
			std::cerr << "Error: Not an hdf5 file: " << filepath << std::endl;
			return 1;
		}
		return process_single_file_evaluation(ModelLocation{filepath, options.group},
		                                     options.run_coefficients, options.run_forecasts,
		                                     options.run_loglik);
	}
	else
	{
		std::cerr << "Error: Path is neither a file nor a directory: " << filepath << std::endl;
		return 1;
	}
}
