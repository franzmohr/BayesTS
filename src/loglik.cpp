// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr


#include "models/models.h"
#include "io/hdf5/hdf5_and_armadillo.h"
#include <iostream>
#include <filesystem>
#include <string>

// Helper function to process a single file
static int process_single_file_loglik(const std::filesystem::path& filepath) {
	try
	{
		std::string model_type;

		{
			// Open HDF5 file (will be closed when scope ends)
			HighFive::File file = open_hdf5_file(filepath);

			// Get model type from file
			model_type = get_algorithm_type(file);

			// File is automatically closed here when 'file' goes out of scope
		}

		// Initialize model
		auto model = create_model(model_type);

		model->log_likelihood(filepath);
		
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error processing " << filepath << ": " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

int loglik(int argc, char *argv[])
{
	// main() already rejects a call without a path, but each subcommand is
	// reachable on its own and must not index argv past the end.
	if (argc < 3)
	{
		std::cerr << "Usage: bayests loglik <file.h5 | directory>" << std::endl;
		return 2;
	}


	// Get filepath from command line argument
	std::filesystem::path filepath = argv[2];

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
				std::cout << "Processing: " << entry.path() << std::endl;
				failures += process_single_file_loglik(entry.path());
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
		return process_single_file_loglik(filepath);
	}
	else
	{
		std::cerr << "Error: Path is neither a file nor a directory: " << filepath << std::endl;
		return 1;
	}
}