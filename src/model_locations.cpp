// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "model_locations.h"

#include "io/hdf5/hdf5_and_armadillo.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

/// The groups in one file `action` is to be run over.
///
/// Without --all-groups that is the one group the command line named, and the
/// file is not opened here at all -- a --group that names nothing is reported
/// by the action, where it always was. With it the file is opened to be asked
/// what models it holds.
bool groups_in_file(const std::filesystem::path &file, const CommandOptions &options,
                    std::vector<std::string> &groups)
{
	if (!options.all_groups)
	{
		groups.push_back(options.group);
		return true;
	}

	try
	{
		// A second open of a file the action opens again -- as
		// process_single_file_*() already opens it read-only for the algorithm
		// before the sampler reopens it read-write. Against a Gibbs run it does
		// not register.
		HighFive::File h5 = open_hdf5_file(file);
		groups = list_model_groups(h5, options.group);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error processing " << file.string() << ": " << e.what() << std::endl;
		return false;
	}

	return true;
}

/// `action` over every model in one file.
int run_over_file(const std::filesystem::path &file, const CommandOptions &options,
                  const ModelAction &action)
{
	std::vector<std::string> groups;

	if (!groups_in_file(file, options, groups))
	{
		return 1;
	}

	if (groups.empty())
	{
		// Having nothing to do is not failing -- the rule the README already
		// states for output that is already there, and for a directory with no
		// HDF5 files in it. Said out loud all the same, because a walk that
		// finds no models is usually a --group pointing at the wrong place.
		std::cerr << "Warning: no model under '"
		          << (options.group.empty() ? "/" : options.group) << "' in " << file.string()
		          << std::endl;
		return 0;
	}

	int failures = 0;
	for (const std::string &group : groups)
	{
		const ModelLocation location{file, group};
		std::cout << "Processing: " << location.describe() << std::endl;
		failures += action(location);
	}

	return failures;
}

} // namespace

int run_over_models(const CommandOptions &options, const ModelAction &action)
{
	const std::filesystem::path &path = options.path;

	// A path that is not there is a command line that cannot be acted on, like a
	// --group that cannot name a group: nothing started, so 2 rather than 1.
	if (!std::filesystem::exists(path))
	{
		std::cerr << "Error: Path does not exist: " << path << std::endl;
		return 2;
	}

	if (std::filesystem::is_directory(path))
	{
		int failures = 0;

		for (const auto &entry : std::filesystem::recursive_directory_iterator(path))
		{
			if (entry.is_regular_file() && is_hdf5_file(entry.path()))
			{
				failures += run_over_file(entry.path(), options, action);
			}
		}

		return failures == 0 ? 0 : 1;
	}

	if (std::filesystem::is_regular_file(path))
	{
		if (!is_hdf5_file(path))
		{
			std::cerr << "Error: Not an hdf5 file: " << path << std::endl;
			return 1;
		}

		return run_over_file(path, options, action) == 0 ? 0 : 1;
	}

	std::cerr << "Error: Path is neither a file nor a directory: " << path << std::endl;
	return 1;
}
