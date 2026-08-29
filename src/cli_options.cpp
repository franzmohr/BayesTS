// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "cli_options.h"

#include "io/hdf5/hdf5_and_armadillo.h"

#include <iostream>
#include <stdexcept>

namespace
{

void print_usage(const std::string &command, bool accept_step_flags)
{
	std::cerr << "Usage: bayests " << command << " <file.h5 | directory> [--group <path>]";
	if (accept_step_flags)
	{
		std::cerr << " [--no-coefficients] [--no-forecasts] [--no-loglik]";
	}
	std::cerr << "\n";
	std::cerr << "  --group <path>  the group each model's tree hangs under, e.g. /models/3.\n"
	             "                  Defaults to the root of the file.\n";
}

} // namespace

bool parse_command_options(int argc, char *argv[], const std::string &command,
                           bool accept_step_flags, CommandOptions &options)
{
	// main() already rejects a call without a path, but each subcommand is
	// reachable on its own and must not index argv past the end.
	if (argc < 3)
	{
		print_usage(command, accept_step_flags);
		return false;
	}

	options.path = argv[2];

	for (int i = 3; i < argc; ++i)
	{
		const std::string arg = argv[i];

		// Both spellings, because a caller writing --group=/models/3 into a
		// script has no reason to expect the other one to be the only one that
		// works.
		std::string group;
		bool have_group = false;

		if (arg == "--group")
		{
			if (i + 1 >= argc)
			{
				std::cerr << "Error: --group needs the path of a group, e.g. --group /models/3\n";
				return false;
			}
			group = argv[++i];
			have_group = true;
		}
		else if (arg.rfind("--group=", 0) == 0)
		{
			group = arg.substr(std::string("--group=").size());
			have_group = true;
		}

		if (have_group)
		{
			// Normalized here rather than where the file is opened, so that a
			// group that cannot be one is a command line that was refused before
			// anything was read -- and, on a directory walk, before the first of
			// several hundred files was opened.
			try
			{
				options.group = normalize_hdf5_group(group);
			}
			catch (const std::exception &e)
			{
				std::cerr << "Error: " << e.what() << "\n";
				return false;
			}
			continue;
		}

		if (accept_step_flags && arg == "--no-coefficients")
		{
			options.run_coefficients = false;
		}
		else if (accept_step_flags && arg == "--no-forecasts")
		{
			options.run_forecasts = false;
		}
		else if (accept_step_flags && arg == "--no-loglik")
		{
			options.run_loglik = false;
		}
		else
		{
			std::cerr << "Warning: Unknown flag '" << arg << "' ignored" << std::endl;
		}
	}

	return true;
}
