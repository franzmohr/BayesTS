// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "model_locations.h"

#include "io/hdf5/hdf5_and_armadillo.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>
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

void report_unreadable(const std::filesystem::path &path, const std::error_code &error)
{
	std::cerr << "Error processing " << path.string() << ": cannot read it: " << error.message()
	          << std::endl;
}

/// Whether a status query found nothing there, however the standard library
/// says so. libstdc++ on Windows reports a missing path as `not_found` and sets
/// ENOENT beside it, where other platforms clear the error; either is the same
/// answer, and neither is a failure to read something that exists.
bool nothing_there(const std::filesystem::file_status &status, const std::error_code &error)
{
	return status.type() == std::filesystem::file_type::not_found ||
	       error == std::errc::no_such_file_or_directory;
}

void warn_dangling(const std::filesystem::path &path)
{
	std::cerr << "Warning: skipping " << path.string()
	          << ": it is a link whose target does not exist" << std::endl;
}

void warn_cycle(const std::filesystem::path &path, const std::filesystem::path &ancestor)
{
	std::cerr << "Warning: skipping " << path.string() << ": it leads back to "
	          << ancestor.string() << ", which this walk is already inside" << std::endl;
}

/// True if `a` and `b` are the same file or directory, however each was reached.
/// An error -- a link whose target is gone, say -- reads as "not the same", so
/// the caller goes on to find out what is wrong with the path itself.
bool same_entry(const std::filesystem::path &a, const std::filesystem::path &b)
{
	std::error_code error;
	return std::filesystem::equivalent(a, b, error) && !error;
}

/// Drops every path in `files` that is the same file as one before it, reached
/// by another route -- a junction into a different branch of the walk. Kept is
/// the first in sorted order, so which one survives is the same on every run.
///
/// Compared only within one file name: a second route to a file cannot change
/// its name, only the folders above it, so this finds every duplicate without
/// asking the file system about every pair.
void unique_files(std::vector<std::filesystem::path> &files)
{
	std::vector<std::filesystem::path> kept;
	kept.reserve(files.size());
	for (const std::filesystem::path &file : files)
	{
		const bool seen = std::any_of(kept.begin(), kept.end(),
		                               [&](const std::filesystem::path &other) {
			                               return other.filename() == file.filename() &&
			                                      same_entry(other, file);
		                               });
		if (!seen)
		{
			kept.push_back(file);
		}
	}
	files = std::move(kept);
}

/// Every HDF5 file below `root`, sorted, with the number of entries that could
/// not be read.
///
/// Walked by hand, with error codes, rather than with the recursive directory
/// iterator this replaces. That iterator throws on the first entry it cannot
/// read -- a subdirectory without permission, a junction or link whose target is
/// gone -- and nothing caught it: the program ended in std::terminate, with
/// neither a line naming the entry nor either of the exit codes a script
/// branches on, and every file after it went unprocessed.
///
/// An entry that cannot be read is reported and counted as a failure, since it
/// may have held models the caller meant to run, and the walk carries on past
/// it. A link whose target does not exist is skipped with a warning instead:
/// there is nothing behind it to have been missed. How such a link shows up
/// depends on the platform. Where the standard library can see links it reports
/// the target as not found. libstdc++ on Windows cannot see them: a junction to a
/// directory that is gone reports as a directory, and only opening it says
/// ENOENT, which is therefore read the same way below the root. Links to
/// directories that the library does recognise are not followed, as the
/// iterator did not follow them.
///
/// Which leaves Windows following junctions, since the library sees each as a
/// plain directory. That is kept -- a junction is how a folder of models gets
/// pulled in from elsewhere -- but a junction to a directory the walk is already
/// inside used to be followed round and round until the path grew past what
/// Windows would open, running every model beneath it once per lap. So each
/// directory is compared with its own ancestors, by file identity rather than
/// by name, and one that leads back to an ancestor is skipped with a warning.
/// Ancestors only: a cycle has to close on one, and checking the chain costs
/// one comparison per level rather than one per directory walked. A junction
/// into a different branch is no cycle, but it reaches the same files twice,
/// so those are dropped at the end instead: see unique_files().
///
/// Sorted, as list_model_groups() sorts the groups of one file, so the order the
/// models run and fail in is the same on every platform.
int collect_hdf5_files(const std::filesystem::path &root, std::vector<std::filesystem::path> &files)
{
	int failures = 0;

	// Every directory opened, with the index of the one it was found in, so that
	// a directory's ancestors are a walk up `parent` -- the chain a cycle closes on.
	struct Walked
	{
		std::filesystem::path path;
		std::size_t parent;
	};
	constexpr std::size_t no_parent = static_cast<std::size_t>(-1);
	std::vector<Walked> walked;

	struct Pending
	{
		std::filesystem::path path;
		std::size_t parent;
	};
	std::vector<Pending> pending{{root, no_parent}};

	while (!pending.empty())
	{
		const Pending next = pending.back();
		pending.pop_back();
		const std::filesystem::path &directory = next.path;

		bool cycle = false;
		for (std::size_t up = next.parent; up != no_parent; up = walked[up].parent)
		{
			if (same_entry(directory, walked[up].path))
			{
				warn_cycle(directory, walked[up].path);
				cycle = true;
				break;
			}
		}
		if (cycle)
		{
			continue;
		}

		std::error_code error;
		std::filesystem::directory_iterator it(directory, error);
		if (error)
		{
			if (directory != root && error == std::errc::no_such_file_or_directory)
			{
				warn_dangling(directory);
			}
			else
			{
				report_unreadable(directory, error);
				++failures;
			}
			continue;
		}

		const std::size_t here = walked.size();
		walked.push_back({directory, next.parent});

		for (const std::filesystem::directory_iterator end; it != end; it.increment(error))
		{
			const std::filesystem::directory_entry &entry = *it;

			std::error_code link_error;
			const bool is_link = std::filesystem::is_symlink(entry.symlink_status(link_error));

			std::error_code status_error;
			const std::filesystem::file_status status = entry.status(status_error);

			if (nothing_there(status, status_error))
			{
				warn_dangling(entry.path());
				continue;
			}
			if (status_error)
			{
				report_unreadable(entry.path(), status_error);
				++failures;
				continue;
			}

			if (std::filesystem::is_directory(status))
			{
				if (!is_link)
				{
					pending.push_back({entry.path(), here});
				}
			}
			else if (std::filesystem::is_regular_file(status) && is_hdf5_file(entry.path()))
			{
				files.push_back(entry.path());
			}
		}

		if (error)
		{
			report_unreadable(directory, error);
			++failures;
		}
	}

	std::sort(files.begin(), files.end());
	unique_files(files);
	return failures;
}

} // namespace

int run_over_models(const CommandOptions &options, const ModelAction &action)
{
	const std::filesystem::path &path = options.path;

	// Error codes throughout, for the reason collect_hdf5_files() gives: a
	// filesystem call that throws here ends the program without an exit code.
	std::error_code error;
	const std::filesystem::file_status status = std::filesystem::status(path, error);

	// A path that is not there, or cannot be asked about, is a command line that
	// cannot be acted on, like a --group that cannot name a group: nothing
	// started, so 2 rather than 1.
	if (nothing_there(status, error))
	{
		std::cerr << "Error: Path does not exist: " << path << std::endl;
		return 2;
	}
	if (error)
	{
		std::cerr << "Error: cannot read the path " << path << ": " << error.message() << std::endl;
		return 2;
	}

	if (std::filesystem::is_directory(status))
	{
		std::vector<std::filesystem::path> files;
		int failures = collect_hdf5_files(path, files);

		for (const std::filesystem::path &file : files)
		{
			failures += run_over_file(file, options, action);
		}

		return failures == 0 ? 0 : 1;
	}

	if (std::filesystem::is_regular_file(status))
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
