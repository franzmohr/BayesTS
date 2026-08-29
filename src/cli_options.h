// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_CLI_OPTIONS_H
#define BAYESTS_CLI_OPTIONS_H

#include <filesystem>
#include <string>

/// What a subcommand reads off the command line.
struct CommandOptions
{
    /// The file or directory to work on.
    std::filesystem::path path;

    /// The group each model's tree hangs under inside its file, normalized, ""
    /// for the root of the file. One group for the whole invocation: a directory
    /// walk looks for the same group in every file it visits, which is what a
    /// caller with a directory of files written the same way wants.
    std::string group;

    // posterior's three steps. The subcommands that run a single step take no
    // flags for these and leave them alone.
    bool run_coefficients = true;
    bool run_forecasts = true;
    bool run_loglik = true;
};

/// Parses `<path> [flags...]`, starting at argv[2] -- argv[1] is the command,
/// which main() has already dispatched on.
///
/// Returns false for a command line that cannot be acted on: no path, a
/// --group with no value after it, or a group that cannot name an HDF5 group.
/// The reason is on stderr by then, and the caller returns 2 -- the exit code
/// the CI smoke tests read as "the binary loaded and reached main()". An
/// unrecognised flag is warned about and ignored rather than refused, which is
/// what posterior did before this was shared.
///
/// `accept_step_flags` admits posterior's --no-coefficients, --no-forecasts and
/// --no-loglik. The other three subcommands run one step each, so for them
/// those names are unrecognised flags.
bool parse_command_options(int argc, char *argv[], const std::string &command,
                           bool accept_step_flags, CommandOptions &options);

#endif // BAYESTS_CLI_OPTIONS_H
