// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_MODEL_LOCATIONS_H
#define BAYESTS_MODEL_LOCATIONS_H

#include "cli_options.h"
#include "models/models.h"

#include <functional>

/// What a subcommand does with one model: 0, or 1 having already reported on
/// stderr why it could not. The shape each subcommand's own
/// process_single_file_*() already has, so they pass those through unchanged.
using ModelAction = std::function<int(const ModelLocation &)>;

/// Runs `action` over every model the command line names, and returns 0, or 1
/// if any of them failed, or 2 if the path does not exist -- a command line
/// that cannot be acted on, which never started a walk.
///
/// Two walks, nested. The outer one is over files: one file, or every HDF5 file
/// below a directory, recursively. The inner one is over the models in each
/// file: the single group options.group, or -- under --all-groups -- every
/// model below it.
///
/// A model that fails is reported and the walk carries on to the next, because
/// a caller pointing this at a directory of models cannot see stderr per model
/// and one bad file must not strand the rest. The exit status is what tells it
/// something failed, which is why the failures are counted rather than
/// returned early.
///
/// This exists once rather than four times: the path validation and the walk
/// used to be duplicated verbatim in posterior.cpp, coefficients.cpp,
/// forecasts.cpp and loglik.cpp, and --all-groups would have been a fifth and
/// sixth copy of the same loop.
int run_over_models(const CommandOptions &options, const ModelAction &action);

#endif // BAYESTS_MODEL_LOCATIONS_H
