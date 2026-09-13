// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_MODEL_SEED_H
#define BAYESTS_MODEL_SEED_H

#include <cstdint>
#include <optional>

/// The stages of a run, in the order `bayests posterior` runs them.
enum class ModelStage
{
	coefficients,
	log_likelihood,
	forecast,
};

/// Seeds Armadillo's generator for one stage of a model whose file names a
/// /model/seed, and leaves it untouched for one that does not -- which is what
/// keeps every file written before the attribute existed drawing what it drew.
/// model_seed.cpp says why it is per stage and what each stage starts from.
void seed_model_stage(const std::optional<std::uint64_t> &seed, ModelStage stage);

#endif // BAYESTS_MODEL_SEED_H
