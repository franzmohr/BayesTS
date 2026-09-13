// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// Seeding the random number generator from /model/seed.
//
// The generator belongs to whoever hosts the samplers: under RcppArmadillo it is
// R's, seeded by set.seed(), and nothing under src/core/ touches it. On the
// command line that host is this program, so a file that names a seed is seeded
// here, and the seed is read by the I/O layer like every other attribute.
//
// Once per stage rather than once per model, so that each stage's draws are a
// function of the file alone. `bayests posterior` and `coefficients`, `loglik`
// and `forecasts` run one after the other give the same numbers; a forecast
// deleted and re-run gives the one it replaced; and a model in a walk over a
// directory or --all-groups does not start from whatever the model before it
// left in the generator.
//
// The chain starts from the seed itself -- what arma_rng::set_seed(seed) just
// before the sampler would give, so a seed can be carried over from a host that
// seeds that way. The log likelihood and the forecast start from streams derived
// from it, so the forecast's shocks are not the numbers the chain began with.

#include "model_seed.h"

#include "bayests/arma.h"

namespace
{

/// splitmix64's finaliser: a bijection on 64 bits that sends neighbouring inputs
/// far apart, so the streams derived from seeds s and s + 1 are unrelated to each
/// other and to either seed.
std::uint64_t mix(std::uint64_t x)
{
	x += 0x9E3779B97F4A7C15ull;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
	return x ^ (x >> 31);
}

std::uint64_t stage_seed(std::uint64_t seed, ModelStage stage)
{
	if (stage == ModelStage::coefficients)
	{
		return seed;
	}
	return mix(mix(seed) ^ static_cast<std::uint64_t>(stage));
}

} // namespace

void seed_model_stage(const std::optional<std::uint64_t> &seed, ModelStage stage)
{
	if (!seed)
	{
		return;
	}

	// seed_type is 64 bits wide wherever Armadillo uses the C++11 generator,
	// which is every build this project supports.
	arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(stage_seed(*seed, stage)));
}
