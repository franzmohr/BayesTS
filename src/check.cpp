// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// `bayests check`: whether a run would accept a model file, and how it read it.
//
// A run that exits 0 proves little about a file written by hand. Most fields are
// read through a default, so a misspelled attribute or the wrong error spelling
// is not an error but a different model. This reads each model the way
// `coefficients` would -- the same reader and the same validate(), see
// models/model_check.h -- and stops before the sampler, so it draws nothing and
// writes nothing. What it prints is how the file was understood, to compare
// against what was meant, plus a warning for each thing in the file the model
// never looked at, which is where a silently different model usually shows.
//
// Exit codes as for the other subcommands: 0 when every model would be accepted,
// warnings or not; 1 when any would be refused, with the reason on stderr; 2 for
// a command line that cannot be acted on.

#include "cli_options.h"
#include "model_locations.h"
#include "models/models.h"
#include "io/hdf5/hdf5_and_armadillo.h"

#include <iostream>
#include <memory>
#include <string>

namespace
{

bool starts_with(const std::string &text, const std::string &prefix)
{
	return text.rfind(prefix, 0) == 0;
}

bool contains(const std::string &text, const std::string &part)
{
	return text.find(part) != std::string::npos;
}

bool ends_with(const std::string &text, const std::string &suffix)
{
	return text.size() >= suffix.size() &&
	       text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// The error spelling that switches this model's covariance block on, or "" for
/// a model without one. Read off the name for the message alone: which spelling
/// actually counts is decided by the reader, and spec.covar is what it decided.
std::string covariance_spelling(const std::string &algorithm)
{
	if (starts_with(algorithm, "Var") || starts_with(algorithm, "Vec"))
	{
		if (ends_with(algorithm, "Gamma"))
		{
			return "gamma+covar";
		}
		if (ends_with(algorithm, "Stochvol"))
		{
			return "sv+covar";
		}
	}
	return "";
}

void print_report(const std::string &algorithm, const ModelCheck &check)
{
	const bayests::VarSpec &spec = check.spec;
	const bool time_varying = contains(algorithm, "Tvp");
	const bool favar = starts_with(algorithm, "Favar");
	const bool factor_model = favar || starts_with(algorithm, "Dfm");
	const bool vec = starts_with(algorithm, "Vec");
	const std::string spelling = covariance_spelling(algorithm);

	const auto per_period = [&](int count) {
		std::string text = std::to_string(count);
		if (time_varying)
		{
			text += " per period, a path over " + std::to_string(check.periods) + " periods";
		}
		return text;
	};

	std::cout << "  accepted: " << algorithm << " reads this file and would run it\n";

	if (factor_model)
	{
		std::cout << "  dimensions: k = " << spec.k << " observed series, n_factors = "
		          << spec.n_factors;
		if (favar)
		{
			std::cout << ", n_obs_factors = " << spec.n_obs_factors;
		}
		std::cout << ", p = " << spec.p << ", tt = " << check.periods << "\n";
		std::cout << "  free loadings: "
		          << per_period(favar ? spec.n_favar_lambda() : spec.n_lambda()) << "\n";
		std::cout << "  transition coefficients: "
		          << per_period(favar ? spec.n_favar_a() : spec.n_factor_a()) << "\n";
	}
	else if (vec)
	{
		std::cout << "  dimensions: k = " << spec.k << ", p = " << spec.p << " (level order), m = "
		          << spec.m << ", s = " << spec.s << ", n = " << spec.n << ", tt = "
		          << check.periods << "\n";
		std::cout << "  cointegration: rank = " << spec.rank << ", k_beta = " << spec.k_beta
		          << ", n_restricted = " << spec.n_restricted << ", " << spec.n_beta()
		          << " elements of beta\n";
		std::cout << "  coefficients: " << per_period(spec.nparams_per_period_vec()) << " ("
		          << spec.n_alpha() << " loadings, " << spec.n_x_vec()
		          << " further regressors per equation";
		if (spec.structural)
		{
			std::cout << ", " << spec.n_structural() << " contemporaneous";
		}
		std::cout << ")\n";
	}
	else
	{
		std::cout << "  dimensions: k = " << spec.k << ", p = " << spec.p << ", m = " << spec.m
		          << ", s = " << spec.s << ", n = " << spec.n << ", tt = " << check.periods << "\n";
		std::cout << "  coefficients: " << per_period(spec.nparams_per_period()) << " ("
		          << spec.n_x() << " regressors per equation";
		if (spec.structural)
		{
			std::cout << ", " << spec.n_structural() << " contemporaneous";
		}
		std::cout << ")\n";
		if (ends_with(algorithm, "Ald"))
		{
			std::cout << "  quantile: " << spec.quantile << "\n";
		}
	}

	if (!spelling.empty())
	{
		std::cout << "  covariance block: "
		          << (spec.uses_covar() ? "on, " + std::to_string(spec.n_psi()) + " free elements"
		                                : std::string("off"))
		          << " (error \"" << check.error_attribute << "\")\n";
	}
	if (!factor_model)
	{
		std::cout << "  variable selection: " << bayests::to_string(spec.varsel) << "\n";
		std::cout << "  structural: " << (spec.structural ? "yes" : "no") << "\n";
	}
	std::cout << "  forecast: "
	          << (spec.h > 0 ? "h = " + std::to_string(spec.h) : std::string("none asked for"))
	          << "\n";
	std::cout << "  chain: " << spec.iterations << " draws kept after " << spec.burnin
	          << " burn-in";
	if (spec.thin > 1)
	{
		std::cout << ", one in " << spec.thin << ", so " << spec.draws() << " run";
	}
	std::cout << "\n";
	std::cout << "  seed: "
	          << (check.seed ? std::to_string(*check.seed)
	                         : std::string("none, so the draws follow the generator's state when the "
	                                       "model's turn comes"))
	          << "\n";
	if (check.has_posterior)
	{
		std::cout << "  posterior: already written, so coefficients will skip this model; "
		             "delete /posterior to re-estimate\n";
	}

	// The warnings: things that do not stop a run and do change what it means.
	if (ends_with(check.error_attribute, "+covar") && !spec.uses_covar())
	{
		std::cout << "  warning: error \"" << check.error_attribute
		          << "\" switches no covariance block on: ";
		if (spelling.empty())
		{
			std::cout << algorithm << " has no covariance block to switch on";
		}
		else if (spec.k <= 1)
		{
			std::cout << "with one variable there is no covariance to model";
		}
		else
		{
			std::cout << algorithm << " is switched on by \"" << spelling << "\"";
		}
		std::cout << "\n";
	}

	if (!factor_model && !vec && check.z_columns > 0 &&
	    check.z_columns != static_cast<arma::uword>(spec.nparams_per_period()))
	{
		// Refused outright when a forecast is asked for -- see
		// require_var_forecast_regressors() -- so this is the no-forecast case.
		std::cout << "  warning: /data/train/z has " << check.z_columns
		          << " columns, but k, p, m, s, n and structural make "
		          << spec.nparams_per_period()
		          << " coefficients per period. The chain runs on z, so the dimensions "
		             "describe a different model, and asking for a forecast would fail\n";
	}

	for (const std::string &name : check.unknown_attributes)
	{
		std::cout << "  warning: /model attribute '" << name << "' is not one any model reads\n";
	}

	for (const std::string &name : check.unread)
	{
		std::cout << "  warning: " << name << " is in the file, but " << algorithm
		          << " never reads it\n";
	}
}

int process_single_file_check(const ModelLocation &location)
{
	std::string algorithm;

	try
	{
		HighFive::File h5 = open_hdf5_file(location.file);
		require_group(h5, location.group);
		algorithm = get_algorithm_type(ModelFile(h5, location.group));
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error processing " << location.describe() << ": " << e.what() << std::endl;
		return 1;
	}

	std::unique_ptr<BaseModel> model;
	try
	{
		model = create_model(algorithm);
	}
	catch (const std::exception &)
	{
		std::cerr << "Error processing " << location.describe() << ": '" << algorithm
		          << "' is not a registered algorithm. The registered ones are:";
		for (const std::string &name : registered_models())
		{
			std::cerr << " " << name;
		}
		std::cerr << std::endl;
		return 1;
	}

	try
	{
		print_report(algorithm, model->check(location));
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error processing " << location.describe() << ": " << algorithm
		          << " refuses this file: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

} // namespace

int check(int argc, char *argv[])
{
	CommandOptions options;
	if (!parse_command_options(argc, argv, "check", false, options))
	{
		return 2;
	}

	return run_over_models(options, process_single_file_check);
}
