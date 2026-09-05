// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// Regression harness for the samplers.
//
// The samplers are stochastic, so a refactor can only be checked against the
// fixtures if the RNG stream is pinned. This tool seeds Armadillo's generator,
// runs the model over a copy of a fixture with its /posterior group removed,
// and prints a fingerprint of every matrix that comes back. Two builds that
// consume the RNG in the same order print the same fingerprint, digit for
// digit; any change in the order of the draws shows up immediately.
//
//   bayests_golden [--group <path>] <fixture.h5> [more.h5 ...]
//
// --group is the group each model's tree hangs under inside its file, for a
// fixture that does not sit at the root -- make_model_fixture writes one there
// when it is given a group of its own. The same group is used for every fixture
// named, since the point of the option here is to run a model written under a
// group through the same fingerprints as one written at the root.
//
// The model is taken from /model/algorithm. A recorded file may predate that
// attribute, in which case its absence means VarNormalWishart; anything
// make_model_fixture writes names itself.
//
// Exit codes: 2 for an unusable command line, 1 if a fixture throws or if any of
// the three entry points produced nothing -- no draws, no log likelihood, or no
// forecast where the horizon is positive. That last check is what keeps a green
// run from meaning less than it looks like: before it, a file the sampler
// rejected outright ran, printed `absent` fourteen times and passed. The
// front-ends throw now rather than swallowing, and run() below catches each
// stage so the other two are still attempted and the fingerprints still
// printed -- the reason goes to stderr and the empty stage is the failure. It is not a check that the model wrote
// everything it should: `absent` remains the right fingerprint for a dataset
// that belongs to another model, and only a per-model table could tell the two
// apart.
//
// Threading has to be pinned as well -- a multi-threaded BLAS reassociates
// floating-point reductions -- so the caller must set OMP_NUM_THREADS=1 and
// OPENBLAS_NUM_THREADS=1. The tool refuses to run otherwise rather than emit a
// fingerprint that cannot be reproduced.

#include "models/models.h"
#include "io/hdf5/hdf5_and_armadillo.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

constexpr arma::arma_rng::seed_type kSeed = 20260808u;

// Datasets the models refuse to overwrite. They have to go before a rerun.
// This is the union over every sampler; the ones a given model does not produce
// are reported as absent, which is itself part of the fingerprint.
//
// A path missing from this list is invisible twice over: stage_fixture() leaves
// it in place, so the sampler that would have written it returns early and the
// rerun silently reports the previous run's numbers -- and nothing fingerprints
// it, so a draw that stopped being written at all still passes. /posterior/beta
// was absent here while the VEC was being added, which is exactly how its
// cointegration draws went missing without a test noticing.
constexpr const char *kOutputs[] = {
    "/posterior/a/coeffs",
    "/posterior/a/lambda",
    "/posterior/a/sigma",
    "/posterior/beta/coeffs",
    // The dynamic factor model's own three. `lambda/coeffs` is the loading
    // matrix and shares no path with the `a/lambda` above it, which is a vector
    // of inclusion indicators; `factors/coeffs` is the drawn state path, which
    // is part of that posterior rather than derivable from it.
    "/posterior/lambda/coeffs",
    "/posterior/lambda/sigma",
    "/posterior/factors/coeffs",
    "/posterior/v_sigma_inv/coeffs",
    "/posterior/psi/coeffs",
    "/posterior/psi/lambda",
    "/posterior/psi/sigma",
    "/posterior/u_scale/coeffs",
    "/posterior/u_omega_inv/coeffs",
    "/posterior/u_sigma_inv/coeffs",
    "/posterior/loglik",
    "/posterior/forecast",
};

// A few order-sensitive reductions. Printed at full double precision: the
// point is to catch a one-draw shift in the RNG stream, not to be readable.
void fingerprint(const std::string &label, const arma::mat &m)
{
    if (m.n_elem == 0)
    {
        printf("  %-30s absent\n", label.c_str());
        return;
    }

    // Weighting by position makes the checksum sensitive to ordering, so a
    // transposed or permuted result cannot pass by having the same total.
    double weighted = 0.0;
    for (arma::uword j = 0; j < m.n_cols; ++j)
    {
        for (arma::uword i = 0; i < m.n_rows; ++i)
        {
            weighted += m(i, j) * static_cast<double>(1 + (i + 7 * j) % 1021);
        }
    }

    printf("  %-30s %llux%llu sum=%.17g ssq=%.17g min=%.17g max=%.17g w=%.17g\n",
           label.c_str(),
           static_cast<unsigned long long>(m.n_rows),
           static_cast<unsigned long long>(m.n_cols),
           arma::accu(m), arma::accu(arma::square(m)),
           m.min(), m.max(), weighted);
}

void report(const ModelFile &file, const std::string &dataset, bool present)
{
    if (!present)
    {
        printf("  %-30s absent\n", dataset.c_str());
        return;
    }
    fingerprint(dataset, hdf5_dataset_to_armadillo_matrix_double(file, dataset));
}

/// The two entries of kOutputs written by an entry point of their own rather
/// than by the draw. Everything else there comes out of draw_coefficients, and
/// which of those a given model writes depends on the model.
constexpr const char *kLoglik = "/posterior/loglik";
constexpr const char *kForecast = "/posterior/forecast";

/// The horizon the file asks for. Absent means zero, which is what a fixture
/// written with h=0 carries -- no forecast regressors and no attribute.
int forecast_horizon(const ModelFile &file)
{
    if (!attribute_exists(file, "/model", "h"))
    {
        return 0;
    }
    return get_attribute_int(file, "/model", "h");
}

// The fixtures carry a full /posterior group, and every entry point returns
// early when its output already exists. Copy the fixture and clear them.
std::filesystem::path stage_fixture(const std::filesystem::path &fixture,
                                    const std::filesystem::path &scratch,
                                    const std::string &group)
{
    std::filesystem::create_directories(scratch);
    std::filesystem::path staged = scratch / fixture.filename();
    std::filesystem::copy_file(fixture, staged,
                               std::filesystem::copy_options::overwrite_existing);

    HighFive::File h5 = open_hdf5_file_readwrite(staged);
    require_group(h5, group);

    const ModelFile file(h5, group);
    for (const char *dataset : kOutputs)
    {
        if (file.exist(dataset))
        {
            file.unlink(dataset);
        }
    }
    return staged;
}

bool threads_pinned()
{
    const char *omp = std::getenv("OMP_NUM_THREADS");
    const char *blas = std::getenv("OPENBLAS_NUM_THREADS");
    return omp && blas && std::string(omp) == "1" && std::string(blas) == "1";
}

std::string model_of(const std::filesystem::path &staged, const std::string &group)
{
    HighFive::File h5 = open_hdf5_file(staged);
    const ModelFile file(h5, group);
    if (!attribute_exists(file, "/model", "algorithm"))
    {
        return "VarNormalWishart";
    }
    return get_algorithm_type(file);
}

int run_fixture(const std::filesystem::path &fixture, const std::filesystem::path &scratch,
                const std::string &group)
{
    std::filesystem::path staged = stage_fixture(fixture, scratch, group);
    const ModelLocation location{staged, group};
    const std::string model_type = model_of(staged, group);

    printf("%s [%s]\n", fixture.filename().string().c_str(), model_type.c_str());

    auto model = create_model(model_type);

    // The entry points throw on a stage that cannot do what it was asked --
    // that is how `bayests` reaches exit 1 -- so each is caught here. Caught
    // rather than allowed to escape, because all three have to be attempted
    // and the fingerprints printed either way: what a fixture wrote before it
    // failed is most of the evidence for why. The dataset checks below turn a
    // stage that produced nothing into the failure, whether it threw or not.
    auto run = [&](const char *stage, void (BaseModel::*entry)(const ModelLocation &)) {
        // One seed per entry point, so a change in one does not shift the
        // stream seen by the next and turn a single regression into three.
        arma::arma_rng::set_seed(kSeed);
        try
        {
            (model.get()->*entry)(location);
        }
        catch (const std::exception &e)
        {
            std::cerr << fixture.filename().string() << ": " << stage << " threw: "
                      << e.what() << std::endl;
        }
    };

    run("draw_coefficients", &BaseModel::draw_coefficients);
    run("log_likelihood", &BaseModel::log_likelihood);
    run("forecast", &BaseModel::forecast);

    HighFive::File h5 = open_hdf5_file(staged);
    const ModelFile file(h5, group);

    int draws_written = 0;
    bool loglik_written = false;
    bool forecast_written = false;
    for (const char *dataset : kOutputs)
    {
        const bool present = dataset_has_data(file, dataset);
        report(file, dataset, present);

        if (std::string(dataset) == kLoglik)
        {
            loglik_written = present;
        }
        else if (std::string(dataset) == kForecast)
        {
            forecast_written = present;
        }
        else if (present)
        {
            ++draws_written;
        }
    }

    // A stage that threw is reported by run() above and then lands here as an
    // absent dataset, which is also what a stage that returned quietly having
    // done nothing looks like. Both are caught the same way. Absent is a
    // legitimate fingerprint for a dataset another model owns; it is not a
    // legitimate outcome for a whole stage.
    // Checked here rather than against a per-model list of expected datasets:
    // this needs no table, and the failure it catches is a stage that produced
    // nothing at all, which is what a rejected input looks like.
    const std::string name = fixture.filename().string();
    int failures = 0;

    if (draws_written == 0)
    {
        std::cerr << name << ": draw_coefficients wrote no posterior draws\n";
        ++failures;
    }
    if (!loglik_written)
    {
        std::cerr << name << ": log_likelihood wrote no " << kLoglik << '\n';
        ++failures;
    }
    if (forecast_horizon(file) > 0 && !forecast_written)
    {
        std::cerr << name << ": h is positive but forecast wrote no " << kForecast << '\n';
        ++failures;
    }

    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char *argv[])
{
    std::string group;
    std::vector<std::string> fixtures;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--group")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Error: --group needs the path of a group\n";
                return 2;
            }
            group = argv[++i];
        }
        else if (arg.rfind("--group=", 0) == 0)
        {
            group = arg.substr(std::string("--group=").size());
        }
        else
        {
            fixtures.push_back(arg);
        }
    }

    if (fixtures.empty())
    {
        std::cerr << "Usage: " << argv[0] << " [--group <path>] <fixture.h5> [more.h5 ...]\n";
        return 2;
    }

    try
    {
        group = normalize_hdf5_group(group);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }

    if (!threads_pinned())
    {
        std::cerr << "Refusing to run: set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 "
                     "so the fingerprints are reproducible.\n";
        return 2;
    }

    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_golden";

    int failures = 0;
    for (const std::string &fixture : fixtures)
    {
        try
        {
            failures += run_fixture(fixture, scratch, group);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error processing " << fixture << ": " << e.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
