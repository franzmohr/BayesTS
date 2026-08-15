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
//   bayests_golden <fixture.h5> [more.h5 ...]
//
// The model is taken from /model/algorithm. The checked-in VarNormalWishart
// fixtures predate that attribute, so its absence means VarNormalWishart;
// anything make_model_fixture writes names itself.
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
    "/posterior/psi/coeffs",
    "/posterior/psi/lambda",
    "/posterior/psi/sigma",
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

void report_if_present(const HighFive::File &file, const std::string &dataset)
{
    if (!dataset_has_data(file, dataset))
    {
        printf("  %-30s absent\n", dataset.c_str());
        return;
    }
    fingerprint(dataset, hdf5_dataset_to_armadillo_matrix_double(file, dataset));
}

// The fixtures carry a full /posterior group, and every entry point returns
// early when its output already exists. Copy the fixture and clear them.
std::filesystem::path stage_fixture(const std::filesystem::path &fixture,
                                    const std::filesystem::path &scratch)
{
    std::filesystem::create_directories(scratch);
    std::filesystem::path staged = scratch / fixture.filename();
    std::filesystem::copy_file(fixture, staged,
                               std::filesystem::copy_options::overwrite_existing);

    HighFive::File file = open_hdf5_file_readwrite(staged);
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

std::string model_of(const std::filesystem::path &staged)
{
    HighFive::File file = open_hdf5_file(staged);
    if (!attribute_exists(file, "/model", "algorithm"))
    {
        return "VarNormalWishart";
    }
    return get_algorithm_type(file);
}

int run_fixture(const std::filesystem::path &fixture, const std::filesystem::path &scratch)
{
    std::filesystem::path staged = stage_fixture(fixture, scratch);
    const std::string model_type = model_of(staged);

    printf("%s [%s]\n", fixture.filename().string().c_str(), model_type.c_str());

    auto model = create_model(model_type);

    // One seed per entry point, so a change in one does not shift the stream
    // seen by the next and turn a single regression into three.
    arma::arma_rng::set_seed(kSeed);
    model->draw_coefficients(staged);

    arma::arma_rng::set_seed(kSeed);
    model->log_likelihood(staged);

    arma::arma_rng::set_seed(kSeed);
    model->forecast(staged);

    HighFive::File file = open_hdf5_file(staged);
    for (const char *dataset : kOutputs)
    {
        report_if_present(file, dataset);
    }
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <fixture.h5> [more.h5 ...]\n";
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
    for (int i = 1; i < argc; ++i)
    {
        try
        {
            failures += run_fixture(argv[i], scratch);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error processing " << argv[i] << ": " << e.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
