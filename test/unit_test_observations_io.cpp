// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_test_observations_io.cpp
/// @brief Checks what `/data/test/y` is read as and what it is refused for.
///
/// The realised values are the one thing in `/data` that no sampler reads: they
/// are what a forecast is scored against rather than anything it is estimated
/// from. That makes the reader the only place their shape is ever looked at, so
/// the shape has to be refused here or not at all -- a wrong one would otherwise
/// surface as a forecast error of the right size and the wrong meaning.
///
/// Short of the horizon is not wrong. An expanding window whose last windows run
/// past the end of the sample realises fewer periods than it forecasts, and the
/// ones it does realise are the ones that can be scored.

#include "io/hdf5/model_io_common.h"

#include <filesystem>
#include <iostream>
#include <string>

using bayests::VarSpec;
using bayests::hdf5_io::read_test_observations;

namespace
{

int failures = 0;

void check(bool condition, const std::string &what)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
        return;
    }
    std::cout << "ok: " << what << '\n';
}

VarSpec spec_of(int k, int h)
{
    VarSpec spec;
    spec.k = k;
    spec.h = h;
    return spec;
}

/// Four periods of two variables, every element distinct so that a transposed
/// write cannot pass by symmetry.
arma::mat realised(arma::uword periods = 4, arma::uword variables = 2)
{
    arma::mat y(periods, variables);
    for (arma::uword i = 0; i < y.n_rows; i++)
    {
        for (arma::uword j = 0; j < y.n_cols; j++)
        {
            y(i, j) = static_cast<double>(1 + i * y.n_cols + j);
        }
    }
    return y;
}

bool throws(const ModelFile &file, const VarSpec &spec)
{
    arma::mat out;
    try
    {
        read_test_observations(file, spec, out);
    }
    catch (const std::exception &)
    {
        return true;
    }
    return false;
}

void test_absent_is_absent(HighFive::File &h5)
{
    const ModelFile file(h5, "/absent");
    file.createGroup("/data");

    arma::mat out(1, 1, arma::fill::value(42.0));
    check(!read_test_observations(file, spec_of(2, 4), out), "a file with none reads as none");
    check(out.n_rows == 1 && out.n_cols == 1 && out(0, 0) == 42.0,
          "and the output is left exactly as the caller had it");
}

void test_whole_horizon(HighFive::File &h5)
{
    const ModelFile file(h5, "/whole");
    const arma::mat y = realised();
    write_armadillo_matrix_to_hdf5(file, "/data/test/y", y, false);

    arma::mat out;
    check(read_test_observations(file, spec_of(2, 4), out), "a full horizon is found");
    check(arma::approx_equal(out, y, "absdiff", 0.0),
          "and read one row per period and one column per variable");
}

void test_short_of_the_horizon(HighFive::File &h5)
{
    const ModelFile file(h5, "/short");
    const arma::mat y = realised(2);
    write_armadillo_matrix_to_hdf5(file, "/data/test/y", y, false);

    arma::mat out;
    check(read_test_observations(file, spec_of(2, 4), out), "fewer periods than h is found");
    check(out.n_rows == 2, "and kept at the length it has, not padded to the horizon");
}

/// More periods than the model forecasts describes a horizon other than the one
/// `/model` asks for, and nothing downstream could say which of the two is meant.
void test_past_the_horizon_is_refused(HighFive::File &h5)
{
    const ModelFile file(h5, "/long");
    write_armadillo_matrix_to_hdf5(file, "/data/test/y", realised(6), false);
    check(throws(file, spec_of(2, 4)), "more periods than h is refused");
}

void test_wrong_width_is_refused(HighFive::File &h5)
{
    const ModelFile file(h5, "/wide");
    write_armadillo_matrix_to_hdf5(file, "/data/test/y", realised(4, 3), false);
    check(throws(file, spec_of(2, 4)), "a width that is not k is refused");

    // The stacked spelling `/data/train/y` also accepts. Eight numbers in one
    // column are the same eight numbers, and which of the two orders they are in
    // cannot be recovered, so scoring against them would be a guess.
    const ModelFile stacked(h5, "/stacked");
    write_armadillo_matrix_to_hdf5(stacked, "/data/test/y", arma::vectorise(realised()), false);
    check(throws(stacked, spec_of(2, 4)), "the stacked layout is refused rather than guessed at");
}

/// A file may carry what it is to be scored on before it is told how far to
/// forecast. Refusing that here would fail a sampler over data no sampler reads;
/// `bayests check` reports it instead.
void test_no_horizon_is_read(HighFive::File &h5)
{
    const ModelFile file(h5, "/nohorizon");
    write_armadillo_matrix_to_hdf5(file, "/data/test/y", realised(), false);

    arma::mat out;
    check(read_test_observations(file, spec_of(2, 0), out),
          "realised values without a horizon are read rather than refused");
    check(out.n_rows == 4, "and are all there");
}

} // namespace

int main()
{
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_test_observations_io";
    std::filesystem::create_directories(scratch);
    const std::filesystem::path dest = scratch / "test_observations.h5";

    try
    {
        std::filesystem::remove(dest);
        HighFive::File h5(dest.string(), HighFive::File::Create);

        test_absent_is_absent(h5);
        test_whole_horizon(h5);
        test_short_of_the_horizon(h5);
        test_past_the_horizon_is_refused(h5);
        test_wrong_width_is_refused(h5);
        test_no_horizon_is_read(h5);
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: threw: " << e.what() << '\n';
        ++failures;
    }

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "all checks passed\n";
    return 0;
}
