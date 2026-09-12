// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_forecast_regressors_io.cpp
/// @brief Checks that the out-of-sample regressors read back in either layout.
///
/// The forecast input is ForecastData::x, h rows by n_x columns, one period per
/// row. It used to be the SUR spelling of the same thing -- kron(x, I_k), at k
/// times the rows and k times the columns -- and files written then are still
/// out there, so read_forecast_regressors() takes either and hands the samplers
/// the compact one.
///
/// Nothing else pins that. A file carrying only the old spelling would otherwise
/// read as "no forecast regressors", which is not an error: the signal term drops
/// out and every horizon comes from the error distribution alone, written to
/// /posterior/forecast and reported as a success. The compaction is exact -- it
/// is a subscript, not an average -- so the equality below is an identity rather
/// than a tolerance.

#include "io/hdf5/model_io_common.h"

#include <filesystem>
#include <iostream>
#include <string>

using bayests::hdf5_io::read_forecast_regressors;

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

/// Three horizons of four regressors, the shape a k = 2, p = 1, n = 1 model
/// forecasts from: two lag columns and an intercept, plus one more so that a
/// transposed read would not pass by symmetry.
arma::mat sample_regressors()
{
    return arma::mat({{1.0, 2.0, 3.0, 1.0}, {4.0, 5.0, 6.0, 1.0}, {7.0, 8.0, 9.0, 1.0}});
}

void test_compact_is_read_verbatim(HighFive::File &h5)
{
    const ModelFile file(h5, "/compact");
    const arma::mat x = sample_regressors();
    write_armadillo_matrix_to_hdf5(file, "/data/forecast/x", x, false);

    arma::mat out;
    check(read_forecast_regressors(file, 2, out), "a compact x is found");
    check(arma::approx_equal(out, x, "absdiff", 0.0), "a compact x is read unchanged");
}

void test_sur_is_compacted(HighFive::File &h5)
{
    const ModelFile file(h5, "/sur");
    const int k = 2;
    const arma::mat x = sample_regressors();
    const arma::mat z = arma::kron(x, arma::eye<arma::mat>(k, k));
    write_armadillo_matrix_to_hdf5(file, "/data/forecast/z", z, false);

    arma::mat out;
    check(read_forecast_regressors(file, k, out), "an old SUR z is found");
    check(out.n_rows == x.n_rows && out.n_cols == x.n_cols,
          "an old SUR z comes back at one k-th of each dimension");
    check(arma::approx_equal(out, x, "absdiff", 0.0),
          "an old SUR z compacts back to exactly what it was kroneckered from");
}

/// k = 1 is the case where the two layouts coincide, so a file written either way
/// has to read the same -- and the compaction must not mangle it.
void test_single_variable(HighFive::File &h5)
{
    const ModelFile file(h5, "/single");
    const arma::mat x = sample_regressors();
    write_armadillo_matrix_to_hdf5(file, "/data/forecast/z", x, false);

    arma::mat out;
    check(read_forecast_regressors(file, 1, out), "a one-variable z is found");
    check(arma::approx_equal(out, x, "absdiff", 0.0), "a one-variable z is its own compaction");
}

/// Both spellings in one file: the compact one is the current layout and wins,
/// which is what lets a writer add it without having to remove the old one.
void test_compact_wins(HighFive::File &h5)
{
    const ModelFile file(h5, "/both");
    const int k = 2;
    const arma::mat x = sample_regressors();
    const arma::mat stale = arma::kron(x + 100.0, arma::eye<arma::mat>(k, k));
    write_armadillo_matrix_to_hdf5(file, "/data/forecast/x", x, false);
    write_armadillo_matrix_to_hdf5(file, "/data/forecast/z", stale, false);

    arma::mat out;
    check(read_forecast_regressors(file, k, out), "a file with both is found");
    check(arma::approx_equal(out, x, "absdiff", 0.0), "the compact x wins over a stale z");
}

/// No forecast requested. Neither dataset is there, and `out` has to be left
/// exactly as the caller had it -- require_forecast_regressors() is what decides
/// whether that is an error, and it can only do so if this does not throw first.
void test_absent(HighFive::File &h5)
{
    const ModelFile file(h5, "/absent");
    file.createGroup("/data");

    arma::mat out(1, 1, arma::fill::value(42.0));
    check(!read_forecast_regressors(file, 2, out), "an absent forecast reads as absent");
    check(out.n_rows == 1 && out.n_cols == 1 && out(0, 0) == 42.0,
          "an absent forecast leaves the output alone");
}

/// A z whose dimensions are not multiples of k is not a kron of anything, and
/// silently taking every k-th element of it would invent regressors.
void test_ragged_sur_is_refused(HighFive::File &h5)
{
    const ModelFile file(h5, "/ragged");
    write_armadillo_matrix_to_hdf5(file, "/data/forecast/z", sample_regressors(), false);

    arma::mat out;
    bool threw = false;
    try
    {
        read_forecast_regressors(file, 2, out);
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "a z that is not a multiple of k is refused");
}

} // namespace

int main()
{
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_forecast_regressors_io";
    std::filesystem::create_directories(scratch);
    const std::filesystem::path dest = scratch / "forecast_regressors.h5";

    try
    {
        std::filesystem::remove(dest);
        HighFive::File h5(dest.string(), HighFive::File::Create);

        test_compact_is_read_verbatim(h5);
        test_sur_is_compacted(h5);
        test_single_variable(h5);
        test_compact_wins(h5);
        test_absent(h5);
        test_ragged_sur_is_refused(h5);
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
