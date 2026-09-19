// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_forecast_group_io.cpp
/// @brief Checks where a forecast lands and what it replaces.
///
/// `/posterior/forecast` is a group, and the paths are one member of it:
/// `forecasts`, beside the `errors` and `loglik` a host writes when it holds the
/// observations the horizon realised. Before 0.3.0 the paths were a dataset at
/// the group's own path, so the migration is a rerun of `bayests forecasts` over
/// such a file -- which only works if the writer clears the name first. HDF5
/// refuses to create a dataset below a dataset, and it refuses with a message
/// about the path rather than about the version of the file, so the case is
/// worth pinning here rather than leaving to whoever meets it.
///
/// The `start`, `end` and `thin` attributes are the second half of the change.
/// Every dataset under `/posterior/<block>/` carried them and these two did not,
/// which an R session sees as a forecast it cannot hand to `coda` without
/// rebuilding what the file already knows.

#include "io/hdf5/model_io_common.h"

#include <filesystem>
#include <iostream>
#include <string>

using bayests::ForecastDraws;
using bayests::hdf5_io::write_forecast;
using bayests::hdf5_io::write_log_likelihood;

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

/// Two horizons of three variables over four draws, with every element distinct
/// so that a transposed write cannot pass by symmetry.
arma::mat sample_paths(double offset = 0.0)
{
    arma::mat values(6, 4);
    for (arma::uword i = 0; i < values.n_rows; i++)
    {
        for (arma::uword j = 0; j < values.n_cols; j++)
        {
            values(i, j) = offset + static_cast<double>(1 + i * values.n_cols + j);
        }
    }
    return values;
}

/// The dataset as HDF5 holds it: one row per quantity, one column per draw.
arma::mat stored(const HighFive::File &h5, const std::string &path)
{
    std::vector<std::vector<double>> data;
    h5.getDataSet(path).read(data);
    arma::mat out(data.size(), data.empty() ? 0 : data[0].size());
    for (arma::uword i = 0; i < out.n_rows; i++)
    {
        for (arma::uword j = 0; j < out.n_cols; j++)
        {
            out(i, j) = data[i][j];
        }
    }
    return out;
}

bool is_group(const HighFive::File &h5, const std::string &path)
{
    return h5.exist(path) && h5.getObjectType(path) == HighFive::ObjectType::Group;
}

int attribute(const HighFive::File &h5, const std::string &path, const std::string &name)
{
    int value = 0;
    h5.getDataSet(path).getAttribute(name).read(value);
    return value;
}

void test_paths_land_in_the_group(HighFive::File &h5)
{
    const ModelFile file(h5, "/plain");
    ForecastDraws forecast;
    forecast.values = sample_paths();
    write_forecast(file, forecast);

    check(is_group(h5, "/plain/posterior/forecast"), "/posterior/forecast is a group");
    check(h5.exist("/plain/posterior/forecast/forecasts"), "the paths are its 'forecasts' member");
    check(arma::approx_equal(stored(h5, "/plain/posterior/forecast/forecasts"), forecast.values,
                             "absdiff", 0.0),
          "the paths are stored one row per quantity and one column per draw");
}

/// Without `/model/thin` the chain is unthinned, so the draws kept are 1 to the
/// number of them. The attributes say so rather than being left off.
void test_mcpar_without_thinning(HighFive::File &h5)
{
    const ModelFile file(h5, "/unthinned");
    ForecastDraws forecast;
    forecast.values = sample_paths();
    write_forecast(file, forecast);

    const std::string path = "/unthinned/posterior/forecast/forecasts";
    check(attribute(h5, path, "start") == 1, "start is 1 without thinning");
    check(attribute(h5, path, "end") == 4, "end is the number of draws without thinning");
    check(attribute(h5, path, "thin") == 1, "thin is 1 without thinning");
}

/// With thinning they count iterations after the burn-in, as the block datasets
/// do: the draws kept are thin, 2 thin, ..., draws * thin.
void test_mcpar_with_thinning(HighFive::File &h5)
{
    const ModelFile file(h5, "/thinned");
    file.createGroup("/model");
    h5.getGroup("/thinned/model").createAttribute("thin", 5);

    ForecastDraws forecast;
    forecast.values = sample_paths();
    write_forecast(file, forecast);
    write_log_likelihood(file, arma::mat(4, 7, arma::fill::ones));

    const std::string path = "/thinned/posterior/forecast/forecasts";
    check(attribute(h5, path, "start") == 5, "start is thin");
    check(attribute(h5, path, "end") == 20, "end is draws times thin");
    check(attribute(h5, path, "thin") == 5, "thin is thin");

    // The likelihood is draws by periods, so its columns are the periods and it
    // is the rows that the attributes count.
    const std::string loglik = "/thinned/posterior/loglik";
    check(attribute(h5, loglik, "start") == 5, "the log likelihood carries start");
    check(attribute(h5, loglik, "end") == 20, "the log likelihood counts draws, not periods");
    check(attribute(h5, loglik, "thin") == 5, "the log likelihood carries thin");
}

/// A file written before 0.3.0. The dataset at the group's path has to give way,
/// or the write below fails on the name.
void test_a_legacy_dataset_is_replaced(HighFive::File &h5)
{
    const ModelFile file(h5, "/legacy");
    const arma::mat old = sample_paths(1000.0);
    file.createGroup("/posterior");
    write_armadillo_matrix_to_hdf5(file, "/posterior/forecast", arma::trans(old), false);
    check(!is_group(h5, "/legacy/posterior/forecast"), "the old layout is a dataset");

    ForecastDraws forecast;
    forecast.values = sample_paths();
    write_forecast(file, forecast);

    check(is_group(h5, "/legacy/posterior/forecast"), "a rerun turns it into a group");
    check(arma::approx_equal(stored(h5, "/legacy/posterior/forecast/forecasts"), forecast.values,
                             "absdiff", 0.0),
          "and the new paths are what the group holds");
}

/// The ordinary rerun, over a file already in the new layout: the member is
/// replaced and the group is not disturbed.
void test_a_rerun_overwrites_the_member(HighFive::File &h5)
{
    const ModelFile file(h5, "/rerun");
    ForecastDraws first;
    first.values = sample_paths();
    write_forecast(file, first);

    ForecastDraws second;
    second.values = sample_paths(500.0);
    write_forecast(file, second);

    check(is_group(h5, "/rerun/posterior/forecast"), "the group survives a second write");
    check(arma::approx_equal(stored(h5, "/rerun/posterior/forecast/forecasts"), second.values,
                             "absdiff", 0.0),
          "the second forecast replaces the first");
}

} // namespace

int main()
{
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_forecast_group_io";
    std::filesystem::create_directories(scratch);
    const std::filesystem::path dest = scratch / "forecast_group.h5";

    try
    {
        std::filesystem::remove(dest);
        HighFive::File h5(dest.string(), HighFive::File::Create);

        test_paths_land_in_the_group(h5);
        test_mcpar_without_thinning(h5);
        test_mcpar_with_thinning(h5);
        test_a_legacy_dataset_is_replaced(h5);
        test_a_rerun_overwrites_the_member(h5);
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
