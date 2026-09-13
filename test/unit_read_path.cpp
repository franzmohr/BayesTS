// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// A starting path is read at exactly the size the model says it has, and a
// forecast of a time-varying quantity refuses a file with no sample.
//
// read_path() used to be a reshape and nothing else. arma::reshape() pads a short
// matrix with zeros and cuts a long one, and validate() then saw a path of the
// shape it asks for whatever the file held: a VarTvpGamma file with 264 of the
// 288 starting coefficients passed `bayests check` and a run with exit code 0.
// last_sample_period() stands where `tt - 1` stood in the forecast readers, which
// wrapped round to the largest index there is for a file with no /data/train/y.
//
// An exact identity: no draws, no fixture, no thread pinning.

#include "io/hdf5/model_io_common.h"

#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

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

bool refuses(const std::function<void()> &call)
{
    try
    {
        call();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    using bayests::hdf5_io::last_sample_period;
    using bayests::hdf5_io::read_path;

    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_read_path";
    std::filesystem::create_directories(scratch);
    const std::filesystem::path dest = scratch / "paths.h5";
    std::filesystem::remove(dest);

    {
        HighFive::File h5(dest.string(), HighFive::File::Create);
        const ModelFile file(h5);

        // Two coefficients over three periods, one period per column.
        const arma::mat path = {{1.0, 3.0, 5.0}, {2.0, 4.0, 6.0}};
        write_armadillo_matrix_to_hdf5(file, "/initial/a", path, false);

        check(arma::approx_equal(read_path(file, "/initial/a", 2, 3), path, "absdiff", 0.0),
              "a path of the right size reads back unchanged");
        check(refuses([&] { read_path(file, "/initial/a", 2, 4); }),
              "a path a period short is refused rather than padded with zeros");
        check(refuses([&] { read_path(file, "/initial/a", 2, 2); }),
              "a path a period long is refused rather than cut");
        check(refuses([&] { read_path(file, "/initial/a", 3, 3); }),
              "a path read at the wrong width is refused");
        check(read_path(file, "/initial/a", 2, 0).is_empty(),
              "with no sample to measure against, nothing is read");
    }

    check(last_sample_period(24) == 23, "the last period of 24 is index 23");
    check(last_sample_period(1) == 0, "the last period of one is index 0");
    check(refuses([] { last_sample_period(0); }),
          "a forecast with no sample is refused rather than reading the largest index there is");

    std::filesystem::remove_all(scratch);

    std::cout << (failures == 0 ? "all checks passed" : "SOME CHECKS FAILED") << '\n';
    return failures == 0 ? 0 : 1;
}
