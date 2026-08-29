// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// What --group resolves to, and that it resolves to nothing when it is absent.
//
// The io layer names absolute paths -- "/model", "/data/train/y" -- and ModelFile
// puts the group in front of every one of them. Two things have to hold for that
// to be safe, and neither is a property of the samplers, so neither is covered by
// the golden harness:
//
//   * a model written under a group is at that group and nowhere else. In HDF5 a
//     leading slash resolves from the root of the file even through a group
//     handle, so a prefix that was silently dropped would write to the root and
//     read back the same numbers -- passing a round-trip test while making the
//     group argument a no-op. The check below therefore looks at the raw paths
//     in the file, not only at what comes back out.
//
//   * an empty group leaves every path exactly as it was, since that is what
//     every existing file, fixture and command line without --group relies on.
//
// An exact identity on both counts: no draws, no fixture, no thread pinning.

#include "io/hdf5/hdf5_and_armadillo.h"

#include <filesystem>
#include <iostream>
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

void check_equal(const std::string &got, const std::string &expected, const std::string &what)
{
    check(got == expected, what + " (got '" + got + "', expected '" + expected + "')");
}

/// The spellings a caller may write, and the ones that cannot name a group.
void test_normalization()
{
    check_equal(normalize_hdf5_group(""), "", "an absent group is the root");
    check_equal(normalize_hdf5_group("/"), "", "the root spelled as a slash is the root");
    check_equal(normalize_hdf5_group("/models/3"), "/models/3", "an absolute group is left alone");
    check_equal(normalize_hdf5_group("models/3"), "/models/3", "a relative group gains a slash");
    check_equal(normalize_hdf5_group("/models/3/"), "/models/3", "a trailing slash is dropped");
    check_equal(normalize_hdf5_group(normalize_hdf5_group("models/3/")), "/models/3",
                "normalization is idempotent");

    for (const char *bad : {"/models//3", "/models/./3", "/models/../3", "//"})
    {
        bool threw = false;
        try
        {
            normalize_hdf5_group(bad);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        check(threw, std::string("'") + bad + "' is refused");
    }
}

void test_resolution(HighFive::File &h5)
{
    const ModelFile root(h5);
    check_equal(root.resolve("/model"), "/model", "a model at the root resolves to the same path");
    check_equal(root.group(), "", "a model at the root reports no group");

    const ModelFile nested(h5, "/models/3");
    check_equal(nested.resolve("/model"), "/models/3/model", "a group goes in front of the path");
    check_equal(nested.resolve("/posterior/a/coeffs"), "/models/3/posterior/a/coeffs",
                "a group goes in front of a deep path");
    check_equal(nested.group(), "/models/3", "a nested model reports its group");
}

/// Written under a group, read back through the same group, and looked for at
/// the two paths that say the group was honoured: present under the group,
/// absent at the root.
void test_round_trip(HighFive::File &h5)
{
    const arma::mat values = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    const ModelFile nested(h5, "models/3"); // relative on purpose: normalized on the way in
    write_armadillo_matrix_to_hdf5(nested, "/data/train/z", values, false);
    write_dataset_double(nested, "/priors/u_sigma/df", 3.0);
    nested.createGroup("/model");

    check(h5.exist("/models/3/data/train/z"), "the dataset is under the group");
    check(!h5.exist("/data/train/z"), "the dataset is not at the root");
    check(h5.exist("/models/3/priors/u_sigma/df"), "the scalar is under the group");
    check(!h5.exist("/priors/u_sigma/df"), "the scalar is not at the root");

    // The intermediate groups the prefix needs are created by HighFive on the
    // way to the dataset, which is what lets a model be written into a file that
    // has never heard of its group.
    check(h5.exist("/models"), "the intermediate group was created");

    const arma::mat read_back = hdf5_dataset_to_armadillo_matrix_double(nested, "/data/train/z");
    check(arma::approx_equal(read_back, values, "absdiff", 0.0),
          "the matrix reads back through the group unchanged");
    check(dataset_has_data(nested, "/data/train/z"), "the dataset is seen through the group");
    check(!dataset_has_data(ModelFile(h5), "/data/train/z"),
          "the dataset is not seen from the root");

    // require_group() is what turns a --group naming nothing into one error
    // rather than a missing dataset further in.
    bool threw = false;
    try
    {
        require_group(h5, "/models/9");
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "a group that is not in the file is refused");

    threw = false;
    try
    {
        require_group(h5, "/models/3/data/train/z");
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "a dataset named as a group is refused");

    require_group(h5, "/models/3"); // must not throw
    require_group(h5, "");          // the root always is a group
    check(true, "a group that is in the file is accepted");
}

/// The same writes with no group, which have to land exactly where they landed
/// before ModelFile existed.
void test_root_unchanged(HighFive::File &h5)
{
    const arma::mat values = {{7.0, 8.0}};

    const ModelFile root(h5);
    write_armadillo_matrix_to_hdf5(root, "/data/forecast/z", values, false);

    check(h5.exist("/data/forecast/z"), "an ungrouped write lands at the root");

    const arma::mat read_back = hdf5_dataset_to_armadillo_matrix_double(root, "/data/forecast/z");
    check(arma::approx_equal(read_back, values, "absdiff", 0.0),
          "an ungrouped matrix reads back unchanged");

    // The implicit conversion, which is what keeps every caller that has no
    // group to name compiling and reading the root.
    check(dataset_has_data(h5, "/data/forecast/z"), "a bare file still reads the root");
}

} // namespace

int main()
{
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_model_group";
    std::filesystem::create_directories(scratch);
    const std::filesystem::path dest = scratch / "grouped.h5";

    try
    {
        test_normalization();

        std::filesystem::remove(dest);
        HighFive::File h5(dest.string(), HighFive::File::Create);

        test_resolution(h5);
        test_round_trip(h5);
        test_root_unchanged(h5);
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
