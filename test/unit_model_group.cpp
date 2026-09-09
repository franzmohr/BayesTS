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
#include <vector>

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

/// A group that list_model_groups() has to recognise: a /model subgroup with an
/// algorithm attribute on it, which is what get_algorithm_type() reads.
void write_model_marker(HighFive::File &h5, const std::string &group)
{
    const ModelFile model(h5, group);
    HighFive::Group model_group = model.createGroup("/model");
    model_group.createAttribute<std::string>("algorithm", std::string("VarNormalWishart"));

    // A model's own subtree, which the search must never mistake for more
    // models -- and must not descend into looking for them.
    write_dataset_double(model, "/priors/u_sigma/df", 3.0);
    model.createGroup("/posterior");
}

std::string joined(const std::vector<std::string> &groups)
{
    std::string result;
    for (const std::string &group : groups)
    {
        result += (result.empty() ? "" : ",") + (group.empty() ? std::string("<root>") : group);
    }
    return result;
}

/// Which groups of a file hold a model. This is what --all-groups walks, so a
/// group missed here is a model silently not sampled, and a group returned that
/// is not one is a run that fails on a file the caller never asked about.
void test_model_discovery()
{
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_model_group";
    const std::filesystem::path dest = scratch / "several.h5";
    std::filesystem::remove(dest);

    {
        HighFive::File h5(dest.string(), HighFive::File::Create);

        // Three models, written out of order so that the sort is doing work.
        write_model_marker(h5, "/submodels/US/001");
        write_model_marker(h5, "/submodels/CA/001");
        write_model_marker(h5, "/submodels/US/002");

        // Not a model: the group a GVAR's shared data would sit in, plus a
        // group called "model" that carries no algorithm.
        const ModelFile root(h5);
        write_dataset_double(root, "/global/index", 1.0);
        h5.createGroup("/decoy/model");
    }

    HighFive::File h5 = open_hdf5_file(dest);

    check_equal(joined(list_model_groups(h5, "")),
                "/submodels/CA/001,/submodels/US/001,/submodels/US/002",
                "every model in the file, sorted");
    check_equal(joined(list_model_groups(h5, "/submodels/US")),
                "/submodels/US/001,/submodels/US/002",
                "a root restricts the walk to what is below it");
    check_equal(joined(list_model_groups(h5, "/submodels/US/001")), "/submodels/US/001",
                "a root that is itself a model is the only result");
    check_equal(joined(list_model_groups(h5, "/global")), "",
                "a group with no model under it comes back empty");
    check_equal(joined(list_model_groups(h5, "/decoy")), "",
                "a 'model' group with no algorithm attribute is not a model");

    // The search stops at a model, so nothing from inside one is ever returned.
    for (const std::string &group : list_model_groups(h5, ""))
    {
        check(group.find("/priors") == std::string::npos &&
                  group.find("/posterior") == std::string::npos &&
                  group.find("/model") == std::string::npos,
              "'" + group + "' is a model, not something inside one");
    }

    // A root that is not a group at all is a command line to fix, and is worth
    // failing on rather than reporting as a file with no models in it.
    bool threw = false;
    try
    {
        list_model_groups(h5, "/does/not/exist");
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "a root that is not in the file is refused");
}

/// A model at the root is found as the root, which is what keeps --all-groups
/// working on every single-model file already written.
void test_model_discovery_at_root()
{
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_model_group";
    const std::filesystem::path dest = scratch / "single.h5";
    std::filesystem::remove(dest);

    {
        HighFive::File h5(dest.string(), HighFive::File::Create);
        write_model_marker(h5, "");
    }

    HighFive::File h5 = open_hdf5_file(dest);
    check_equal(joined(list_model_groups(h5, "")), "<root>",
                "a model at the root is found as the root");
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

        test_model_discovery();
        test_model_discovery_at_root();
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
