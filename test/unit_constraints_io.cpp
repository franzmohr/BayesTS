// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_constraints_io.cpp
/// @brief Checks how a constraint set is read from a file.
///
/// Six one-dimensional datasets under one group. `row`, `period` and
/// `variable` count from one in the file, as selection positions do, and come
/// back counted from zero; `group` counts from zero, zero meaning hard. All
/// four are read as doubles and refused by name when they hold something that
/// is not a whole number -- R writes whole numbers as doubles unless told
/// otherwise, and a conversion by HDF5 would turn a NaN into 0 and 2.5 into 2
/// without a word.

#include "io/hdf5/model_io_common.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using bayests::Constraints;
using bayests::hdf5_io::read_constraints;

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

/// A one-dimensional dataset, the way h5py writes a NumPy vector and the way
/// the format documents these. Replaces one that is already there.
void write_list(const ModelFile &file, const std::string &name, const std::vector<double> &values)
{
    if (file.exist(name))
    {
        file.unlink(name);
    }
    file.createDataSet<double>(name, HighFive::DataSpace::From(values)).write(values);
}

/// Two rows: a pinned value and a two-month average, the second soft.
void write_set(const ModelFile &file, const std::string &group)
{
    write_list(file, group + "/value", {3.0, 0.5});
    write_list(file, group + "/group", {0.0, 1.0});
    write_list(file, group + "/row", {1.0, 2.0, 2.0});
    write_list(file, group + "/period", {4.0, 1.0, 2.0});
    write_list(file, group + "/variable", {1.0, 2.0, 2.0});
    write_list(file, group + "/weight", {1.0, 0.5, 0.5});
}

bool throws(const ModelFile &file, const std::string &group, const std::string &contains)
{
    Constraints out;
    try
    {
        read_constraints(file, group, out);
    }
    catch (const std::exception &e)
    {
        if (std::string(e.what()).find(contains) == std::string::npos)
        {
            std::cerr << "  refused, but the message does not say '" << contains
                      << "': " << e.what() << '\n';
            return false;
        }
        return true;
    }
    return false;
}

void test_absent_is_absent(HighFive::File &h5)
{
    const ModelFile file(h5, "/absent");
    file.createGroup("/data");

    Constraints out;
    out.value = arma::vec{42.0};
    check(!read_constraints(file, "/data/train/constraints", out), "a file with none reads as none");
    check(out.value.n_elem == 1 && out.value(0) == 42.0,
          "and the output is left exactly as the caller had it");
}

void test_round_trip(HighFive::File &h5)
{
    const ModelFile file(h5, "/whole");
    write_set(file, "/data/train/constraints");

    Constraints out;
    check(read_constraints(file, "/data/train/constraints", out), "a full set is found");
    check(arma::approx_equal(out.value, arma::vec{3.0, 0.5}, "absdiff", 0.0) &&
              arma::approx_equal(out.weight, arma::vec{1.0, 0.5, 0.5}, "absdiff", 0.0),
          "values and weights are read as written");
    check(arma::all(out.group == arma::uvec{0, 1}), "groups are read as written, zero hard");
    check(arma::all(out.row == arma::uvec{0, 1, 1}) && arma::all(out.period == arma::uvec{3, 0, 1}) &&
              arma::all(out.variable == arma::uvec{0, 1, 1}),
          "rows, periods and variables come back counted from zero");
}

/// A single row or column is how every other vector in these files is stored,
/// so it reads the same; a matrix is refused, its reading order unknowable.
void test_two_dimensional_spellings(HighFive::File &h5)
{
    const ModelFile file(h5, "/twod");
    write_set(file, "/c");
    write_armadillo_matrix_to_hdf5(file, "/c/weight", arma::rowvec{1.0, 0.5, 0.5}, false);
    write_armadillo_matrix_to_hdf5(file, "/c/period", arma::vec{4.0, 1.0, 2.0}, false);

    Constraints out;
    check(read_constraints(file, "/c", out) &&
              arma::approx_equal(out.weight, arma::vec{1.0, 0.5, 0.5}, "absdiff", 0.0) &&
              arma::all(out.period == arma::uvec{3, 0, 1}),
          "a single row or a single column reads as the list it holds");

    write_armadillo_matrix_to_hdf5(file, "/c/weight", arma::mat{{1.0, 0.5}, {0.5, 1.0}}, false);
    check(throws(file, "/c", "one-dimensional"), "a matrix is refused rather than flattened");
}

void test_group_under_a_model(HighFive::File &h5)
{
    // Under --group the absolute path is resolved inside the model's group.
    const ModelFile file(h5, "/models/one");
    write_set(file, "/data/test/constraints");

    Constraints out;
    check(read_constraints(file, "/data/test/constraints", out) && out.value.n_elem == 2,
          "a set is found inside a model group");
}

void test_a_missing_member_is_refused(HighFive::File &h5)
{
    const ModelFile file(h5, "/partial");
    write_list(file, "/data/train/constraints/value", {1.0});
    check(throws(file, "/data/train/constraints", "has no 'group'"),
          "a set missing one of its six datasets is refused by name");
}

void test_positions_are_whole_numbers(HighFive::File &h5)
{
    const ModelFile zero(h5, "/zero");
    write_set(zero, "/c");
    write_list(zero, "/c/period", {0.0, 1.0, 2.0});
    check(throws(zero, "/c", "at least 1"), "a period of zero is refused: they count from one");

    const ModelFile half(h5, "/half");
    write_set(half, "/c");
    write_list(half, "/c/row", {1.0, 2.5, 2.0});
    check(throws(half, "/c", "not a whole number"), "a row of 2.5 is refused, not truncated");

    const ModelFile nan(h5, "/nan");
    write_set(nan, "/c");
    write_list(nan, "/c/variable", {1.0, arma::datum::nan, 2.0});
    check(throws(nan, "/c", "not a whole number"), "a NaN variable is refused, not read as zero");

    const ModelFile negative(h5, "/negative");
    write_set(negative, "/c");
    write_list(negative, "/c/group", {0.0, -1.0});
    check(throws(negative, "/c", "at least 0"), "a negative group is refused");
}

} // namespace

int main()
{
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "bayests_unit_constraints_io";
    std::filesystem::create_directories(scratch);
    const std::filesystem::path dest = scratch / "constraints.h5";

    try
    {
        std::filesystem::remove(dest);
        HighFive::File h5(dest.string(), HighFive::File::Create);

        test_absent_is_absent(h5);
        test_round_trip(h5);
        test_two_dimensional_spellings(h5);
        test_group_under_a_model(h5);
        test_a_missing_member_is_refused(h5);
        test_positions_are_whole_numbers(h5);
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
