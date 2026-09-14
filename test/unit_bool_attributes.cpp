// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// How a boolean attribute is read, whichever way its writer encoded it.
//
// A boolean has no HDF5 type of its own, so every writer picks one. h5py and
// HighFive store an enumeration with the members FALSE and TRUE. R's hdf5r stores
// a logical as an enumeration over an unsigned byte with a third member, NA. Read
// through HighFive's conversion to bool, that R enumeration came back false
// whatever it held, so /model/structural = TRUE written by bvartools reached the
// samplers as false: a structural VAR ran with its contemporaneous columns as
// ordinary regressors, and a structural VEC was refused for having more columns
// in z than its dimensions describe.
//
// An exact identity: no draws, no fixture, no thread pinning.

#include "io/hdf5/hdf5_and_armadillo.h"

#include <hdf5.h>

#include <filesystem>
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

/// An attribute the way R's hdf5r writes a logical: a scalar enumeration over
/// H5T_STD_U8LE with the members FALSE = 0, TRUE = 1 and NA = 2.
void write_r_logical(HighFive::Group &group, const std::string &name, unsigned char value)
{
    hid_t type = H5Tenum_create(H5T_STD_U8LE);
    unsigned char member = 0;
    H5Tenum_insert(type, "FALSE", &member);
    member = 1;
    H5Tenum_insert(type, "TRUE", &member);
    member = 2;
    H5Tenum_insert(type, "NA", &member);

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(group.getId(), name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, &value);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
}

bool throws(const ModelFile &file, const std::string &name)
{
    try
    {
        get_attribute_bool(file, "/model", name);
    }
    catch (const std::runtime_error &)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "bayests_unit_bool_attributes.h5";

    {
        HighFive::File h5(path.string(), HighFive::File::Truncate);
        HighFive::Group model = h5.createGroup("model");

        model.createAttribute("highfive_true", true);
        model.createAttribute("highfive_false", false);
        write_r_logical(model, "r_true", 1);
        write_r_logical(model, "r_false", 0);
        write_r_logical(model, "r_na", 2);
        model.createAttribute("int_one", 1);
        model.createAttribute("int_zero", 0);
    }

    {
        HighFive::File h5(path.string(), HighFive::File::ReadOnly);
        const ModelFile file(h5);

        check(get_attribute_bool(file, "/model", "highfive_true"), "a HighFive TRUE reads as true");
        check(!get_attribute_bool(file, "/model", "highfive_false"), "a HighFive FALSE reads as false");
        check(get_attribute_bool(file, "/model", "r_true"), "an R logical TRUE reads as true");
        check(!get_attribute_bool(file, "/model", "r_false"), "an R logical FALSE reads as false");
        check(throws(file, "r_na"), "an R logical NA is refused rather than read as either");
        check(get_attribute_bool(file, "/model", "int_one"), "an integer 1 reads as true");
        check(!get_attribute_bool(file, "/model", "int_zero"), "an integer 0 reads as false");
    }

    std::filesystem::remove(path);

    if (failures > 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    return 0;
}
