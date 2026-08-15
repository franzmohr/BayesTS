// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef READ_HDF5_H
#define READ_HDF5_H

#include <filesystem>
#include <string>
#include <highfive/H5File.hpp>
#include <highfive/H5Attribute.hpp>
#include "bayests/arma.h"

// Returns true if the path has an HDF5 extension (.hdf5 or .h5), case-insensitive
bool is_hdf5_file(const std::filesystem::path &filepath);

// Open the file in read-only mode and return HighFive File object
HighFive::File open_hdf5_file(const std::filesystem::path &filepath);

// Open the file in read-write mode and return HighFive File object
HighFive::File open_hdf5_file_readwrite(const std::filesystem::path &filepath);

// Get algorithm type from /model group
std::string get_algorithm_type(const HighFive::File &file);

// Get integer attribute from group
int get_attribute_int(const HighFive::File &file, const std::string &group_name, const std::string &attr_name);

// Get double attribute from group
double get_attribute_double(const HighFive::File &file, const std::string &group_name, const std::string &attr_name);

// Get string attribute from group
std::string get_attribute_string(const HighFive::File &file, const std::string &group_name, const std::string &attr_name);

// Get boolean attribute from group
bool get_attribute_bool(const HighFive::File &file, const std::string &group_name, const std::string &attr_name);

// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_double(const HighFive::File &file, const std::string &dataset_name);

// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_integer(const HighFive::File &file, const std::string &dataset_name);

// Read integer value from dataset
int get_dataset_int(const HighFive::File &file, const std::string &dataset_name);

/// A single number held as a dataset rather than as an attribute, which is how
/// these files carry a scalar prior. Not read through the matrix path: those
/// datasets are one-dimensional, and the Armadillo conversion wants two.
double get_dataset_double(const HighFive::File &file, const std::string &dataset_name);

// Write Armadillo matrix to HDF5 dataset
void write_armadillo_matrix_to_hdf5(HighFive::File &file, const std::string &dataset_name, const arma::mat &matrix, const bool add_mcpar);

// Write a single double value to HDF5 dataset
void write_dataset_double(HighFive::File &file, const std::string &dataset_name, double value);

// Check if a dataset exists and contains data
bool dataset_has_data(const HighFive::File &file, const std::string &dataset_name);

// Check if an attribute exists in a group
bool attribute_exists(const HighFive::File &file, const std::string &group_name, const std::string &attr_name);

#endif // READ_HDF5_H
