// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr


#include "hdf5_and_armadillo.h"
#include <vector>
#include <algorithm>
#include <cctype>

// Returns true if the path has an HDF5 extension (.hdf5 or .h5), case-insensitive
bool is_hdf5_file(const std::filesystem::path &filepath)
{
	std::string ext = filepath.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(),
				   [](unsigned char c) { return std::tolower(c); });
	return ext == ".hdf5" || ext == ".h5";
}

// Open the file in read-only mode and return HighFive File object
HighFive::File open_hdf5_file(const std::filesystem::path &filepath)
{
	try {
		return HighFive::File(filepath.string(), HighFive::File::ReadOnly);
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to open HDF5 file: " + std::string(e.what()));
	}
}

// Open the file in read-write mode and return HighFive File object
HighFive::File open_hdf5_file_readwrite(const std::filesystem::path &filepath)
{
	try {
		return HighFive::File(filepath.string(), HighFive::File::ReadWrite);
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to open HDF5 file for writing: " + std::string(e.what()));
	}
}

// Get algorithm type
std::string get_algorithm_type(const HighFive::File &file)
{
	try {
		HighFive::Group model_group = file.getGroup("/model");
		HighFive::Attribute type_attr = model_group.getAttribute("algorithm");
		std::string model_type;
		type_attr.read(model_type);
		return model_type;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to detect algorithm flag: " + std::string(e.what()));
	}
}

// Get integer attribute from group
int get_attribute_int(const HighFive::File &file, const std::string &group_name, const std::string &attr_name)
{
	try {
		HighFive::Group group = file.getGroup(group_name);
		HighFive::Attribute attr = group.getAttribute(attr_name);
		int value;
		attr.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read integer attribute '" + attr_name +
		                         "' from group '" + group_name + "': " + std::string(e.what()));
	}
}

// Get double attribute from group
double get_attribute_double(const HighFive::File &file, const std::string &group_name, const std::string &attr_name)
{
	try {
		HighFive::Group group = file.getGroup(group_name);
		HighFive::Attribute attr = group.getAttribute(attr_name);
		double value;
		attr.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read double attribute '" + attr_name +
		                         "' from group '" + group_name + "': " + std::string(e.what()));
	}
}

// Get double attribute from group
std::string get_attribute_string(const HighFive::File &file, const std::string &group_name, const std::string &attr_name)
{
	try {
		HighFive::Group group = file.getGroup(group_name);
		HighFive::Attribute attr = group.getAttribute(attr_name);
		std::string value;
		attr.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read string attribute '" + attr_name +
		                         "' from group '" + group_name + "': " + std::string(e.what()));
	}
}

// Get double attribute from group
bool get_attribute_bool(const HighFive::File &file, const std::string &group_name, const std::string &attr_name)
{
	try {
		HighFive::Group group = file.getGroup(group_name);
		HighFive::Attribute attr = group.getAttribute(attr_name);
		bool value;
		attr.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read boolean attribute '" + attr_name +
		                         "' from group '" + group_name + "': " + std::string(e.what()));
	}
}

// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_double(const HighFive::File &file, const std::string &dataset_name)
{
	try {
		// Open dataset
		HighFive::DataSet dataset = file.getDataSet(dataset_name);

		// Get dimensions
		std::vector<size_t> dims = dataset.getDimensions();

		if (dims.size() != 2) {
			throw std::runtime_error("Dataset must be 2D for conversion to Armadillo matrix");
		}

		// Read the data into a 2D vector (row-major order from HDF5)
		std::vector<std::vector<double>> data;
		dataset.read(data);

		// Convert to Armadillo matrix (column-major)
		// HDF5 stores in row-major, so we need to transpose
		arma::mat result(dims[0], dims[1]);
		for (size_t i = 0; i < dims[0]; i++) {
			for (size_t j = 0; j < dims[1]; j++) {
				result(i, j) = data[i][j];
			}
		}

		// Transpose to match expected layout
		return arma::trans(result);
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read dataset '" + dataset_name + "': " + std::string(e.what()));
	}
}


// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_integer(const HighFive::File &file, const std::string &dataset_name)
{
	try {
		// Open dataset
		HighFive::DataSet dataset = file.getDataSet(dataset_name);

		// Get dimensions
		std::vector<size_t> dims = dataset.getDimensions();

		if (dims.size() != 2) {
			throw std::runtime_error("Dataset must be 2D for conversion to Armadillo matrix");
		}

		// Read the data into a 2D vector (row-major order from HDF5)
		std::vector<std::vector<int>> data;
		dataset.read(data);

		// Convert to Armadillo matrix (column-major)
		// HDF5 stores in row-major, so we need to transpose
		arma::mat result(dims[0], dims[1]);
		for (size_t i = 0; i < dims[0]; i++) {
			for (size_t j = 0; j < dims[1]; j++) {
				result(i, j) = data[i][j];
			}
		}

		// Transpose to match expected layout
		return arma::trans(result);
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read dataset '" + dataset_name + "': " + std::string(e.what()));
	}
}

// Read integer value from dataset
int get_dataset_int(const HighFive::File &file, const std::string &dataset_name)
{
	try {
		HighFive::DataSet dataset = file.getDataSet(dataset_name);
		int value;
		dataset.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read integer from dataset '" + dataset_name + "': " + std::string(e.what()));
	}
}

// Write Armadillo matrix to HDF5 dataset
void write_armadillo_matrix_to_hdf5(HighFive::File &file, const std::string &dataset_name, const arma::mat &matrix, const bool add_mcpar)
{
	try {
		// Transpose matrix back (Armadillo is column-major, HDF5 expects row-major)
		arma::mat transposed = arma::trans(matrix);

		// Get dimensions
		size_t rows = transposed.n_rows;
		size_t cols = transposed.n_cols;

		// Convert Armadillo matrix to 2D vector for HighFive
		std::vector<std::vector<double>> data(rows, std::vector<double>(cols));
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				data[i][j] = transposed(i, j);
			}
		}

		// If a dataset with this name already exists, remove it so it can be overwritten
		// (unlink also covers the case where the new data has different dimensions)
		if (file.exist(dataset_name)) {
			file.unlink(dataset_name);
		}

		// Create dataset and write data
		HighFive::DataSet dataset = file.createDataSet<double>(dataset_name, HighFive::DataSpace::From(data));
		dataset.write(data);

		if (add_mcpar) {
			dataset.createAttribute("start", 1);
			dataset.createAttribute("end", cols);
			dataset.createAttribute("thin", 1);
		}
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to write dataset '" + dataset_name + "': " + std::string(e.what()));
	}
}

// Write a single double value to HDF5 dataset
void write_dataset_double(HighFive::File &file, const std::string &dataset_name, double value)
{
	try {
		// Create a scalar dataspace
		HighFive::DataSpace dataspace = HighFive::DataSpace::From(value);

		// Create dataset and write the value
		HighFive::DataSet dataset = file.createDataSet<double>(dataset_name, dataspace);
		dataset.write(value);
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to write value to dataset '" + dataset_name + "': " + std::string(e.what()));
	}
}

// Check if a dataset exists and contains data
bool dataset_has_data(const HighFive::File &file, const std::string &dataset_name)
{
	try {
		if (!file.exist(dataset_name)) {
			return false;
		}

		HighFive::DataSet dataset = file.getDataSet(dataset_name);
		std::vector<size_t> dims = dataset.getDimensions();

		// Check if dataset has data (non-zero dimensions)
		if (dims.size() > 0 && dims[0] > 0) {
			return true;
		}

		return false;
	}
	catch (const HighFive::Exception &e) {
		// If there's any error accessing the dataset, assume it doesn't have data
		return false;
	}
}

// Check if an attribute exists in a group
bool attribute_exists(const HighFive::File &file, const std::string &group_name, const std::string &attr_name)
{
	try {
		if (!file.exist(group_name)) {
			return false;
		}

		HighFive::Group group = file.getGroup(group_name);
		return group.hasAttribute(attr_name);
	}
	catch (const HighFive::Exception &e) {
		// If there's any error accessing the group or attribute, assume it doesn't exist
		return false;
	}
}
