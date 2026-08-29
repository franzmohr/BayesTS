// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr


#include "hdf5_and_armadillo.h"
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

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

std::string normalize_hdf5_group(const std::string &group)
{
	// The root of the file, however it was spelled. Left as the empty string
	// rather than "/" so that resolve() is a plain concatenation and a model at
	// the root names exactly the paths it named before this existed.
	if (group.empty() || group == "/")
	{
		return "";
	}

	std::string normalized = group;

	if (normalized.front() != '/')
	{
		normalized.insert(normalized.begin(), '/');
	}
	while (normalized.size() > 1 && normalized.back() == '/')
	{
		normalized.pop_back();
	}

	// Every component between the slashes, checked for the two things that would
	// make the resolved path name something other than what the caller wrote.
	std::size_t start = 1;
	while (start <= normalized.size())
	{
		const std::size_t end = normalized.find('/', start);
		const std::string component =
			normalized.substr(start, end == std::string::npos ? std::string::npos : end - start);

		if (component.empty())
		{
			throw std::invalid_argument("Group '" + group + "' has an empty component");
		}
		if (component == "." || component == "..")
		{
			throw std::invalid_argument("Group '" + group +
			                            "' contains '" + component +
			                            "', which would resolve relative to somewhere else in "
			                            "the file");
		}

		if (end == std::string::npos)
		{
			break;
		}
		start = end + 1;
	}

	return normalized;
}

void require_group(const HighFive::File &file, const std::string &group)
{
	if (group.empty())
	{
		return;
	}

	if (!file.exist(group))
	{
		throw std::runtime_error("No group '" + group + "' in this file");
	}

	if (file.getObjectType(group) != HighFive::ObjectType::Group)
	{
		throw std::runtime_error("'" + group + "' is not a group in this file");
	}
}

// Get algorithm type
std::string get_algorithm_type(const ModelFile &file)
{
	try {
		HighFive::Group model_group = file.getGroup("/model");
		HighFive::Attribute type_attr = model_group.getAttribute("algorithm");
		std::string model_type;
		type_attr.read(model_type);
		return model_type;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to detect algorithm flag at '" +
		                         file.resolve("/model") + "': " + std::string(e.what()));
	}
}

// Get integer attribute from group
int get_attribute_int(const ModelFile &file, const std::string &group_name, const std::string &attr_name)
{
	try {
		HighFive::Group group = file.getGroup(group_name);
		HighFive::Attribute attr = group.getAttribute(attr_name);

		// The class is read rather than assumed, because a writer does not
		// reliably pick the one the value deserves. R has no integer literal
		// unless it is asked for -- `p = 2` is a double, `p = 2L` an integer --
		// so a dimension written from that side arrives as H5T_IEEE_F64LE as
		// often as H5T_STD_I32LE, and the recorded VEC model files carry
		// /model/p exactly that way while every dimension beside it is an int.
		// Rejecting those files would be rejecting them over a detail of how
		// their numbers were typed, not over anything they mean.
		if (attr.getDataType().getClass() == HighFive::DataTypeClass::Float) {
			double value;
			attr.read(value);

			// These are counts, so a fractional one is a file that does not say
			// what it appears to say. Truncating would silently turn a lag order
			// of 2.5 into 2; the caller would rather hear about it.
			const double rounded = std::round(value);
			if (std::abs(value - rounded) > 1e-9) {
				throw std::runtime_error("Attribute '" + attr_name + "' of group '" +
				                         file.resolve(group_name) + "' is " + std::to_string(value) +
				                         ", which is not a whole number");
			}
			return static_cast<int>(rounded);
		}

		int value;
		attr.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read integer attribute '" + attr_name +
		                         "' from group '" + file.resolve(group_name) + "': " + std::string(e.what()));
	}
}

// Get double attribute from group
double get_attribute_double(const ModelFile &file, const std::string &group_name, const std::string &attr_name)
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
		                         "' from group '" + file.resolve(group_name) + "': " + std::string(e.what()));
	}
}

// Get double attribute from group
std::string get_attribute_string(const ModelFile &file, const std::string &group_name, const std::string &attr_name)
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
		                         "' from group '" + file.resolve(group_name) + "': " + std::string(e.what()));
	}
}

// Get double attribute from group
bool get_attribute_bool(const ModelFile &file, const std::string &group_name, const std::string &attr_name)
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
		                         "' from group '" + file.resolve(group_name) + "': " + std::string(e.what()));
	}
}

// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_double(const ModelFile &file, const std::string &dataset_name)
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
		throw std::runtime_error("Failed to read dataset '" + file.resolve(dataset_name) + "': " + std::string(e.what()));
	}
}


// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_integer(const ModelFile &file, const std::string &dataset_name)
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
		throw std::runtime_error("Failed to read dataset '" + file.resolve(dataset_name) + "': " + std::string(e.what()));
	}
}

// Read integer value from dataset
double get_dataset_double(const ModelFile &file, const std::string &dataset_name)
{
	try {
		HighFive::DataSet dataset = file.getDataSet(dataset_name);
		double value;
		dataset.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read double from dataset '" + file.resolve(dataset_name) + "': " + std::string(e.what()));
	}
}

int get_dataset_int(const ModelFile &file, const std::string &dataset_name)
{
	try {
		HighFive::DataSet dataset = file.getDataSet(dataset_name);

		// The class is read rather than assumed, for the same reason as in
		// get_attribute_int(): a whole number written from R is a double unless
		// the writer asked for an integer, so /priors/u_sigma/df arrives as
		// H5T_IEEE_F64LE in the recorded model files. HDF5 converts it happily,
		// but HighFive logs a warning on every such read, and a warning that
		// fires on correct input is one a reader learns to ignore.
		if (dataset.getDataType().getClass() == HighFive::DataTypeClass::Float) {
			double value;
			dataset.read(value);

			// A degrees-of-freedom or dimension that is not whole is a file that
			// does not say what it appears to; truncating would hide it.
			const double rounded = std::round(value);
			if (std::abs(value - rounded) > 1e-9) {
				throw std::runtime_error("Dataset '" + file.resolve(dataset_name) + "' is " +
				                         std::to_string(value) +
				                         ", which is not a whole number");
			}
			return static_cast<int>(rounded);
		}

		int value;
		dataset.read(value);
		return value;
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to read integer from dataset '" + file.resolve(dataset_name) + "': " + std::string(e.what()));
	}
}

// Write Armadillo matrix to HDF5 dataset
void write_armadillo_matrix_to_hdf5(const ModelFile &file, const std::string &dataset_name, const arma::mat &matrix, const bool add_mcpar)
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
		throw std::runtime_error("Failed to write dataset '" + file.resolve(dataset_name) + "': " + std::string(e.what()));
	}
}

// Write a single double value to HDF5 dataset
void write_dataset_double(const ModelFile &file, const std::string &dataset_name, double value)
{
	try {
		// Create a scalar dataspace
		HighFive::DataSpace dataspace = HighFive::DataSpace::From(value);

		// Create dataset and write the value
		HighFive::DataSet dataset = file.createDataSet<double>(dataset_name, dataspace);
		dataset.write(value);
	}
	catch (const HighFive::Exception &e) {
		throw std::runtime_error("Failed to write value to dataset '" + file.resolve(dataset_name) + "': " + std::string(e.what()));
	}
}

// Check if a dataset exists and contains data
bool dataset_has_data(const ModelFile &file, const std::string &dataset_name)
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
bool attribute_exists(const ModelFile &file, const std::string &group_name, const std::string &attr_name)
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
