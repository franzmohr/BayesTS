// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef READ_HDF5_H
#define READ_HDF5_H

#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <highfive/H5File.hpp>
#include <highfive/H5Attribute.hpp>
#include "bayests/arma.h"

// Returns true if the path has an HDF5 extension (.hdf5 or .h5), case-insensitive
bool is_hdf5_file(const std::filesystem::path &filepath);

// Open the file in read-only mode and return HighFive File object
HighFive::File open_hdf5_file(const std::filesystem::path &filepath);

// Open the file in read-write mode and return HighFive File object
HighFive::File open_hdf5_file_readwrite(const std::filesystem::path &filepath);

/// The spelling of a group name this file layer works in: a leading slash, no
/// trailing slash, and "" for the root of the file. Idempotent, so a name that
/// has already been through it can be passed again.
///
/// Throws std::invalid_argument on a name that cannot be a group -- an empty
/// component, or a "." or ".." that HDF5 would resolve relative to somewhere the
/// caller did not name. That is a command line that cannot be acted on rather
/// than a run that failed, which is why it is rejected before a file is opened.
std::string normalize_hdf5_group(const std::string &group);

/// A model inside an HDF5 file: the file, plus the group the model's tree hangs
/// under.
///
/// Every path this layer names is absolute -- "/model", "/data/train/y",
/// "/posterior/a/coeffs" -- and in HDF5 a leading slash resolves from the root
/// of the file even when the call goes through a group handle. So a model that
/// does not sit at the root cannot be reached by handing the readers a
/// HighFive::Group; the group has to go on the front of the path. That is all
/// this does, and it is why every reader and writer below takes one of these
/// rather than a file.
///
/// The conversion from a bare HighFive::File is deliberately implicit: a caller
/// with no group to name -- the fixture writers, and every command line without
/// --group -- goes on naming the same paths at the root of the file as before.
///
/// This is a handle, like a pointer: it refers to a file it does not own, and
/// the file has to outlive it. The methods are const for the same reason a write
/// through a pointer-to-const-pointer is allowed -- what decides whether a write
/// is permitted is the mode the file was opened in, not the constness of this.
class ModelFile
{
public:
    ModelFile(HighFive::File &file) : file_(file) {}

    ModelFile(HighFive::File &file, const std::string &group)
        : file_(file), prefix_(normalize_hdf5_group(group))
    {
    }

    /// The path `name` names within this model: "/model" becomes
    /// "/models/3/model" for a model under /models/3, and stays "/model" for one
    /// at the root.
    std::string resolve(const std::string &name) const { return prefix_ + name; }

    /// The group the model hangs under, "" for the root of the file.
    const std::string &group() const { return prefix_; }

    /// The file itself, for the few callers that have to reach past one model --
    /// listing the models in a file, or reporting which file an error came from.
    HighFive::File &file() const { return file_; }

    // The rest is HighFive::File's own interface, restricted to the parts this
    // layer uses and with every path resolved on the way through. The names
    // match HighFive's so that a reader written against a file reads the same
    // written against a model.

    bool exist(const std::string &name) const { return file_.exist(resolve(name)); }

    HighFive::Group getGroup(const std::string &name) const
    {
        return file_.getGroup(resolve(name));
    }

    HighFive::DataSet getDataSet(const std::string &name) const
    {
        if (reads_)
        {
            reads_->insert(name);
        }
        return file_.getDataSet(resolve(name));
    }

    /// From here on, remember the name of every dataset opened through this
    /// handle -- and through every copy of it, since the readers take one by
    /// const reference and pass it on. `bayests check` compares that list with
    /// the datasets the model's tree holds: a dataset the reader never opened
    /// is usually the trace of a model other than the one the file was written
    /// for. Off unless asked for, so a run pays nothing.
    void record_reads() { reads_ = std::make_shared<std::set<std::string>>(); }

    /// The datasets opened since record_reads(), model-relative and sorted.
    std::vector<std::string> datasets_read() const
    {
        return reads_ ? std::vector<std::string>(reads_->begin(), reads_->end())
                      : std::vector<std::string>();
    }

    HighFive::Group createGroup(const std::string &name) const
    {
        return file_.createGroup(resolve(name));
    }

    void unlink(const std::string &name) const { file_.unlink(resolve(name)); }

    template <typename T>
    HighFive::DataSet createDataSet(const std::string &name,
                                    const HighFive::DataSpace &space) const
    {
        return file_.createDataSet<T>(resolve(name), space);
    }

private:
    HighFive::File &file_;
    std::string prefix_;
    std::shared_ptr<std::set<std::string>> reads_;
};

/// Throws unless `group` is a group in `file`. An empty group is the root, which
/// always is one.
///
/// Checked before a model is read so that a --group naming something that is
/// not there fails once, saying so, rather than as whichever dataset the reader
/// reached for first.
void require_group(const HighFive::File &file, const std::string &group);

/// The groups at or below `root` that hold a model, normalized and sorted,
/// `root` itself included if it is one. Empty if there is no model under it.
///
/// A model is a group with a `model` subgroup carrying an `algorithm`
/// attribute -- the same thing get_algorithm_type() reads, so a group this
/// returns is one the model factory can be asked about.
///
/// The search does not descend into a model. A model's /data, /priors and
/// /posterior are its own subtree, not a place further models could be, and
/// stopping there is what keeps this from walking every dataset in a file that
/// holds a hundred of them.
///
/// Throws, through require_group(), if `root` is not a group in the file: a
/// misspelled --group is worth failing on, and is not the same thing as a
/// well-formed root with no models under it.
std::vector<std::string> list_model_groups(const HighFive::File &file, const std::string &root);

/// Every dataset in one model's tree, model-relative ("/data/train/y") and
/// sorted. /posterior is left out, being the run's output rather than its
/// input, and so is any group below that is a model of its own -- a model at
/// the root of a file that also holds /models/3 does not own what is in it.
std::vector<std::string> list_model_datasets(const ModelFile &file);

/// The names of the attributes on `group` within a model, sorted.
std::vector<std::string> list_attribute_names(const ModelFile &file, const std::string &group);

// Get algorithm type from the model's /model group
std::string get_algorithm_type(const ModelFile &file);

// Get integer attribute from group
int get_attribute_int(const ModelFile &file, const std::string &group_name, const std::string &attr_name);

// Get double attribute from group
double get_attribute_double(const ModelFile &file, const std::string &group_name, const std::string &attr_name);

// Get string attribute from group
std::string get_attribute_string(const ModelFile &file, const std::string &group_name, const std::string &attr_name);

// Get boolean attribute from group
bool get_attribute_bool(const ModelFile &file, const std::string &group_name, const std::string &attr_name);

// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_double(const ModelFile &file, const std::string &dataset_name);

// Read dataset and transform it into an Armadillo matrix
arma::mat hdf5_dataset_to_armadillo_matrix_integer(const ModelFile &file, const std::string &dataset_name);

// Read integer value from dataset
int get_dataset_int(const ModelFile &file, const std::string &dataset_name);

/// A single number held as a dataset rather than as an attribute, which is how
/// these files carry a scalar prior. Not read through the matrix path: those
/// datasets are one-dimensional, and the Armadillo conversion wants two.
double get_dataset_double(const ModelFile &file, const std::string &dataset_name);

// Write Armadillo matrix to HDF5 dataset
void write_armadillo_matrix_to_hdf5(const ModelFile &file, const std::string &dataset_name, const arma::mat &matrix, const bool add_mcpar);

// Write a single double value to HDF5 dataset
void write_dataset_double(const ModelFile &file, const std::string &dataset_name, double value);

// Check if a dataset exists and contains data
bool dataset_has_data(const ModelFile &file, const std::string &dataset_name);

// Check if an attribute exists in a group
bool attribute_exists(const ModelFile &file, const std::string &group_name, const std::string &attr_name);

#endif // READ_HDF5_H
