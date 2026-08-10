// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// Derives a model file that exercises variable selection and forecasting from
// one that does not.
//
// Every checked-in fixture has varsel="none" and no forecast data, which
// leaves three branches of VarNormalWishart with no coverage at all -- and one
// of them, the forecast, indexes the posterior draws. This tool bolts the
// missing priors and regressors onto a copy of an existing fixture so the same
// before/after comparison can be run over those branches too.
//
//   make_varsel_fixture <source.h5> <dest.h5> <none|ssvs|bvs> <horizon>
//
// The values it invents are not meant to be a realistic prior. They only have
// to be admissible, and identical between the two builds being compared.

#include "io/hdf5/hdf5_and_armadillo.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

void replace_string_attribute(HighFive::File &file, const std::string &group_name,
                              const std::string &name, const std::string &value)
{
    HighFive::Group group = file.getGroup(group_name);
    if (group.hasAttribute(name))
    {
        group.deleteAttribute(name);
    }
    group.createAttribute<std::string>(name, HighFive::DataSpace::From(value)).write(value);
}

void replace_int_attribute(HighFive::File &file, const std::string &group_name,
                           const std::string &name, int value)
{
    HighFive::Group group = file.getGroup(group_name);
    if (group.hasAttribute(name))
    {
        group.deleteAttribute(name);
    }
    group.createAttribute<int>(name, HighFive::DataSpace::From(value)).write(value);
}

/// Mirrors write_armadillo_matrix_to_hdf5 for the one dataset the reader pulls
/// in as integers.
void write_int_row(HighFive::File &file, const std::string &dataset,
                   const std::vector<int> &values)
{
    if (file.exist(dataset))
    {
        file.unlink(dataset);
    }
    const std::vector<std::vector<int>> data{values};
    file.createDataSet<int>(dataset, HighFive::DataSpace::From(data)).write(data);
}

void write_row(HighFive::File &file, const std::string &dataset, const arma::vec &values)
{
    write_armadillo_matrix_to_hdf5(file, dataset, arma::mat(values.t()), false);
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <source.h5> <dest.h5> <none|ssvs|bvs> <horizon>\n";
        return 2;
    }

    const std::filesystem::path source = argv[1];
    const std::filesystem::path dest = argv[2];
    const std::string varsel = argv[3];
    const int h = std::stoi(argv[4]);

    try
    {
        std::filesystem::create_directories(dest.parent_path());
        std::filesystem::copy_file(source, dest,
                                   std::filesystem::copy_options::overwrite_existing);

        HighFive::File file = open_hdf5_file_readwrite(dest);

        const int k = get_attribute_int(file, "/model", "k");
        const int p = get_attribute_int(file, "/model", "p");
        const arma::mat z = hdf5_dataset_to_armadillo_matrix_double(file, "/data/train/z");
        const arma::uword nparams = z.n_cols;

        replace_string_attribute(file, "/model", "varsel", varsel);

        if (varsel != "none")
        {
            // Select every coefficient, so the loop over positions runs its
            // full length and the shuffle has something to permute.
            std::vector<int> include(nparams);
            for (arma::uword i = 0; i < nparams; ++i)
            {
                include[i] = static_cast<int>(i) + 1; // stored one-based
            }
            write_int_row(file, "/priors/a/include", include);

            write_row(file, "/priors/a/inprior", arma::vec(nparams, arma::fill::value(0.5)));
            write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));

            if (varsel == "ssvs")
            {
                // Spike well inside the slab, so both mixture components are
                // reachable and the inclusion draws actually flip.
                write_row(file, "/priors/a/tau0", arma::vec(nparams, arma::fill::value(0.1)));
                write_row(file, "/priors/a/tau1", arma::vec(nparams, arma::fill::value(10.0)));
            }
        }

        if (h > 0)
        {
            const arma::uword lag_cols = static_cast<arma::uword>(p) * k * k;
            if (lag_cols + k != nparams)
            {
                std::cerr << "Cannot build forecast regressors for this fixture: " << nparams
                          << " coefficients do not decompose into " << lag_cols
                          << " lag columns plus " << k << " deterministic columns.\n";
                return 1;
            }

            const arma::vec y = arma::vectorise(arma::trans(
                hdf5_dataset_to_armadillo_matrix_double(file, "/data/train/y")));
            const arma::mat diag_k = arma::eye<arma::mat>(k, k);

            arma::mat z_fcst = arma::zeros<arma::mat>(h * k, nparams);
            for (int i = 0; i < h; ++i)
            {
                // Intercept block. The lag block for i > 0 is overwritten by
                // the sampler as the path unfolds, so only i = 0 is seeded --
                // with the last observation of the sample.
                z_fcst.submat(i * k, lag_cols, (i + 1) * k - 1, nparams - 1) = diag_k;
            }
            if (lag_cols > 0)
            {
                const arma::vec last = y.tail(k);
                z_fcst.submat(0, 0, k - 1, k * k - 1) = arma::kron(arma::trans(last), diag_k);
            }

            if (!file.exist("/data/forecast"))
            {
                file.createGroup("/data/forecast");
            }
            write_armadillo_matrix_to_hdf5(file, "/data/forecast/z", z_fcst, false);
            replace_int_attribute(file, "/model", "h", h);
        }

        std::cout << "wrote " << dest.string() << " (varsel=" << varsel << ", h=" << h
                  << ", k=" << k << ", p=" << p << ", nparams=" << nparams << ")\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
