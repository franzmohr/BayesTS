// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// Builds a complete model file for the samplers that have no checked-in
// fixtures.
//
// VarNormalWishart is the only model with recorded inputs in test/, so the
// other three could not be put through the same before/after comparison as a
// refactor. make_varsel_fixture cannot fill the gap: it decorates an existing
// file, and there is nothing to decorate. This tool writes one from scratch.
//
//   make_model_fixture <dest.h5> <model> <varsel> <covar> <structural> <h>
//
//     model       VarNormalGamma | VarNormalStochvol | VarTvpGamma
//                 | VarTvpWishart | VarTvpStochvol
//     varsel      none | ssvs | bvs        (ssvs only reaches VarNormalGamma)
//     covar       0 | 1                    the "+covar" error specification
//     structural  0 | 1
//     h           forecast horizon; 0 writes no forecast regressors
//
// The numbers it invents are not a realistic prior or a realistic sample. They
// only have to be admissible and reproducible: the file is an input to a
// fingerprint comparison between two builds, so what matters is that
// regenerating it produces the same bytes. Nothing here uses Armadillo's RNG --
// that one belongs to the sampler under test, and drawing from it here would
// couple the fixture to the very stream the comparison is pinning.

#include "io/hdf5/hdf5_and_armadillo.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

// A model small enough to run in a second and large enough that every loop in
// the samplers goes round more than once.
constexpr int kK = 3;  // endogenous variables
constexpr int kP = 1;  // lags
constexpr int kM = 0;  // exogenous variables
constexpr int kS = 0;  // exogenous lags
constexpr int kN = 1;  // deterministic terms (an intercept)
constexpr int kTT = 24; // periods

// The three samplers divide by draws/100 to decide when to repaint their
// progress bar, so a chain shorter than 100 draws divides by zero. Kept above
// the threshold to stay on the behaviour the fixtures are meant to record.
constexpr int kIterations = 80;
constexpr int kBurnin = 40;

/// A 64-bit LCG, so the fixtures do not depend on the host's <random>
/// implementation the way std::mt19937 plus a distribution would.
class Lcg
{
public:
    explicit Lcg(unsigned long long seed) : state_(seed) {}

    /// Uniform on [0, 1).
    double uniform()
    {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const unsigned long long mantissa = (state_ >> 11) & ((1ULL << 53) - 1);
        return static_cast<double>(mantissa) / 9007199254740992.0; // 2^53
    }

    /// Roughly standard normal. Twelve uniforms minus six: not a good sampler,
    /// but this only has to produce a plausible-looking series.
    double normal()
    {
        double sum = 0.0;
        for (int i = 0; i < 12; ++i)
        {
            sum += uniform();
        }
        return sum - 6.0;
    }

private:
    unsigned long long state_;
};

void write_mat(HighFive::File &file, const std::string &dataset, const arma::mat &m)
{
    write_armadillo_matrix_to_hdf5(file, dataset, m, false);
}

/// Vectors live on disk as a single HDF5 row, which is how the R writer stores
/// them. Written as an Armadillo column, because the writer transposes on the
/// way out -- and the orientation matters: the readers that skip vectorise()
/// and assign the result straight to an arma::vec only accept a column back.
void write_row(HighFive::File &file, const std::string &dataset, const arma::vec &v)
{
    write_armadillo_matrix_to_hdf5(file, dataset, arma::mat(v), false);
}

/// A one-element integer dataset read back with get_dataset_int(), which reads
/// a scalar rather than a row.
void write_int_scalar(HighFive::File &file, const std::string &dataset, int value)
{
    if (file.exist(dataset))
    {
        file.unlink(dataset);
    }
    file.createDataSet<int>(dataset, HighFive::DataSpace::From(value)).write(value);
}

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

void ensure_group(HighFive::File &file, const std::string &group)
{
    if (!file.exist(group))
    {
        file.createGroup(group);
    }
}

template <typename T>
void write_attribute(HighFive::File &file, const std::string &group_name,
                     const std::string &name, const T &value)
{
    ensure_group(file, group_name);
    HighFive::Group group = file.getGroup(group_name);
    if (group.hasAttribute(name))
    {
        group.deleteAttribute(name);
    }
    group.createAttribute<T>(name, HighFive::DataSpace::From(value)).write(value);
}

/// Positions are stored one-based, matching R and the rest of the format.
void write_selection(HighFive::File &file, const std::string &prior_group, arma::uword n)
{
    std::vector<int> include(n);
    for (arma::uword i = 0; i < n; ++i)
    {
        include[i] = static_cast<int>(i) + 1;
    }
    write_int_row(file, prior_group + "/include", include);
    write_row(file, prior_group + "/inprior", arma::vec(n, arma::fill::value(0.5)));
}

/// Spike well inside the slab, so both mixture components stay reachable and
/// the inclusion draws actually flip.
void write_ssvs(HighFive::File &file, const std::string &prior_group, arma::uword n)
{
    write_row(file, prior_group + "/tau0", arma::vec(n, arma::fill::value(0.1)));
    write_row(file, prior_group + "/tau1", arma::vec(n, arma::fill::value(10.0)));
}

struct Layout
{
    int n_lag = kK * kK * kP;      // 9
    int n_non_structural = kK * (kK * kP + kM * (kS + 1) + kN); // 12, lags + intercepts
    int n_structural = 0;
    int nparams = 0;
    int n_psi = kK * (kK - 1) / 2; // 3
};

/// A stable VAR(1) with an intercept, kP periods longer than the sample so the
/// first row of regressors has a lag to point at.
arma::mat simulate_series(Lcg &rng)
{
    arma::mat series(kK, kTT + kP, arma::fill::zeros);
    for (int t = 1; t < kTT + kP; ++t)
    {
        for (int i = 0; i < kK; ++i)
        {
            series(i, t) = 0.5 * series(i, t - 1) + rng.normal();
        }
    }
    return series;
}

/// (tt * k) x nparams, laid out to multiply the stacked response directly:
/// per period a kron(y'_{t-1}, I_k) lag block, an identity intercept block and,
/// for a structural model, k(k-1)/2 contemporaneous columns.
arma::mat build_train_regressors(const arma::mat &series, const Layout &layout, Lcg &rng)
{
    const arma::mat diag_k = arma::eye<arma::mat>(kK, kK);
    arma::mat z(kTT * kK, layout.nparams, arma::fill::zeros);

    for (int t = 0; t < kTT; ++t)
    {
        const arma::vec lag = series.col(t); // series is kP ahead of the sample
        z.submat(t * kK, 0, (t + 1) * kK - 1, layout.n_lag - 1) =
            arma::kron(arma::trans(lag), diag_k);
        z.submat(t * kK, layout.n_lag, (t + 1) * kK - 1, layout.n_non_structural - 1) = diag_k;

        for (int j = 0; j < layout.n_structural; ++j)
        {
            for (int i = 0; i < kK; ++i)
            {
                z(t * kK + i, layout.n_non_structural + j) = 0.2 * rng.normal();
            }
        }
    }
    return z;
}

/// (h * k) x n_non_structural. Only the first lag block and the intercepts are
/// filled in; the sampler overwrites the rest as the path unfolds. The
/// structural columns are deliberately absent -- the forecast code splits them
/// off the posterior draws instead.
arma::mat build_forecast_regressors(const arma::mat &series, const Layout &layout, int h)
{
    const arma::mat diag_k = arma::eye<arma::mat>(kK, kK);
    arma::mat z(h * kK, layout.n_non_structural, arma::fill::zeros);

    for (int i = 0; i < h; ++i)
    {
        z.submat(i * kK, layout.n_lag, (i + 1) * kK - 1, layout.n_non_structural - 1) = diag_k;
    }
    z.submat(0, 0, kK - 1, layout.n_lag - 1) =
        arma::kron(arma::trans(series.col(kTT + kP - 1)), diag_k);

    return z;
}

void write_common(HighFive::File &file, const std::string &model, const std::string &varsel,
                  bool covar, bool structural, int h, const Layout &layout,
                  const arma::mat &series, const arma::mat &z_train)
{
    // The CLI dispatches on /model/algorithm; the checked-in fixtures predate
    // it, which is why the golden harness has to tolerate its absence.
    write_attribute<std::string>(file, "/model", "algorithm", model);
    write_attribute<int>(file, "/model", "k", kK);
    write_attribute<int>(file, "/model", "p", kP);
    write_attribute<int>(file, "/model", "m", kM);
    write_attribute<int>(file, "/model", "s", kS);
    write_attribute<int>(file, "/model", "n", kN);
    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", varsel);
    write_attribute<bool>(file, "/model", "structural", structural);

    // The error specification the reader dispatches on. VarTvpWishart carries
    // no psi block, so it has no "+covar" spelling to reach.
    std::string prefix = "gamma";
    if (model == "VarNormalStochvol" || model == "VarTvpStochvol")
    {
        prefix = "sv";
    }
    else if (model == "VarTvpWishart")
    {
        prefix = "wishart";
    }
    write_attribute<std::string>(file, "/model", "error",
                                 (covar && prefix != "wishart") ? prefix + "+covar" : prefix);

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/initial");

    // The response is stored already stacked, one row, the way the R writer
    // emits it: vec(y') with the sample periods in order.
    const arma::mat sample = series.cols(kP, kTT + kP - 1);
    write_row(file, "/data/train/y", arma::vectorise(sample));
    write_mat(file, "/data/train/z", z_train);

    if (h > 0)
    {
        ensure_group(file, "/data/forecast");
        write_mat(file, "/data/forecast/z", build_forecast_regressors(series, layout, h));
        write_attribute<int>(file, "/model", "h", h);
    }
}

void write_var_normal_gamma(HighFive::File &file, const std::string &varsel, bool covar,
                            const Layout &layout)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);
    const arma::uword n_psi = static_cast<arma::uword>(layout.n_psi);

    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));
    write_row(file, "/initial/a", arma::vec(nparams, arma::fill::zeros));

    write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(2.0)));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));

    if (varsel != "none")
    {
        write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));
        write_selection(file, "/priors/a", nparams);
        if (varsel == "ssvs")
        {
            write_ssvs(file, "/priors/a", nparams);
        }
    }

    if (covar)
    {
        ensure_group(file, "/priors/psi");
        write_row(file, "/priors/psi/mu", arma::vec(n_psi, arma::fill::zeros));
        write_mat(file, "/priors/psi/v_inv", arma::eye<arma::mat>(n_psi, n_psi));
        write_row(file, "/initial/psi", arma::vec(n_psi, arma::fill::zeros));

        if (varsel != "none")
        {
            write_row(file, "/initial/psi_lambda", arma::vec(n_psi, arma::fill::ones));
            write_selection(file, "/priors/psi", n_psi);
            if (varsel == "ssvs")
            {
                write_ssvs(file, "/priors/psi", n_psi);
            }
        }
    }
}

void write_var_normal_stochvol(HighFive::File &file, const std::string &varsel, bool covar,
                               const Layout &layout)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);
    const arma::uword n_psi = static_cast<arma::uword>(layout.n_psi);

    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));
    write_row(file, "/initial/a", arma::vec(nparams, arma::fill::zeros));

    // The offset keeps log(u^2 + offset) finite when a residual lands on zero,
    // and bounds it well inside the range the ten-component mixture covers.
    write_row(file, "/priors/u_sigma/offset", arma::vec(kK, arma::fill::value(1e-4)));
    write_row(file, "/priors/u_sigma/sigma", arma::vec(kK, arma::fill::value(0.1)));
    write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(0.2)));
    write_row(file, "/priors/u_sigma/mu", arma::vec(kK, arma::fill::zeros));
    write_mat(file, "/priors/u_sigma/v_inv", arma::eye<arma::mat>(kK, kK));

    write_mat(file, "/initial/h", arma::mat(kTT, kK, arma::fill::zeros));
    write_row(file, "/initial/h_init", arma::vec(kK, arma::fill::zeros));

    if (varsel != "none")
    {
        write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));
        write_selection(file, "/priors/a", nparams);
    }

    if (covar)
    {
        ensure_group(file, "/priors/psi");
        write_row(file, "/priors/psi/mu", arma::vec(n_psi, arma::fill::zeros));
        write_mat(file, "/priors/psi/v_inv", arma::eye<arma::mat>(n_psi, n_psi));
        write_row(file, "/initial/psi", arma::vec(n_psi, arma::fill::zeros));

        if (varsel != "none")
        {
            write_row(file, "/initial/psi_lambda", arma::vec(n_psi, arma::fill::ones));
            write_selection(file, "/priors/psi", n_psi);
        }
    }
}

void write_var_tvp_gamma(HighFive::File &file, const std::string &varsel, bool covar,
                         const Layout &layout)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);
    const arma::uword n_psi = static_cast<arma::uword>(layout.n_psi);

    // The state innovation variances are stored inverted; the sampler flips the
    // diagonal back on the way in. A tight prior keeps the random walk from
    // wandering far enough to make the Kalman filter ill-conditioned.
    write_mat(file, "/initial/a", arma::mat(nparams, kTT, arma::fill::zeros));
    write_mat(file, "/initial/a_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(nparams, arma::fill::value(100.0)))));
    write_row(file, "/initial/a_init", arma::vec(nparams, arma::fill::zeros));

    write_row(file, "/priors/a/shape", arma::vec(nparams, arma::fill::value(3.0)));
    write_row(file, "/priors/a/rate", arma::vec(nparams, arma::fill::value(0.01)));
    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));

    write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(2.0)));
    write_mat(file, "/initial/u_omega_inv", arma::eye<arma::mat>(kK, kK));

    if (varsel != "none")
    {
        write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));
        write_selection(file, "/priors/a", nparams);
    }

    if (covar)
    {
        ensure_group(file, "/priors/psi");
        write_mat(file, "/initial/psi", arma::mat(n_psi, kTT, arma::fill::zeros));
        write_mat(file, "/initial/psi_sigma_inv",
                  arma::mat(arma::diagmat(arma::vec(n_psi, arma::fill::value(100.0)))));
        write_row(file, "/initial/psi_init", arma::vec(n_psi, arma::fill::zeros));

        write_row(file, "/priors/psi/shape", arma::vec(n_psi, arma::fill::value(3.0)));
        write_row(file, "/priors/psi/rate", arma::vec(n_psi, arma::fill::value(0.01)));
        write_row(file, "/priors/psi/mu", arma::vec(n_psi, arma::fill::zeros));
        write_mat(file, "/priors/psi/v_inv", arma::eye<arma::mat>(n_psi, n_psi));

        // This model reads the psi selection scheme from its own group rather
        // than from /model, so the two can differ.
        ensure_group(file, "/model/priors");
        write_attribute<std::string>(file, "/model/priors/psi", "varsel", varsel);

        if (varsel != "none")
        {
            write_row(file, "/initial/psi_lambda", arma::vec(n_psi, arma::fill::ones));
            write_selection(file, "/priors/psi", n_psi);
        }
    }
}

/// The coefficient block every time-varying parameter model shares: a path, the
/// precision of its innovations and the state it starts from.
void write_tvp_coefficients(HighFive::File &file, const std::string &varsel, arma::uword nparams)
{
    // The state innovation variances are stored inverted; the sampler flips the
    // diagonal back on the way in. A tight prior keeps the random walk from
    // wandering far enough to make the Kalman filter ill-conditioned.
    write_mat(file, "/initial/a", arma::mat(nparams, kTT, arma::fill::zeros));
    write_mat(file, "/initial/a_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(nparams, arma::fill::value(100.0)))));
    write_row(file, "/initial/a_init", arma::vec(nparams, arma::fill::zeros));

    write_row(file, "/priors/a/shape", arma::vec(nparams, arma::fill::value(3.0)));
    write_row(file, "/priors/a/rate", arma::vec(nparams, arma::fill::value(0.01)));
    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));

    if (varsel != "none")
    {
        write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));
        write_selection(file, "/priors/a", nparams);
    }
}

void write_var_tvp_wishart(HighFive::File &file, const std::string &varsel, const Layout &layout)
{
    write_tvp_coefficients(file, varsel, static_cast<arma::uword>(layout.nparams));

    // The error covariance is the Wishart precision alone: no psi block, so
    // nothing here depends on the covar flag.
    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));
}

void write_var_tvp_stochvol(HighFive::File &file, const std::string &varsel, bool covar,
                            const Layout &layout)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);
    const arma::uword n_psi = static_cast<arma::uword>(layout.n_psi);

    write_tvp_coefficients(file, varsel, nparams);

    // The offset keeps log(u^2 + offset) finite when a residual lands on zero,
    // and bounds it well inside the range the ten-component mixture covers.
    write_row(file, "/priors/u_sigma/offset", arma::vec(kK, arma::fill::value(1e-4)));
    write_row(file, "/priors/u_sigma/sigma", arma::vec(kK, arma::fill::value(0.1)));
    write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(0.2)));
    write_row(file, "/priors/u_sigma/mu", arma::vec(kK, arma::fill::zeros));
    write_mat(file, "/priors/u_sigma/v_inv", arma::eye<arma::mat>(kK, kK));

    write_mat(file, "/initial/h", arma::mat(kTT, kK, arma::fill::zeros));
    write_row(file, "/initial/h_init", arma::vec(kK, arma::fill::zeros));

    if (covar)
    {
        ensure_group(file, "/priors/psi");
        write_mat(file, "/initial/psi", arma::mat(n_psi, kTT, arma::fill::zeros));
        write_mat(file, "/initial/psi_sigma_inv",
                  arma::mat(arma::diagmat(arma::vec(n_psi, arma::fill::value(100.0)))));
        write_row(file, "/initial/psi_init", arma::vec(n_psi, arma::fill::zeros));

        write_row(file, "/priors/psi/shape", arma::vec(n_psi, arma::fill::value(3.0)));
        write_row(file, "/priors/psi/rate", arma::vec(n_psi, arma::fill::value(0.01)));
        write_row(file, "/priors/psi/mu", arma::vec(n_psi, arma::fill::zeros));
        write_mat(file, "/priors/psi/v_inv", arma::eye<arma::mat>(n_psi, n_psi));

        // This model reads the psi selection scheme from its own group rather
        // than from /model, so the two can differ.
        ensure_group(file, "/model/priors");
        write_attribute<std::string>(file, "/model/priors/psi", "varsel", varsel);

        if (varsel != "none")
        {
            write_row(file, "/initial/psi_lambda", arma::vec(n_psi, arma::fill::ones));
            write_selection(file, "/priors/psi", n_psi);
        }
    }
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <dest.h5> <model> <none|ssvs|bvs> <covar 0|1> <structural 0|1> <h>\n";
        return 2;
    }

    const std::filesystem::path dest = argv[1];
    const std::string model = argv[2];
    const std::string varsel = argv[3];
    const bool covar = std::string(argv[4]) != "0";
    const bool structural = std::string(argv[5]) != "0";
    const int h = std::stoi(argv[6]);

    if (model != "VarNormalGamma" && model != "VarNormalStochvol" && model != "VarTvpGamma" &&
        model != "VarTvpWishart" && model != "VarTvpStochvol")
    {
        std::cerr << "Unknown model: " << model << '\n';
        return 2;
    }
    if (varsel != "none" && varsel != "ssvs" && varsel != "bvs")
    {
        std::cerr << "Unknown variable selection scheme: " << varsel << '\n';
        return 2;
    }

    try
    {
        if (!dest.parent_path().empty())
        {
            std::filesystem::create_directories(dest.parent_path());
        }
        std::filesystem::remove(dest);

        HighFive::File file(dest.string(), HighFive::File::Create);

        Layout layout;
        layout.n_structural = structural ? kK * (kK - 1) / 2 : 0;
        layout.nparams = layout.n_non_structural + layout.n_structural;

        // One generator for the whole file, so every dataset that draws from it
        // is reproducible as a set rather than individually.
        Lcg rng(20260808ULL);
        const arma::mat series = simulate_series(rng);
        const arma::mat z_train = build_train_regressors(series, layout, rng);

        write_common(file, model, varsel, covar, structural, h, layout, series, z_train);

        if (model == "VarNormalGamma")
        {
            write_var_normal_gamma(file, varsel, covar, layout);
        }
        else if (model == "VarNormalStochvol")
        {
            write_var_normal_stochvol(file, varsel, covar, layout);
        }
        else if (model == "VarTvpWishart")
        {
            write_var_tvp_wishart(file, varsel, layout);
        }
        else if (model == "VarTvpStochvol")
        {
            write_var_tvp_stochvol(file, varsel, covar, layout);
        }
        else
        {
            write_var_tvp_gamma(file, varsel, covar, layout);
        }

        std::cout << "wrote " << dest.string() << " (" << model << ", varsel=" << varsel
                  << ", covar=" << covar << ", structural=" << structural << ", h=" << h
                  << ", k=" << kK << ", tt=" << kTT << ", nparams=" << layout.nparams << ")\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
