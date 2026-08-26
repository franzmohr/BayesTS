// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// Builds a complete model file for the samplers that have no checked-in
// fixtures.
//
// VarNormalWishart and VecNormalWishart are the only models with recorded
// inputs, so the rest could not be put through the same before/after comparison
// as a refactor. make_varsel_fixture cannot fill the gap: it decorates an
// existing file, and there is nothing to decorate. This tool writes one from
// scratch.
//
//   make_model_fixture <dest.h5> <model> <varsel> <covar> <structural> <h>
//
//     model       VarNormalGamma | VarNormalStochvol | VarTvpGamma
//                 | VarTvpWishart | VarTvpStochvol
//                 | VecKlgs2010 | VecNormalGamma | VecNormalStochvol
//                 | VecTvpGamma | VecTvpWishart | VecTvpStochvol
//     varsel      none | ssvs | bvs        (ssvs reaches VarNormalGamma and
//                                           VecNormalGamma only; VecKlgs2010
//                                           takes none at all)
//     covar       0 | 1                    the "+covar" error specification
//     structural  0 | 1                    (refused with a Wishart error
//                                           precision or a covariance block --
//                                           A_0 is not identified against an
//                                           unrestricted Sigma)
//     h           forecast horizon; 0 writes no forecast regressors
//
// The VECs are written from a different set of dimensions and a different
// regressor layout than the VAR models -- differences with an error correction
// term, and a forecast in levels -- so they have a block of their own below
// rather than a branch inside the VAR one. Only the coefficient count differs
// between a structural VEC and a plain one: the contemporaneous columns go on
// the end of /data/train/z and the coefficients on the end of `a`, while the
// forecast regressors are unchanged, because the samplers split that block off
// the posterior rather than reading a column for it.
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

// The VEC, in the level orders VarSpec counts in. p = 2 leaves one lagged
// difference, which is what makes the model more than its loadings and puts the
// sampler through the branch that nets the rest of the regressors out before
// beta is drawn; p = 1 would have skipped it. The constant is restricted to the
// cointegration space, so `n` is zero and n_restricted is one.
constexpr int kVecP = 2;
constexpr int kVecRank = 1;
constexpr int kVecNRestricted = 1;
constexpr int kVecKBeta = kK + kM + kVecNRestricted; // 4
constexpr int kVecNAlpha = kK * kVecRank;            // 3
constexpr int kVecNBeta = kVecKBeta * kVecRank;      // 4
constexpr int kVecNGamma = kK * kK * (kVecP - 1);    // 9
constexpr int kVecNParams = kVecNAlpha + kVecNGamma; // 12

// Written out rather than left to the default, and deliberately not equal to
// it: a fixture that agreed with the default would pass just as well if the
// dataset were never read.
constexpr double kVecRho = 0.99;

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
///
/// `skip` leaves the first that many coefficients out of the selection while
/// still giving them an inclusion probability, which is what a VEC needs: the
/// prior is indexed by the whole of `a`, but selection may not reach the
/// loadings at the front of it.
void write_selection(HighFive::File &file, const std::string &prior_group, arma::uword n,
                     arma::uword skip = 0)
{
    std::vector<int> include;
    for (arma::uword i = skip; i < n; ++i)
    {
        include.push_back(static_cast<int>(i) + 1);
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

//////////////////////////////////////////////////////////////////////////////
// The VEC models.
//
// Different dimensions and a different regressor layout from the VAR block
// above -- differences with an error correction term, and a forecast in levels
// -- so they get their own data builders rather than a branch inside those. All
// five share every one of them: what differs between a VecNormalGamma and a
// VecTvpStochvol is the priors and the initial values, never the sample.

/// A stable VAR(1) in levels, kVecP periods longer than the sample so that the
/// first observation has both a lagged level for the error correction term and
/// a lagged difference for Gamma_1.
///
/// Not a cointegrated process, deliberately: the fixture is an input to a
/// fingerprint comparison, not a recovery experiment, and a series that has to
/// be simulated from a rank-one system is one more thing to keep reproducible
/// for no gain.
arma::mat simulate_levels(Lcg &rng)
{
    arma::mat levels(kK, kTT + kVecP, arma::fill::zeros);
    for (int t = 1; t < kTT + kVecP; ++t)
    {
        for (int i = 0; i < kK; ++i)
        {
            levels(i, t) = 0.5 * levels(i, t - 1) + rng.normal();
        }
    }
    return levels;
}

/// The cointegration vectors the file starts from, normalised on the first
/// variable. A point in the space rather than an identifying restriction --
/// what pins the scale is the prior, whichever of the two the model carries.
arma::vec vec_initial_beta()
{
    arma::vec beta(kVecNBeta, arma::fill::zeros);
    beta(0) = 1.0;
    return beta;
}

/// The response: first differences of the levels, k x tt.
arma::mat build_vec_differences(const arma::mat &levels)
{
    arma::mat dy(kK, kTT);
    for (int t = 0; t < kTT; ++t)
    {
        dy.col(t) = levels.col(kVecP + t) - levels.col(kVecP + t - 1);
    }
    return dy;
}

/// The error correction term, tt x k_beta: the levels of the endogenous
/// variables one period back, then the constant restricted to the
/// cointegration space.
arma::mat build_vec_w(const arma::mat &levels)
{
    arma::mat w(kTT, kVecKBeta, arma::fill::ones);
    for (int t = 0; t < kTT; ++t)
    {
        w.submat(t, 0, t, kK - 1) = arma::trans(levels.col(kVecP + t - 1));
    }
    return w;
}

/// tt x k(p-1), the compact reading of the same lagged differences the SUR
/// regressors below kronecker up with I_k.
///
/// Written into every VEC fixture, next to `z` rather than instead of it: the
/// two layouts describe one sample, and a file that carries both can be read by
/// either sampler -- which is what makes VecKlgs2010 and VecNormalWishart
/// comparable on the same input. Only the /model/algorithm attribute decides
/// which one runs, and each reads only its own dataset.
arma::mat build_vec_compact_regressors(const arma::mat &levels)
{
    arma::mat x(kTT, kK * (kVecP - 1));
    for (int t = 0; t < kTT; ++t)
    {
        x.row(t) = arma::trans(levels.col(kVecP + t - 1) - levels.col(kVecP + t - 2));
    }
    return x;
}

/// (tt * k) x nparams. The loadings' columns come first and hold
/// kron((beta' w_t)', I_k) for the beta the file starts from -- every sampler
/// rebuilds them from its own draw before it reads them, so they are here for a
/// file that can be read on its own rather than because anything depends on
/// them. Then the lagged difference, then the structural columns.
arma::mat build_vec_train_regressors(const arma::mat &levels, const arma::mat &w, int n_structural,
                                     Lcg &rng)
{
    const arma::mat diag_k = arma::eye<arma::mat>(kK, kK);
    const arma::mat beta = arma::reshape(vec_initial_beta(), kVecKBeta, kVecRank);
    arma::mat z(kTT * kK, kVecNParams + n_structural, arma::fill::zeros);

    for (int t = 0; t < kTT; ++t)
    {
        z.submat(t * kK, 0, (t + 1) * kK - 1, kVecNAlpha - 1) = arma::kron(
            arma::trans(arma::trans(beta) * arma::trans(w.row(t))), diag_k);

        const arma::vec lagged_difference =
            levels.col(kVecP + t - 1) - levels.col(kVecP + t - 2);
        z.submat(t * kK, kVecNAlpha, (t + 1) * kK - 1, kVecNParams - 1) =
            arma::kron(arma::trans(lagged_difference), diag_k);

        for (int j = 0; j < n_structural; ++j)
        {
            for (int i = 0; i < kK; ++i)
            {
                z(t * kK + i, kVecNParams + j) = 0.2 * rng.normal();
            }
        }
    }
    return z;
}

/// (h * k) x k(k p + n_restricted), in the *level* layout every VEC forecast
/// expects: p blocks of endogenous lags, then the constant, which was restricted
/// to the cointegration space and becomes an ordinary regressor of the level
/// VAR. Nothing here is in differences, and none of it can be derived from
/// /data/train/z -- which is the demand on the caller the samplers document.
///
/// update_forecast_lags() overwrites the block for lag j at horizon i only once
/// i >= j, so with two lags both blocks have to be seeded at horizon 0 and the
/// second one again at horizon 1.
arma::mat build_vec_forecast_regressors(const arma::mat &levels, int h)
{
    const arma::mat diag_k = arma::eye<arma::mat>(kK, kK);
    const int lag_cols = kK * kK * kVecP;
    const int ncols = lag_cols + kK * kVecNRestricted;
    arma::mat z(h * kK, ncols, arma::fill::zeros);

    for (int i = 0; i < h; ++i)
    {
        z.submat(i * kK, lag_cols, (i + 1) * kK - 1, ncols - 1) = diag_k;
    }

    const arma::mat y_last = arma::kron(arma::trans(levels.col(kTT + kVecP - 1)), diag_k);
    const arma::mat y_before = arma::kron(arma::trans(levels.col(kTT + kVecP - 2)), diag_k);

    z.submat(0, 0, kK - 1, kK * kK - 1) = y_last;
    z.submat(0, kK * kK, kK - 1, 2 * kK * kK - 1) = y_before;
    if (h > 1)
    {
        z.submat(kK, kK * kK, 2 * kK - 1, 2 * kK * kK - 1) = y_last;
    }

    return z;
}

/// The error specification attribute, which is what every reader dispatches on.
/// VecTvpWishart carries no psi block, so it has no "+covar" spelling to reach.
std::string vec_error_spec(const std::string &model, bool covar)
{
    if (model == "VecTvpWishart" || model == "VecKlgs2010")
    {
        return "wishart";
    }
    const std::string prefix =
        (model == "VecNormalStochvol" || model == "VecTvpStochvol") ? "sv" : "gamma";
    return covar ? prefix + "+covar" : prefix;
}

void write_vec_common(HighFive::File &file, const std::string &model, const std::string &varsel,
                      bool covar, bool structural, int h, const arma::mat &levels,
                      const arma::mat &dy, const arma::mat &w, const arma::mat &z_train,
                      const arma::mat &x_train)
{
    write_attribute<std::string>(file, "/model", "algorithm", model);
    write_attribute<int>(file, "/model", "k", kK);
    write_attribute<int>(file, "/model", "p", kVecP);
    write_attribute<int>(file, "/model", "m", kM);
    write_attribute<int>(file, "/model", "s", kS);

    // Zero unrestricted deterministic terms: the intercept the VAR fixtures
    // carry is inside the cointegration space here, and counted by
    // n_restricted instead.
    write_attribute<int>(file, "/model", "n", 0);
    write_attribute<int>(file, "/model", "n_restricted", kVecNRestricted);
    write_attribute<int>(file, "/model", "rank", kVecRank);
    write_attribute<int>(file, "/model", "k_beta", kVecKBeta);

    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", varsel);
    write_attribute<bool>(file, "/model", "structural", structural);
    write_attribute<std::string>(file, "/model", "error", vec_error_spec(model, covar));

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/beta");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/initial");

    write_row(file, "/data/train/y", arma::vectorise(dy));
    write_mat(file, "/data/train/w", w);
    write_mat(file, "/data/train/z", z_train);
    write_mat(file, "/data/train/x", x_train);

    if (h > 0)
    {
        ensure_group(file, "/data/forecast");
        write_mat(file, "/data/forecast/z", build_vec_forecast_regressors(levels, h));
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// Selection over a VEC's coefficients starts past the loadings: every VEC's
/// validate() rejects a scheme that reaches them, and an indicator left at one
/// for a position the sweep never visits is how they stay in.
void write_vec_selection(HighFive::File &file, const std::string &varsel, arma::uword nparams,
                         bool with_ssvs)
{
    write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));
    write_selection(file, "/priors/a", nparams, kVecNAlpha);
    if (with_ssvs && varsel == "ssvs")
    {
        write_ssvs(file, "/priors/a", nparams);
    }
}

/// The constant coefficient block: a normal prior and one starting value.
void write_vec_constant_coefficients(HighFive::File &file, const std::string &varsel,
                                     arma::uword nparams, bool with_ssvs)
{
    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));
    write_row(file, "/initial/a", arma::vec(nparams, arma::fill::zeros));

    if (varsel != "none")
    {
        write_vec_selection(file, varsel, nparams, with_ssvs);
    }
}

/// The constant cointegration space prior of Koop, Leon-Gonzalez and Strachan
/// (2010): a scalar shrinkage and the central location of the space, k_beta
/// square rather than n_beta square.
void write_vec_constant_coint(HighFive::File &file)
{
    write_row(file, "/initial/beta", vec_initial_beta());
    write_dataset_double(file, "/priors/beta/v_inv", 0.1);
    write_mat(file, "/priors/beta/p_tau_inv", arma::eye<arma::mat>(kVecKBeta, kVecKBeta));
}

/// The time-varying cointegration space: a path, where it starts, and how fast
/// it may turn. No state variance among them -- the innovation variance is the
/// identity and is what pins beta's scale.
void write_vec_tvp_coint(HighFive::File &file)
{
    const arma::vec beta = vec_initial_beta();
    write_mat(file, "/initial/beta", arma::repmat(beta, 1, kTT));
    write_row(file, "/initial/beta_init", beta);
    write_row(file, "/priors/beta/mu", arma::vec(kVecNBeta, arma::fill::zeros));
    write_mat(file, "/priors/beta/v_inv", arma::eye<arma::mat>(kVecNBeta, kVecNBeta));
    write_dataset_double(file, "/priors/beta/rho", kVecRho);
}

/// The time-varying coefficient block, which is the VAR's: a path, the precision
/// of its innovations and the state it starts from. Selection is the VEC's,
/// though, so it is written here rather than by write_tvp_coefficients().
void write_vec_tvp_coefficients(HighFive::File &file, const std::string &varsel,
                                arma::uword nparams)
{
    write_tvp_coefficients(file, "none", nparams);

    if (varsel != "none")
    {
        write_vec_selection(file, varsel, nparams, false);
    }
}

/// The constant covariance block, as the constant-coefficient VARs write it.
void write_vec_constant_psi(HighFive::File &file, const std::string &varsel, arma::uword n_psi,
                            bool with_ssvs)
{
    ensure_group(file, "/priors/psi");
    write_row(file, "/priors/psi/mu", arma::vec(n_psi, arma::fill::zeros));
    write_mat(file, "/priors/psi/v_inv", arma::eye<arma::mat>(n_psi, n_psi));
    write_row(file, "/initial/psi", arma::vec(n_psi, arma::fill::zeros));

    if (varsel != "none")
    {
        write_row(file, "/initial/psi_lambda", arma::vec(n_psi, arma::fill::ones));
        write_selection(file, "/priors/psi", n_psi);
        if (with_ssvs && varsel == "ssvs")
        {
            write_ssvs(file, "/priors/psi", n_psi);
        }
    }
}

/// The time-varying covariance block, as the time-varying VARs write it --
/// including the selection scheme in its own group, which is why it can differ
/// from the model's.
void write_vec_tvp_psi(HighFive::File &file, const std::string &varsel, arma::uword n_psi)
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

    ensure_group(file, "/model/priors");
    write_attribute<std::string>(file, "/model/priors/psi", "varsel", varsel);

    if (varsel != "none")
    {
        write_row(file, "/initial/psi_lambda", arma::vec(n_psi, arma::fill::ones));
        write_selection(file, "/priors/psi", n_psi);
    }
}

/// The stochastic volatility block, shared by the two VECs that carry one.
void write_vec_stochvol(HighFive::File &file)
{
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
}

/// The independent gamma priors on the error precisions. `initial` is the
/// dataset the model reads its starting precision from: the constant-coefficient
/// VECs call it u_sigma_inv and the time-varying ones u_omega_inv.
void write_vec_gamma_errors(HighFive::File &file, const std::string &initial)
{
    write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(2.0)));
    write_mat(file, initial, arma::eye<arma::mat>(kK, kK));
}

/// The non-SUR Koop, Leon-Gonzalez and Strachan (2010) sampler: the same priors
/// VecNormalWishart carries, and no selection block -- validate() rejects one.
void write_vec_klgs_2010(HighFive::File &file, arma::uword nparams)
{
    write_vec_constant_coefficients(file, "none", nparams, false);
    write_vec_constant_coint(file);

    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));
}

void write_vec_normal_gamma(HighFive::File &file, const std::string &varsel, bool covar,
                            arma::uword nparams)
{
    write_vec_constant_coefficients(file, varsel, nparams, true);
    write_vec_constant_coint(file);
    write_vec_gamma_errors(file, "/initial/u_sigma_inv");

    if (covar)
    {
        write_vec_constant_psi(file, varsel, kK * (kK - 1) / 2, true);
    }
}

void write_vec_normal_stochvol(HighFive::File &file, const std::string &varsel, bool covar,
                               arma::uword nparams)
{
    write_vec_constant_coefficients(file, varsel, nparams, false);
    write_vec_constant_coint(file);
    write_vec_stochvol(file);

    if (covar)
    {
        write_vec_constant_psi(file, varsel, kK * (kK - 1) / 2, false);
    }
}

void write_vec_tvp_wishart(HighFive::File &file, const std::string &varsel, arma::uword nparams)
{
    write_vec_tvp_coefficients(file, varsel, nparams);
    write_vec_tvp_coint(file);

    // The error covariance is the Wishart precision alone: no psi block, so
    // nothing here depends on the covar flag.
    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));
}

void write_vec_tvp_gamma(HighFive::File &file, const std::string &varsel, bool covar,
                         arma::uword nparams)
{
    write_vec_tvp_coefficients(file, varsel, nparams);
    write_vec_tvp_coint(file);
    write_vec_gamma_errors(file, "/initial/u_omega_inv");

    if (covar)
    {
        write_vec_tvp_psi(file, varsel, kK * (kK - 1) / 2);
    }
}

void write_vec_tvp_stochvol(HighFive::File &file, const std::string &varsel, bool covar,
                            arma::uword nparams)
{
    write_vec_tvp_coefficients(file, varsel, nparams);
    write_vec_tvp_coint(file);
    write_vec_stochvol(file);

    if (covar)
    {
        write_vec_tvp_psi(file, varsel, kK * (kK - 1) / 2);
    }
}

/// Dispatches to the six above. Returns false if the name is not a VEC.
bool write_vec_model(HighFive::File &file, const std::string &model, const std::string &varsel,
                     bool covar, arma::uword nparams)
{
    if (model == "VecKlgs2010")
    {
        write_vec_klgs_2010(file, nparams);
    }
    else if (model == "VecNormalGamma")
    {
        write_vec_normal_gamma(file, varsel, covar, nparams);
    }
    else if (model == "VecNormalStochvol")
    {
        write_vec_normal_stochvol(file, varsel, covar, nparams);
    }
    else if (model == "VecTvpWishart")
    {
        write_vec_tvp_wishart(file, varsel, nparams);
    }
    else if (model == "VecTvpGamma")
    {
        write_vec_tvp_gamma(file, varsel, covar, nparams);
    }
    else if (model == "VecTvpStochvol")
    {
        write_vec_tvp_stochvol(file, varsel, covar, nparams);
    }
    else
    {
        return false;
    }
    return true;
}

bool is_vec_model(const std::string &model)
{
    return model == "VecKlgs2010" || model == "VecNormalGamma" || model == "VecNormalStochvol" ||
           model == "VecTvpWishart" || model == "VecTvpGamma" || model == "VecTvpStochvol";
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

    const bool is_vec = is_vec_model(model);

    if (model != "VarNormalGamma" && model != "VarNormalStochvol" && model != "VarTvpGamma" &&
        model != "VarTvpWishart" && model != "VarTvpStochvol" && !is_vec)
    {
        std::cerr << "Unknown model: " << model << '\n';
        return 2;
    }
    if (varsel != "none" && varsel != "ssvs" && varsel != "bvs")
    {
        std::cerr << "Unknown variable selection scheme: " << varsel << '\n';
        return 2;
    }

    // The combination every validate() now rejects: A_0 is identified only
    // against a diagonal error covariance, and both a Wishart prior and a
    // covariance block leave Sigma unrestricted. Refused here rather than
    // written, because a fixture the sampler will not accept is a golden test
    // that passes while producing nothing -- see the note in CONTRIBUTING.md.
    const bool wishart_errors = model == "VarNormalWishart" || model == "VarTvpWishart" ||
                                model == "VecNormalWishart" || model == "VecTvpWishart" ||
                                model == "VecKlgs2010";
    if (structural && kK > 1 && (wishart_errors || covar))
    {
        std::cerr << "structural is not identified with "
                  << (wishart_errors ? "a Wishart error precision" : "a covariance block")
                  << ": see require_identified_structural() in src/core/inputs.cpp\n";
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

        if (is_vec)
        {
            const arma::mat levels = simulate_levels(rng);
            const arma::mat dy = build_vec_differences(levels);
            const arma::mat w = build_vec_w(levels);
            const int n_structural = layout.n_structural;
            const arma::uword nparams = static_cast<arma::uword>(kVecNParams + n_structural);
            const arma::mat z_train = build_vec_train_regressors(levels, w, n_structural, rng);
            const arma::mat x_train = build_vec_compact_regressors(levels);

            write_vec_common(file, model, varsel, covar, structural, h, levels, dy, w, z_train,
                             x_train);
            write_vec_model(file, model, varsel, covar, nparams);

            std::cout << "wrote " << dest.string() << " (" << model << ", varsel=" << varsel
                      << ", covar=" << covar << ", structural=" << structural << ", h=" << h
                      << ", k=" << kK << ", tt=" << kTT << ", rank=" << kVecRank
                      << ", nparams=" << nparams << ")\n";
            return 0;
        }

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
