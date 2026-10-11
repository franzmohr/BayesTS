// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// Builds a complete model file for every registered algorithm, so that all
// twenty-six can be put through the same before/after fingerprint comparison
// from a clean clone. No model file is checked in -- *.h5 is gitignored, and a
// recorded one is mostly posterior draws -- so this tool writes one from
// scratch.
//
//   make_model_fixture <dest.h5> <model> <varsel> <covar> <structural> <h> [group]
//
//     model       VarNormalWishart | VarNormalGamma | VarNormalStochvol
//                 | VarTvpGamma | VarTvpWishart | VarTvpStochvol
//                 | VarNormalAld | VarTvpAld | VarTvpDiscount
//                 | VecKlgs2010 | VecNormalWishart | VecNormalGamma
//                 | VecNormalStochvol | VecTvpGamma | VecTvpWishart
//                 | VecTvpStochvol | VecTvpDiscount
//                 | DfmNormalGamma | DfmNormalStochvol
//                 | DfmTvpGamma | DfmTvpStochvol | FavarNormalWishart
//                 | UcNormalGamma | UcNormalWishart | UcNormalStochvol
//                 | UcNormalSkew
//     varsel      none | ssvs | bvs        (ssvs reaches the four models with
//                                           constant coefficients and no
//                                           stochastic volatility only;
//                                           VecKlgs2010, the factor models and
//                                           the two discounted ones take none
//                                           at all)
//     covar       0 | 1                    the "+covar" error specification
//     structural  0 | 1                    (refused with a Wishart error
//                                           precision or a covariance block --
//                                           A_0 is not identified against an
//                                           unrestricted Sigma)
//     h           forecast horizon; 0 writes no forecast regressors
//     group       the group to write the model under, e.g. /models/3. Optional;
//                 without it the model's tree hangs at the root of the file,
//                 which is where a file holding one model puts it. With it the
//                 same file can be read back by `bayests <command> --group`, so
//                 a run over a nested model is comparable to one over a model at
//                 the root.
//     --noncentred
//                 writes omega_v in place of shape and rate in every prior on
//                 how far a random walk moves: the non-centred parameterisation.
//                 Refused for any model but VarTvpStochvol, VarTvpGamma,
//                 VecTvpStochvol and VecTvpGamma, the four that read it.
//     --deterministic
//                 for a factor model: writes /model/n = 3 deterministic terms --
//                 a constant, a linear trend and a dummy on every fourth period
//                 -- under /data/train/x and /data/forecast/x, with their prior
//                 and starting value under /priors/c and /initial/c, and shifts
//                 the simulated panel by them so the sampler has something to
//                 find. FavarNormalWishart also gets /priors/c_obs and
//                 /initial/c_obs, and its observed factors are shifted too.
//                 Refused for any model but the four DFMs and the FAVAR, and
//                 UcNormalGamma, for which it writes a drift with a break: a
//                 constant and a step dummy from the middle of the sample.
//     --iid
//                 writes /model/n_iid = 1, so the first endogenous variable's
//                 equation carries no coefficients at all: white noise, related
//                 to the rest of the model only through Sigma. Refused for any
//                 model but VarNormalWishart, which is the one the fixtures
//                 cover it on; the other three constant VARs read it too and
//                 are covered by unit.iid_block.
//     --mixed
//                 writes /data/train/constraints for a panel not observed whole:
//                 the third series quarterly, each quarter's three-month average
//                 observed exactly, the second series missing its last two
//                 months, the first observed throughout. /data/train/y holds
//                 zeros wherever nothing was observed. VarNormalWishart only.
//     --soft
//                 with --mixed: the quarterly averages observed up to an error,
//                 one soft group whose precision is drawn, with its gamma prior
//                 at /priors/constraints and its start at /initial/constraints_inv.
//     --missing
//                 the same, with every series monthly but the second missing its
//                 last two months and the third missing a month in the middle:
//                 missing values and a ragged edge, no aggregation.
//     --partial-test
//                 with --score: /data/test/constraints, a horizon realised in
//                 part -- the first series throughout, the third as the average
//                 of the first three periods, the second not at all -- and
//                 /data/test/y with zeros where nothing was realised.
//                 VarNormalWishart only.
//     --scenario
//                 /data/forecast/constraints, a forecast conditioned on the first
//                 series' path over two periods and the third series' average
//                 over three. Needs h >= 3 and no --score. VarNormalWishart only.
//     --shrinkage=minnesota | --shrinkage=horseshoe | --shrinkage=normal_gamma
//                 writes /model/shrinkage and /priors/a/shrinkage: own lags in
//                 group 1, other lags in group 2, the intercepts as given, and
//                 under minnesota an inverse gamma per group, under
//                 normal_gamma theta = 0.1 and a gamma on each global rate. The three
//                 constant-coefficient Gaussian VARs only, without varsel.
//     --quantile-grid
//                 VarNormalAld only: writes /model/quantiles, a grid of five
//                 levels from 0.1 to 0.9, which makes the model a structural
//                 quantile VAR. With a structural form it takes a horizon, and
//                 its forecast is simulated through the grid.
//     --draw-theta
//                 with --shrinkage=normal_gamma: theta drawn under an
//                 exponential prior, /priors/a/shrinkage/theta_rate in place of
//                 theta, started at /initial/a_theta.
//     --stationary
//                 writes /model/stationary = true. The same three only.
//     --steady-state
//                 writes /model/steady_state = true, a normal prior on the
//                 unconditional mean at /priors/mu and its start at /initial/mu.
//                 The same three only, without --shrinkage.
//     --volatility-break
//                 VarNormalWishart only: writes /model/volatility_break at the
//                 fifth period from the end, three free scales, a Pareto(1)
//                 prior on them and the beta of Lenza and Primiceri (2022) on
//                 the decay, at /priors/volatility, and a start at
//                 /initial/volatility. Not with --steady-state or a panel or
//                 horizon not observed whole.
//     --smooth
//                 UcNormalGamma only: writes /model/trend = "smooth", the trend
//                 whose second difference is the shock, with no drift and a
//                 tau0 of two periods.
//     --damped
//                 UcNormalGamma and UcNormalWishart: writes /model/trend =
//                 "damped", the slope an AR(1) around its mean, with a normal
//                 prior at /priors/rho and its start at /initial/rho, and a tau0
//                 of two periods. The drift is the one the rw trend has.
//     --signals
//                 UcNormalGamma and UcNormalWishart: two signal equations, one
//                 per series, each on a constant and the cycles at lags zero and
//                 one -- /model/n_signals, signal_lags and n_signal_x, the data
//                 at /data/train/signal and signal_x, the priors at
//                 /priors/signal and /priors/signal_noise.
//     --stationary-init
//                 UcNormalGamma and UcNormalWishart: writes /model/cycle_init =
//                 "stationary" together with /model/stationary, the cycle
//                 before the sample drawn from its stationary distribution.
//     --own-cycle
//                 The unobserved components models: writes /model/cycle = "own",
//                 each cycle an AR(p) in its own lags. --smooth is theirs too.
//     --irregular
//                 The same four: writes /model/irregular = true, with its gamma
//                 prior at /priors/irregular and its start at
//                 /initial/irregular_inv.
//     --ald, --skew-trend, --skew-index
//                 UcNormalSkew only: the asymmetric Laplace in place of the skew
//                 normal, the trend shock skewed rather than the cycle's, and one
//                 index variable in the scales, /data/train/skew_x and
//                 /data/forecast/skew_x.
//     --hold-states
//                 writes /model/forecast_states = "hold", so the forecast keeps
//                 the last in-sample states rather than simulating them forward.
//                 Refused for any model but the time-varying VARs, VECs and
//                 factor models that read it.
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
#include <utility>
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

// The support of the uniform prior on rho, written only by the `--coint-rho`
// fixtures. Neither end is the one Koop, Leon-Gonzalez and Strachan use --
// theirs is (0.999, 1), narrow enough that the truncation does all the work and
// a wrong conditional would still land inside it. This one is wide enough for
// the draw to move, and kVecRho above sits inside it, which is what the
// starting-value check in validate_tvp_coint_rho() demands.
constexpr double kVecRhoMin = 0.9;
constexpr double kVecRhoMax = 0.999;

// The spread of the informative marginal prior on the space, written only by the
// `--coint-p-tau` fixtures as P_tau = H H' + tau H_perp H_perp', with H the
// cointegration vector the chain starts from. Away from both ends: at one P_tau
// is the identity and the fixture would run the noninformative path it is there
// to differ from, and at zero the directions off sp(H) forget their past
// entirely, which is admissible but the less interesting half of the range.
constexpr double kVecTau = 0.5;

// The two discount factors of VarTvpDiscount and VecTvpDiscount. Both away
// from one, so the fixtures run the drifting model rather than the conjugate
// one at both discounts of one -- which is the case the unit tests pin against
// a closed form and needs no fixture. delta_sigma settles the degrees of
// freedom at 1 / (1 - 0.95) = 20, which clears k = 3 as validate() demands;
// anything below 1 - 1/k would be refused rather than written.
constexpr double kDeltaBeta = 0.98;
constexpr double kDeltaSigma = 0.95;

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

void write_mat(const ModelFile &file, const std::string &dataset, const arma::mat &m)
{
    write_armadillo_matrix_to_hdf5(file, dataset, m, false);
}

/// Vectors live on disk as a single HDF5 row, which is how the R writer stores
/// them. Written as an Armadillo column, because the writer transposes on the
/// way out -- and the orientation matters: the readers that skip vectorise()
/// and assign the result straight to an arma::vec only accept a column back.
void write_row(const ModelFile &file, const std::string &dataset, const arma::vec &v)
{
    write_armadillo_matrix_to_hdf5(file, dataset, arma::mat(v), false);
}

/// A one-element integer dataset read back with get_dataset_int(), which reads
/// a scalar rather than a row.
void write_int_scalar(const ModelFile &file, const std::string &dataset, int value)
{
    if (file.exist(dataset))
    {
        file.unlink(dataset);
    }
    file.createDataSet<int>(dataset, HighFive::DataSpace::From(value)).write(value);
}

void write_int_row(const ModelFile &file, const std::string &dataset,
                   const std::vector<int> &values)
{
    if (file.exist(dataset))
    {
        file.unlink(dataset);
    }
    const std::vector<std::vector<int>> data{values};
    file.createDataSet<int>(dataset, HighFive::DataSpace::From(data)).write(data);
}

void ensure_group(const ModelFile &file, const std::string &group)
{
    if (!file.exist(group))
    {
        file.createGroup(group);
    }
}

template <typename T>
void write_attribute(const ModelFile &file, const std::string &group_name,
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

/// A vector of doubles as one attribute, as h5py and R write one.
void write_vector_attribute(const ModelFile &file, const std::string &group_name,
                            const std::string &name, const std::vector<double> &value)
{
    ensure_group(file, group_name);
    HighFive::Group group = file.getGroup(group_name);
    if (group.hasAttribute(name))
    {
        group.deleteAttribute(name);
    }
    group.createAttribute<double>(name, HighFive::DataSpace::From(value)).write(value);
}

/// Positions are stored one-based, matching R and the rest of the format.
///
/// `skip` leaves the first that many coefficients out of the selection while
/// still giving them an inclusion probability, which is what a VEC needs: the
/// prior is indexed by the whole of `a`, but selection may not reach the
/// loadings at the front of it.
void write_selection(const ModelFile &file, const std::string &prior_group, arma::uword n,
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
void write_ssvs(const ModelFile &file, const std::string &prior_group, arma::uword n)
{
    write_row(file, prior_group + "/tau0", arma::vec(n, arma::fill::value(0.1)));
    write_row(file, prior_group + "/tau1", arma::vec(n, arma::fill::value(10.0)));
}

struct Layout
{
    int n_lag = kK * kK * kP;      // 9
    int n_non_structural = kK * (kK * kP + kM * (kS + 1) + kN); // 12, lags + intercepts
    // The same two counts in the compact layout the forecast regressors use:
    // one column per regressor rather than k of them. /data/train/z is still
    // the SUR form, so both spellings are needed in the one fixture.
    int n_x_lag = kK * kP;                             // 3
    int n_x = kK * kP + kM * (kS + 1) + kN;            // 4
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

/// h x n_x, one period per row. Only the first lag block and the intercepts are
/// filled in; the sampler overwrites the rest as the path unfolds. The
/// structural columns are deliberately absent -- the forecast code splits them
/// off the posterior draws instead.
arma::mat build_forecast_regressors(const arma::mat &series, const Layout &layout, int h)
{
    arma::mat x(h, layout.n_x, arma::fill::zeros);

    // The intercept, which the SUR spelling of this carried as a k x k identity
    // per horizon and which is one number per period here.
    x.cols(layout.n_x_lag, layout.n_x - 1).ones();
    x.submat(0, 0, 0, layout.n_x_lag - 1) = arma::trans(series.col(kTT + kP - 1));

    return x;
}

/// The observations the horizon realised, written where a fixture asks to be
/// scored: one row per period and one column per variable, the layout
/// read_test_observations() expects. `bayests forecasts` writes
/// /posterior/forecast/loglik wherever it finds this dataset and the algorithm
/// can be scored, so this is the only thing standing between a fixture and the
/// score path.
///
/// The values continue the same stable AR(1) the sample was drawn from, from
/// its last in-sample column -- `last` -- so a scored fixture is a forecast
/// against a plausible future rather than against noise a model must strain to
/// explain.
///
/// Drawn from a generator of its own rather than the file's. That is what makes
/// `--score` additive: the file's stream is consumed in the same order and the
/// same number of times whether the flag is there or not, so adding the flag to
/// a fixture row adds a dataset and moves no number that was there before.
arma::mat write_test_observations(const ModelFile &file, const arma::vec &last, int h)
{
    Lcg rng(20260920ULL);

    arma::mat realised(static_cast<arma::uword>(h), last.n_elem, arma::fill::zeros);
    arma::vec previous = last;
    for (int t = 0; t < h; ++t)
    {
        for (arma::uword i = 0; i < previous.n_elem; ++i)
        {
            previous(i) = 0.5 * previous(i) + rng.normal();
        }
        realised.row(static_cast<arma::uword>(t)) = arma::trans(previous);
    }

    ensure_group(file, "/data/test");
    write_mat(file, "/data/test/y", realised);
    return realised;
}

/// Six one-row datasets under `group`, the constraint set a host writes. Rows,
/// periods and variables are zero-based here and one-based in the file.
void write_constraint_set(const ModelFile &file, const std::string &group,
                          const std::vector<double> &value,
                          const std::vector<std::vector<std::pair<int, int>>> &entries,
                          const std::vector<double> &weights)
{
    std::vector<double> row, period, variable, weight;
    for (std::size_t r = 0; r < entries.size(); ++r)
    {
        for (const auto &e : entries[r])
        {
            row.push_back(static_cast<double>(r + 1));
            period.push_back(static_cast<double>(e.first + 1));
            variable.push_back(static_cast<double>(e.second + 1));
            weight.push_back(weights[r]);
        }
    }
    ensure_group(file, group);
    write_row(file, group + "/value", arma::vec(value));
    write_row(file, group + "/group", arma::vec(value.size(), arma::fill::zeros));
    write_row(file, group + "/row", arma::vec(row));
    write_row(file, group + "/period", arma::vec(period));
    write_row(file, group + "/variable", arma::vec(variable));
    write_row(file, group + "/weight", arma::vec(weight));
}

/// --partial-test: what a horizon of at least three periods realised in part,
/// and /data/test/y again with zeros where it realised nothing.
void write_partial_test(const ModelFile &file, arma::mat realised)
{
    std::vector<double> value;
    std::vector<std::vector<std::pair<int, int>>> entries;
    std::vector<double> weights;
    for (int t = 0; t < static_cast<int>(realised.n_rows); ++t)
    {
        value.push_back(realised(static_cast<arma::uword>(t), 0));
        entries.push_back({{t, 0}});
        weights.push_back(1.0);
    }
    value.push_back(arma::mean(realised.col(2).head(3)));
    entries.push_back({{0, 2}, {1, 2}, {2, 2}});
    weights.push_back(1.0 / 3.0);
    write_constraint_set(file, "/data/test/constraints", value, entries, weights);

    realised.col(1).zeros();
    realised.col(2).zeros();
    write_mat(file, "/data/test/y", realised);
}

/// --scenario: the first series at 0.5 and then -0.25, the third averaging 0.1
/// over the first three periods of the horizon.
void write_scenario(const ModelFile &file)
{
    write_constraint_set(file, "/data/forecast/constraints", {0.5, -0.25, 0.1},
                         {{{0, 0}}, {{1, 0}}, {{0, 2}, {1, 2}, {2, 2}}}, {1.0, 1.0, 1.0 / 3.0});
}

/// tt x n_x, the compact reading of the same regressors build_train_regressors()
/// kroneckers up with I_k -- the one lag block and the intercept, one column
/// each rather than k of them.
///
/// Written only into a discounted VAR's file, which reads one k x n_x
/// coefficient matrix per period and has no SUR design to undo. The two
/// layouts describe one sample, the way they do for a VEC.
arma::mat build_var_compact_regressors(const arma::mat &series, const Layout &layout)
{
    arma::mat x(kTT, layout.n_x, arma::fill::zeros);
    x.cols(layout.n_x_lag, layout.n_x - 1).ones();
    for (int t = 0; t < kTT; ++t)
    {
        // `series` runs kP ahead of the sample, so column t is period t's lag.
        x.submat(t, 0, t, layout.n_x_lag - 1) = arma::trans(series.col(t));
    }
    return x;
}

/// /data/train/constraints for --mixed and --missing, and /data/train/y again
/// with zeros where nothing was observed: the panel as a host writes one it did
/// not observe whole. Positions are one-based, as the file counts them; each
/// dataset is one row, how every other vector here is stored.
void write_incomplete_panel(const ModelFile &file, const arma::mat &series, const bool mixed,
                            const bool soft)
{
    const arma::mat sample = series.cols(kP, kTT + kP - 1); // k x tt
    arma::mat held = sample;
    std::vector<double> value, group, row, period, variable, weight;
    const auto add = [&](const double v, const std::vector<std::pair<int, int>> &entries,
                         const double w, const double g = 0.0) {
        value.push_back(v);
        group.push_back(g);
        for (const auto &e : entries)
        {
            row.push_back(static_cast<double>(value.size()));
            period.push_back(static_cast<double>(e.first + 1));
            variable.push_back(static_cast<double>(e.second + 1));
            weight.push_back(w);
        }
    };

    for (int t = 0; t < kTT; ++t)
    {
        add(sample(0, t), {{t, 0}}, 1.0);
        if (t < kTT - 2)
        {
            add(sample(1, t), {{t, 1}}, 1.0);
        }
        else
        {
            held(1, t) = 0.0;
        }
        if (!mixed && t != kTT / 2)
        {
            add(sample(2, t), {{t, 2}}, 1.0);
        }
    }
    if (mixed)
    {
        held.row(2).zeros();
        for (int q = 0; q < kTT / 3; ++q)
        {
            const double average =
                (sample(2, 3 * q) + sample(2, 3 * q + 1) + sample(2, 3 * q + 2)) / 3.0;
            add(average, {{3 * q, 2}, {3 * q + 1, 2}, {3 * q + 2, 2}}, 1.0 / 3.0,
                soft ? 1.0 : 0.0);
        }
    }
    else
    {
        held(2, kTT / 2) = 0.0;
    }

    write_row(file, "/data/train/y", arma::vectorise(held));
    ensure_group(file, "/data/train/constraints");
    write_row(file, "/data/train/constraints/value", arma::vec(value));
    write_row(file, "/data/train/constraints/group", arma::vec(group));
    write_row(file, "/data/train/constraints/row", arma::vec(row));
    write_row(file, "/data/train/constraints/period", arma::vec(period));
    write_row(file, "/data/train/constraints/variable", arma::vec(variable));
    write_row(file, "/data/train/constraints/weight", arma::vec(weight));

    if (soft)
    {
        // A gamma prior on the one soft group's precision, centred on a
        // standard deviation of about a third, and a start at its mean.
        ensure_group(file, "/priors/constraints");
        write_row(file, "/priors/constraints/shape", arma::vec{2.0});
        write_row(file, "/priors/constraints/rate", arma::vec{0.2});
        write_row(file, "/initial/constraints_inv", arma::vec{10.0});
    }
}

/// --shrinkage: /model/shrinkage and /priors/a/shrinkage. `a` is vec of the
/// k x n_x coefficient matrix, lags first, so coefficient (i, j) sits at j k + i:
/// own lags (j == i within the lag block) in group 1, other lags in group 2,
/// the intercepts in group 0, their prior as given.
void write_shrinkage(const ModelFile &file, const std::string &scheme, const Layout &layout,
                     const bool draw_theta = false)
{
    write_attribute<std::string>(file, "/model", "shrinkage", scheme);
    arma::vec group(static_cast<arma::uword>(layout.nparams), arma::fill::zeros);
    for (int j = 0; j < layout.n_x_lag; ++j)
    {
        for (int i = 0; i < kK; ++i)
        {
            group(static_cast<arma::uword>(j * kK + i)) = (j % kK == i) ? 1.0 : 2.0;
        }
    }
    ensure_group(file, "/priors/a/shrinkage");
    write_row(file, "/priors/a/shrinkage/group", group);
    if (scheme == "minnesota")
    {
        write_row(file, "/priors/a/shrinkage/shape", arma::vec{3.0, 3.0});
        write_row(file, "/priors/a/shrinkage/rate", arma::vec{1.0, 0.25});
    }
    if (scheme == "normal_gamma" && draw_theta)
    {
        write_row(file, "/priors/a/shrinkage/theta_rate", arma::vec{1.0, 1.0});
        write_row(file, "/initial/a_theta", arma::vec{0.1, 0.1});
    }
    else if (scheme == "normal_gamma")
    {
        write_row(file, "/priors/a/shrinkage/theta", arma::vec{0.1, 0.1});
    }
    if (scheme == "normal_gamma")
    {
        write_row(file, "/priors/a/shrinkage/shape", arma::vec{0.01, 0.01});
        write_row(file, "/priors/a/shrinkage/rate", arma::vec{0.01, 0.01});
    }
}

void write_common(const ModelFile &file, const std::string &model, const std::string &varsel,
                  bool covar, bool structural, int h, const Layout &layout,
                  const arma::mat &series, const arma::mat &z_train, int n_iid = 0)
{
    // The CLI dispatches on /model/algorithm; the checked-in fixtures predate
    // it, which is why the golden harness has to tolerate its absence.
    write_attribute<std::string>(file, "/model", "algorithm", model);
    write_attribute<int>(file, "/model", "k", kK);
    write_attribute<int>(file, "/model", "p", kP);
    write_attribute<int>(file, "/model", "m", kM);
    write_attribute<int>(file, "/model", "s", kS);
    write_attribute<int>(file, "/model", "n", kN);
    const bool discount = model == "VarTvpDiscount";

    write_attribute<int>(file, "/model", "iterations", kIterations);

    // Nothing is iterated in the discounted model, so there is nothing to
    // discard: its validate() refuses a burn-in rather than ignoring one, and
    // `iterations` is how many i.i.d. draws its forecast takes.
    write_attribute<int>(file, "/model", "burnin", discount ? 0 : kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", varsel);
    write_attribute<bool>(file, "/model", "structural", structural);

    // Left out altogether where nothing is restricted, so that every fixture
    // but the one that asks for it stays the file it was.
    if (n_iid > 0)
    {
        write_attribute<int>(file, "/model", "n_iid", n_iid);
    }

    if (discount)
    {
        write_attribute<double>(file, "/model", "delta_beta", kDeltaBeta);
        write_attribute<double>(file, "/model", "delta_sigma", kDeltaSigma);
    }

    // The error specification the reader dispatches on. Neither Wishart model
    // carries a psi block, so neither has a "+covar" spelling to reach -- and
    // neither reader consults the attribute at all. It is written so the file
    // can be described on its own.
    std::string prefix = "gamma";
    if (model == "VarNormalStochvol" || model == "VarTvpStochvol")
    {
        prefix = "sv";
    }
    else if (model == "VarNormalWishart" || model == "VarTvpWishart" || discount)
    {
        // The discounted model's error covariance is the inverse Wishart whole,
        // so it has no "+covar" spelling to reach either, and its reader is
        // handed a null expectation the way the Wishart models' are.
        prefix = "wishart";
    }
    else if (model == "VarNormalAld" || model == "VarTvpAld")
    {
        // Written so the file describes itself, though the reader never consults
        // it: a quantile model has no covariance block, so read_spec is handed a
        // null expectation and the attribute has nothing to switch on.
        prefix = "ald";

        // The quantile the model is estimated at. Off the median, so that a
        // fixture whose skew term had been dropped somewhere would move rather
        // than staying put -- at 0.5 theta is zero and the omission is invisible.
        write_attribute<double>(file, "/model", "quantile", 0.25);
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

    // Each file carries the one layout its model reads: writing both made
    // `bayests check` warn, rightly, that one of them is never opened.
    if (discount)
    {
        write_mat(file, "/data/train/x", build_var_compact_regressors(series, layout));
    }
    else
    {
        write_mat(file, "/data/train/z", z_train);
    }

    if (h > 0)
    {
        ensure_group(file, "/data/forecast");
        write_mat(file, "/data/forecast/x", build_forecast_regressors(series, layout, h));
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// The plainest model in the taxonomy: a normal prior on the coefficients and a
/// Wishart one on the whole error precision. No psi block, so no covar
/// parameter -- the "+covar" spelling has nothing to turn on here, the same way
/// it has nothing to turn on in write_var_tvp_wishart below.
void write_var_normal_wishart(const ModelFile &file, const std::string &varsel,
                              const Layout &layout)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);

    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));
    write_row(file, "/initial/a", arma::vec(nparams, arma::fill::zeros));

    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
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
}

/// The discounted VAR: a matrix normal prior on the coefficients, a Wishart one
/// on the error precision, and nothing else.
///
/// Shorter than every writer beside it by exactly what a chain needs and this
/// does not -- no `/initial` group at all, because nothing starts anywhere. The
/// coefficient prior is at `/priors/a/mean` and `/priors/a/cov` rather than the
/// `/priors/a/mu` and `/priors/a/v_inv` every sampler reads: a mean that is an
/// n_x x k matrix rather than a stacked vector, and the regressor side of a
/// covariance rather than a precision over the whole of it. See
/// src/io/hdf5/var_tvp_discount_io.h.
void write_var_tvp_discount(const ModelFile &file, const Layout &layout)
{
    const arma::uword n_x = static_cast<arma::uword>(layout.n_x);

    write_mat(file, "/priors/a/mean", arma::mat(n_x, kK, arma::fill::zeros));
    write_mat(file, "/priors/a/cov", arma::eye<arma::mat>(n_x, n_x));

    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
}

void write_var_normal_gamma(const ModelFile &file, const std::string &varsel, bool covar,
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

void write_var_normal_stochvol(const ModelFile &file, const std::string &varsel, bool covar,
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

/// `noncentred` puts the normal prior on the signed standard deviation in place
/// of the inverse gamma on the variance, for the coefficients and the
/// covariance block, at the prior mean of the variance the inverse gamma has.
void write_var_tvp_gamma(const ModelFile &file, const std::string &varsel, bool covar,
                         const Layout &layout, bool noncentred = false)
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

    if (noncentred)
    {
        write_row(file, "/priors/a/omega_v", arma::vec(nparams, arma::fill::value(0.005)));
    }
    else
    {
        write_row(file, "/priors/a/shape", arma::vec(nparams, arma::fill::value(3.0)));
        write_row(file, "/priors/a/rate", arma::vec(nparams, arma::fill::value(0.01)));
    }
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

        if (noncentred)
        {
            write_row(file, "/priors/psi/omega_v", arma::vec(n_psi, arma::fill::value(0.005)));
        }
        else
        {
            write_row(file, "/priors/psi/shape", arma::vec(n_psi, arma::fill::value(3.0)));
            write_row(file, "/priors/psi/rate", arma::vec(n_psi, arma::fill::value(0.01)));
        }
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
///
/// `noncentred` writes the normal prior on the signed standard deviation in
/// place of the inverse gamma on the variance, with the prior mean of the
/// variance the inverse gamma has -- rate over shape less one, 0.005.
void write_tvp_coefficients(const ModelFile &file, const std::string &varsel, arma::uword nparams,
                            bool noncentred = false)
{
    // The state innovation variances are stored inverted; the sampler flips the
    // diagonal back on the way in. A tight prior keeps the random walk from
    // wandering far enough to make the Kalman filter ill-conditioned.
    write_mat(file, "/initial/a", arma::mat(nparams, kTT, arma::fill::zeros));
    write_mat(file, "/initial/a_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(nparams, arma::fill::value(100.0)))));
    write_row(file, "/initial/a_init", arma::vec(nparams, arma::fill::zeros));

    if (noncentred)
    {
        write_row(file, "/priors/a/omega_v", arma::vec(nparams, arma::fill::value(0.005)));
    }
    else
    {
        write_row(file, "/priors/a/shape", arma::vec(nparams, arma::fill::value(3.0)));
        write_row(file, "/priors/a/rate", arma::vec(nparams, arma::fill::value(0.01)));
    }
    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));

    if (varsel != "none")
    {
        write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));
        write_selection(file, "/priors/a", nparams);
    }
}


/// A VAR estimated at a conditional quantile.
///
/// The coefficient block is VarNormalStochvol's; what replaces the stochastic
/// volatility group is a single inverse gamma on the scale of the asymmetric
/// Laplace, plus the latent scales the mixture is over. Those start at one --
/// the mean of an Exp(1) -- rather than at zero, because they are a variance the
/// first coefficient draw is weighted by and a zero would divide by nothing.
void write_var_normal_ald(const ModelFile &file, const std::string &varsel, const Layout &layout)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);

    write_row(file, "/priors/a/mu", arma::vec(nparams, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(nparams, nparams));
    write_row(file, "/initial/a", arma::vec(nparams, arma::fill::zeros));

    ensure_group(file, "/priors/u_scale");
    write_row(file, "/priors/u_scale/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_scale/rate", arma::vec(kK, arma::fill::value(0.2)));

    write_mat(file, "/initial/w", arma::mat(kTT, kK, arma::fill::ones));
    write_row(file, "/initial/u_scale", arma::vec(kK, arma::fill::ones));

    if (varsel != "none")
    {
        write_row(file, "/initial/a_lambda", arma::vec(nparams, arma::fill::ones));
        write_selection(file, "/priors/a", nparams);
    }
}

/// The same model with the coefficients turned into a path.
void write_var_tvp_ald(const ModelFile &file, const std::string &varsel, const Layout &layout,
                       bool noncentred = false)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);

    write_tvp_coefficients(file, varsel, nparams, noncentred);

    ensure_group(file, "/priors/u_scale");
    write_row(file, "/priors/u_scale/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_scale/rate", arma::vec(kK, arma::fill::value(0.2)));

    write_mat(file, "/initial/w", arma::mat(kTT, kK, arma::fill::ones));
    write_row(file, "/initial/u_scale", arma::vec(kK, arma::fill::ones));
}

void write_var_tvp_wishart(const ModelFile &file, const std::string &varsel, const Layout &layout,
                           bool noncentred = false)
{
    write_tvp_coefficients(file, varsel, static_cast<arma::uword>(layout.nparams), noncentred);

    // The error covariance is the Wishart precision alone: no psi block, so
    // nothing here depends on the covar flag.
    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));
}

/// `noncentred` switches every random walk the model has -- coefficients,
/// covariance block and log-volatility -- to the normal prior on its signed
/// standard deviation, each at the prior mean of the variance its inverse
/// gamma has.
void write_var_tvp_stochvol(const ModelFile &file, const std::string &varsel, bool covar,
                            const Layout &layout, bool noncentred = false)
{
    const arma::uword nparams = static_cast<arma::uword>(layout.nparams);
    const arma::uword n_psi = static_cast<arma::uword>(layout.n_psi);

    write_tvp_coefficients(file, varsel, nparams, noncentred);

    // The offset keeps log(u^2 + offset) finite when a residual lands on zero,
    // and bounds it well inside the range the ten-component mixture covers.
    write_row(file, "/priors/u_sigma/offset", arma::vec(kK, arma::fill::value(1e-4)));
    write_row(file, "/priors/u_sigma/sigma", arma::vec(kK, arma::fill::value(0.1)));
    if (noncentred)
    {
        write_row(file, "/priors/u_sigma/omega_v", arma::vec(kK, arma::fill::value(0.1)));
    }
    else
    {
        write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
        write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(0.2)));
    }
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

        if (noncentred)
        {
            write_row(file, "/priors/psi/omega_v", arma::vec(n_psi, arma::fill::value(0.005)));
        }
        else
        {
            write_row(file, "/priors/psi/shape", arma::vec(n_psi, arma::fill::value(3.0)));
            write_row(file, "/priors/psi/rate", arma::vec(n_psi, arma::fill::value(0.01)));
        }
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
/// Built for every VEC fixture, next to `z`, and written only where the model
/// reads it: the two layouts describe one sample, which is what makes VecKlgs2010
/// and VecNormalWishart comparable on the same input. Writing both into every
/// file made `bayests check` warn, rightly, that the model never reads one of
/// them -- on forty fixtures, which teaches a reader to ignore the warning.
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

/// h x (k p + n_restricted), one period per row, in the *level* layout every VEC
/// forecast expects: p blocks of endogenous lags, then the constant, which was
/// restricted to the cointegration space and becomes an ordinary regressor of
/// the level VAR. Nothing here is in differences, and none of it can be derived
/// from /data/train/z -- which is the demand on the caller the samplers
/// document.
///
/// update_forecast_lags() overwrites the block for lag j at horizon i only once
/// i >= j, so with two lags both blocks have to be seeded at horizon 0 and the
/// second one again at horizon 1.
arma::mat build_vec_forecast_regressors(const arma::mat &levels, int h)
{
    const int lag_cols = kK * kVecP;
    const int ncols = lag_cols + kVecNRestricted;
    arma::mat x(h, ncols, arma::fill::zeros);

    x.cols(lag_cols, ncols - 1).ones();

    const arma::rowvec y_last = arma::trans(levels.col(kTT + kVecP - 1));
    const arma::rowvec y_before = arma::trans(levels.col(kTT + kVecP - 2));

    x.submat(0, 0, 0, kK - 1) = y_last;
    x.submat(0, kK, 0, 2 * kK - 1) = y_before;
    if (h > 1)
    {
        x.submat(1, kK, 1, 2 * kK - 1) = y_last;
    }

    return x;
}

/// The error specification attribute, which is what every reader dispatches on.
/// None of the three Wishart-precision VECs carries a psi block, so none of them
/// has a "+covar" spelling to reach.
std::string vec_error_spec(const std::string &model, bool covar)
{
    if (model == "VecTvpWishart" || model == "VecNormalWishart" || model == "VecKlgs2010" ||
        model == "VecTvpDiscount")
    {
        return "wishart";
    }
    const std::string prefix =
        (model == "VecNormalStochvol" || model == "VecTvpStochvol") ? "sv" : "gamma";
    return covar ? prefix + "+covar" : prefix;
}

void write_vec_common(const ModelFile &file, const std::string &model, const std::string &varsel,
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

    const bool discount = model == "VecTvpDiscount";

    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", discount ? 0 : kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", varsel);
    write_attribute<bool>(file, "/model", "structural", structural);
    write_attribute<std::string>(file, "/model", "error", vec_error_spec(model, covar));

    if (discount)
    {
        write_attribute<double>(file, "/model", "delta_beta", kDeltaBeta);
        write_attribute<double>(file, "/model", "delta_sigma", kDeltaSigma);
    }

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/beta");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/initial");

    write_row(file, "/data/train/y", arma::vectorise(dy));
    write_mat(file, "/data/train/w", w);
    // Both layouts are built from the same levels, so a VecKlgs2010 fixture and a
    // SUR one still describe one sample; each file carries only the one its
    // model reads.
    if (model == "VecKlgs2010" || discount)
    {
        write_mat(file, "/data/train/x", x_train);
    }
    else
    {
        write_mat(file, "/data/train/z", z_train);
    }

    if (h > 0)
    {
        ensure_group(file, "/data/forecast");
        write_mat(file, "/data/forecast/x", build_vec_forecast_regressors(levels, h));
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// Selection over a VEC's coefficients starts past the loadings: every VEC's
/// validate() rejects a scheme that reaches them, and an indicator left at one
/// for a position the sweep never visits is how they stay in.
void write_vec_selection(const ModelFile &file, const std::string &varsel, arma::uword nparams,
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
void write_vec_constant_coefficients(const ModelFile &file, const std::string &varsel,
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
void write_vec_constant_coint(const ModelFile &file)
{
    write_row(file, "/initial/beta", vec_initial_beta());
    write_dataset_double(file, "/priors/beta/v_inv", 0.1);
    write_mat(file, "/priors/beta/p_tau_inv", arma::eye<arma::mat>(kVecKBeta, kVecKBeta));
}

/// The time-varying cointegration space: a path, where it starts, and how fast
/// it may turn. No state variance among them -- the innovation variance is the
/// identity and is what pins beta's scale.
void write_vec_tvp_coint(const ModelFile &file)
{
    const arma::vec beta = vec_initial_beta();
    write_mat(file, "/initial/beta", arma::repmat(beta, 1, kTT));
    write_row(file, "/initial/beta_init", beta);
    write_row(file, "/priors/beta/mu", arma::vec(kVecNBeta, arma::fill::zeros));
    write_mat(file, "/priors/beta/v_inv", arma::eye<arma::mat>(kVecNBeta, kVecNBeta));
    write_dataset_double(file, "/priors/beta/rho", kVecRho);
}

/// The support of the uniform prior on rho, which is what turns rho from a fixed
/// hyperparameter into a drawn one. Written beside the block above rather than
/// inside it, so that the same three models are covered both ways: the fixtures
/// that leave this out still run the fixed-rho path every existing file takes.
void write_vec_tvp_coint_rho_prior(const ModelFile &file)
{
    write_dataset_double(file, "/priors/beta/rho_min", kVecRhoMin);
    write_dataset_double(file, "/priors/beta/rho_max", kVecRhoMax);
}

/// Koop, Leon-Gonzalez and Strachan's informative marginal prior on the space:
/// the transition P_tau, centred on the cointegration vector the chain starts
/// from. Written beside the fixed-rho block for the same reason the rho prior is,
/// and only the transition -- the prior on the state before the sample stays the
/// one write_vec_tvp_coint() wrote, which the sampler reads either way.
void write_vec_tvp_coint_p_tau(const ModelFile &file)
{
    const arma::vec h = arma::normalise(vec_initial_beta());
    const arma::mat along = h * arma::trans(h);
    write_mat(file, "/priors/beta/p_tau",
              along + kVecTau * (arma::eye<arma::mat>(kVecKBeta, kVecKBeta) - along));
}

/// The time-varying coefficient block, which is the VAR's: a path, the precision
/// of its innovations and the state it starts from. Selection is the VEC's,
/// though, so it is written here rather than by write_tvp_coefficients().
void write_vec_tvp_coefficients(const ModelFile &file, const std::string &varsel,
                                arma::uword nparams, bool noncentred = false)
{
    write_tvp_coefficients(file, "none", nparams, noncentred);

    if (varsel != "none")
    {
        write_vec_selection(file, varsel, nparams, false);
    }
}

/// The constant covariance block, as the constant-coefficient VARs write it.
void write_vec_constant_psi(const ModelFile &file, const std::string &varsel, arma::uword n_psi,
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
void write_vec_tvp_psi(const ModelFile &file, const std::string &varsel, arma::uword n_psi,
                       bool noncentred = false)
{
    ensure_group(file, "/priors/psi");
    write_mat(file, "/initial/psi", arma::mat(n_psi, kTT, arma::fill::zeros));
    write_mat(file, "/initial/psi_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(n_psi, arma::fill::value(100.0)))));
    write_row(file, "/initial/psi_init", arma::vec(n_psi, arma::fill::zeros));

    if (noncentred)
    {
        write_row(file, "/priors/psi/omega_v", arma::vec(n_psi, arma::fill::value(0.005)));
    }
    else
    {
        write_row(file, "/priors/psi/shape", arma::vec(n_psi, arma::fill::value(3.0)));
        write_row(file, "/priors/psi/rate", arma::vec(n_psi, arma::fill::value(0.01)));
    }
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
void write_vec_stochvol(const ModelFile &file, bool noncentred = false)
{
    // The offset keeps log(u^2 + offset) finite when a residual lands on zero,
    // and bounds it well inside the range the ten-component mixture covers.
    write_row(file, "/priors/u_sigma/offset", arma::vec(kK, arma::fill::value(1e-4)));
    write_row(file, "/priors/u_sigma/sigma", arma::vec(kK, arma::fill::value(0.1)));
    if (noncentred)
    {
        write_row(file, "/priors/u_sigma/omega_v", arma::vec(kK, arma::fill::value(0.1)));
    }
    else
    {
        write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
        write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(0.2)));
    }
    write_row(file, "/priors/u_sigma/mu", arma::vec(kK, arma::fill::zeros));
    write_mat(file, "/priors/u_sigma/v_inv", arma::eye<arma::mat>(kK, kK));

    write_mat(file, "/initial/h", arma::mat(kTT, kK, arma::fill::zeros));
    write_row(file, "/initial/h_init", arma::vec(kK, arma::fill::zeros));
}

/// The independent gamma priors on the error precisions. `initial` is the
/// dataset the model reads its starting precision from: the constant-coefficient
/// VECs call it u_sigma_inv and the time-varying ones u_omega_inv.
void write_vec_gamma_errors(const ModelFile &file, const std::string &initial)
{
    write_row(file, "/priors/u_sigma/shape", arma::vec(kK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kK, arma::fill::value(2.0)));
    write_mat(file, initial, arma::eye<arma::mat>(kK, kK));
}

/// The non-SUR Koop, Leon-Gonzalez and Strachan (2010) sampler: the same priors
/// VecNormalWishart carries, and no selection block -- validate() rejects one.
void write_vec_klgs_2010(const ModelFile &file, arma::uword nparams)
{
    write_vec_constant_coefficients(file, "none", nparams, false);
    write_vec_constant_coint(file);

    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));
}

/// The same priors as VecKlgs2010 above, read from the SUR regressors rather
/// than the compact ones -- both layouts are built from one sample, so the two
/// models differ here only in what /model/algorithm names and in the selection
/// block this one accepts and that one refuses.
void write_vec_normal_wishart(const ModelFile &file, const std::string &varsel,
                              arma::uword nparams)
{
    write_vec_constant_coefficients(file, varsel, nparams, true);
    write_vec_constant_coint(file);

    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));
}

void write_vec_normal_gamma(const ModelFile &file, const std::string &varsel, bool covar,
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

void write_vec_normal_stochvol(const ModelFile &file, const std::string &varsel, bool covar,
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

void write_vec_tvp_wishart(const ModelFile &file, const std::string &varsel, arma::uword nparams,
                           bool noncentred = false)
{
    write_vec_tvp_coefficients(file, varsel, nparams, noncentred);
    write_vec_tvp_coint(file);

    // The error covariance is the Wishart precision alone: no psi block, so
    // nothing here depends on the covar flag.
    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
    write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(kK, kK));
}

/// `noncentred` switches the coefficients and the covariance block to the
/// normal prior on their signed standard deviation. The cointegration space
/// keeps its fixed unit state variance, and the error precision its gamma prior.
void write_vec_tvp_gamma(const ModelFile &file, const std::string &varsel, bool covar,
                         arma::uword nparams, bool noncentred = false)
{
    write_vec_tvp_coefficients(file, varsel, nparams, noncentred);
    write_vec_tvp_coint(file);
    write_vec_gamma_errors(file, "/initial/u_omega_inv");

    if (covar)
    {
        write_vec_tvp_psi(file, varsel, kK * (kK - 1) / 2, noncentred);
    }
}

/// `noncentred` switches the coefficients, the covariance block and the
/// log-volatility to the normal prior on their signed standard deviation. The
/// cointegration space keeps its fixed unit state variance.
void write_vec_tvp_stochvol(const ModelFile &file, const std::string &varsel, bool covar,
                            arma::uword nparams, bool noncentred = false)
{
    write_vec_tvp_coefficients(file, varsel, nparams, noncentred);
    write_vec_tvp_coint(file);
    write_vec_stochvol(file, noncentred);

    if (covar)
    {
        write_vec_tvp_psi(file, varsel, kK * (kK - 1) / 2, noncentred);
    }
}

//////////////////////////////////////////////////////////////////////////////
// The dynamic factor model.
//
// Neither the VAR block nor the VEC one: a DFM has no regressors at all, so it
// shares none of their data builders and writes no /data/train/z. It also needs
// more series than they carry -- a factor model with three of them is not one --
// so its dimensions are its own.
//
// Two factors and a transition of order two, deliberately. One factor makes the
// loading draw a sequence of scalars and the initial-state covariance the
// identity times V, so most of what is written for the general case would go
// unexercised; order two is what puts a second subdiagonal into the band and a
// second block into the prior over the first states.

constexpr int kDfmK = 6;  // observed series
constexpr int kDfmN = 2;  // factors
constexpr int kDfmP = 2;  // order of the factor transition
constexpr int kDfmNLambda = kDfmN * (2 * kDfmK - kDfmN - 1) / 2; // 9
constexpr int kDfmNA = kDfmN * kDfmN * kDfmP;                    // 8

/// A sample with a factor structure in it: two AR(1) factors, loadings that
/// satisfy the identifying restriction, and idiosyncratic noise. The sampler
/// has something to find, which makes the fingerprints worth reading.
arma::mat simulate_dfm(Lcg &rng)
{
    arma::mat factors(kDfmN, kTT, arma::fill::zeros);
    for (int t = 1; t < kTT; ++t)
    {
        for (int j = 0; j < kDfmN; ++j)
        {
            factors(j, t) = 0.6 * factors(j, t - 1) + rng.normal();
        }
    }

    arma::mat lambda(kDfmK, kDfmN, arma::fill::zeros);
    lambda.diag().ones();
    for (int i = 1; i < kDfmK; ++i)
    {
        for (int j = 0; j < std::min(i, kDfmN); ++j)
        {
            lambda(i, j) = 0.5 + 0.1 * static_cast<double>(i + j);
        }
    }

    arma::mat x = lambda * factors;
    for (int t = 0; t < kTT; ++t)
    {
        for (int i = 0; i < kDfmK; ++i)
        {
            x(i, t) += 0.3 * rng.normal();
        }
    }
    return x;
}

void write_dfm_normal_gamma(const ModelFile &file, int h, const arma::mat &x)
{
    write_attribute<std::string>(file, "/model", "algorithm", "DfmNormalGamma");
    write_attribute<int>(file, "/model", "k", kDfmK);
    write_attribute<int>(file, "/model", "p", kDfmP);
    write_attribute<int>(file, "/model", "n_factors", kDfmN);
    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", "none");
    write_attribute<bool>(file, "/model", "structural", false);
    write_attribute<std::string>(file, "/model", "error", "gamma");

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/lambda");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/priors/v_sigma");
    ensure_group(file, "/initial");

    // Stored already stacked, one row, the way the other models' response is.
    write_row(file, "/data/train/y", arma::vectorise(x));

    // The free loadings, in the row-major order the sampler draws them.
    write_row(file, "/priors/lambda/mu", arma::vec(kDfmNLambda, arma::fill::zeros));
    write_mat(file, "/priors/lambda/v_inv", arma::eye<arma::mat>(kDfmNLambda, kDfmNLambda) * 0.01);
    write_row(file, "/initial/lambda", arma::vec(kDfmNLambda, arma::fill::value(0.5)));

    write_row(file, "/priors/a/mu", arma::vec(kDfmNA, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(kDfmNA, kDfmNA) * 0.01);
    write_row(file, "/initial/a", arma::vec(kDfmNA, arma::fill::zeros));

    // Both precisions are diagonal and are written as the diagonal.
    write_row(file, "/priors/u_sigma/shape", arma::vec(kDfmK, arma::fill::value(5.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kDfmK, arma::fill::value(4.0)));
    write_row(file, "/initial/u_sigma_inv", arma::vec(kDfmK, arma::fill::ones));

    write_row(file, "/priors/v_sigma/shape", arma::vec(kDfmN, arma::fill::value(5.0)));
    write_row(file, "/priors/v_sigma/rate", arma::vec(kDfmN, arma::fill::value(4.0)));
    write_row(file, "/initial/v_sigma_inv", arma::vec(kDfmN, arma::fill::ones));

    // No /data/forecast: a DFM forecasts by running its transition on from the
    // last drawn factors, so the horizon is the whole of what it needs.
    if (h > 0)
    {
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// Deterministic terms of the factor fixtures written with --deterministic: a
/// constant, a linear trend scaled to one over the sample, and a dummy on every
/// fourth period -- one of each kind a caller is likely to pass. `first` is the
/// period the rows start at, so the horizon continues the sample's pattern.
constexpr int kFactorDetN = 3;

arma::mat fixture_deterministic(const int first, const int periods)
{
    arma::mat d(static_cast<arma::uword>(periods), kFactorDetN);
    for (int t = 0; t < periods; ++t)
    {
        const arma::uword row = static_cast<arma::uword>(t);
        d(row, 0) = 1.0;
        d(row, 1) = static_cast<double>(first + t + 1) / kTT;
        d(row, 2) = ((first + t) % 4 == 0) ? 1.0 : 0.0;
    }
    return d;
}

/// The deterministic part a fixture's series are shifted by, `rows` x n: a
/// level, a trend and a seasonal effect that differ from row to row.
arma::mat fixture_deterministic_effects(const arma::uword rows)
{
    arma::mat c(rows, kFactorDetN);
    for (arma::uword i = 0; i < rows; ++i)
    {
        c(i, 0) = 0.5 + 0.1 * static_cast<double>(i);
        c(i, 1) = 0.3;
        c(i, 2) = (i % 2 == 0) ? 0.4 : -0.4;
    }
    return c;
}

/// /model/n, the deterministic terms over the sample and the horizon, and the
/// prior and starting value of each coefficient block in `blocks`, given as
/// (name, rows) pairs.
void write_factor_deterministic(const ModelFile &file, const int h,
                                const std::vector<std::pair<std::string, int>> &blocks)
{
    write_attribute<int>(file, "/model", "n", kFactorDetN);
    write_mat(file, "/data/train/x", fixture_deterministic(0, kTT));
    if (h > 0)
    {
        ensure_group(file, "/data/forecast");
        write_mat(file, "/data/forecast/x", fixture_deterministic(kTT, h));
    }
    for (const auto &block : blocks)
    {
        const arma::uword width = static_cast<arma::uword>(block.second) * kFactorDetN;
        ensure_group(file, "/priors/" + block.first);
        write_row(file, "/priors/" + block.first + "/mu", arma::vec(width, arma::fill::zeros));
        write_mat(file, "/priors/" + block.first + "/v_inv",
                  arma::eye<arma::mat>(width, width) * 0.01);
        write_row(file, "/initial/" + block.first, arma::vec(width, arma::fill::zeros));
    }
}

/// Dimensions of the factor augmented VAR fixture.
///
/// The panel and the factors are the DFM's, so the two fixtures are comparable,
/// plus two observed factors. Two rather than one for the same reason there are
/// two unobserved ones: with a single observed factor the state's observed block
/// is a scalar, and the cross covariance in Q -- the thing this model exists to
/// estimate -- would be a single number rather than a block.
constexpr int kFavarNObs = 2;                                  // observed factors
constexpr int kFavarNState = kDfmN + kFavarNObs;               // 4
constexpr int kFavarNLambda = (kDfmK - kDfmN) * kFavarNState;  // 16
constexpr int kFavarNA = kFavarNState * kFavarNState * kDfmP;  // 32

/// A sample with a factor augmented structure in it: two AR(1) factors, two
/// observed factors that respond to them and are responded to, loadings that
/// satisfy the identifying restriction, and idiosyncratic noise.
///
/// The observed block is generated from the same joint transition the model
/// assumes rather than independently, so the cross terms are not zero in the
/// data -- a fixture whose observed factors were independent noise would run the
/// sampler through every line and leave the one block that distinguishes this
/// model from a dynamic factor model with nothing to find.
arma::mat simulate_favar(Lcg &rng, arma::mat &obs)
{
    arma::mat state(kFavarNState, kTT, arma::fill::zeros);
    for (int t = 1; t < kTT; ++t)
    {
        for (int j = 0; j < kFavarNState; ++j)
        {
            state(j, t) = 0.5 * state(j, t - 1) + rng.normal();
        }
        // The coupling: the observed block leans on the previous factors, and
        // the factors lean back.
        state(0, t) += 0.25 * state(kDfmN, t - 1);
        state(kDfmN, t) += 0.30 * state(0, t - 1);
    }

    // The identifying block is the identity, so the first kDfmN rows carry no
    // free loading at all; every row after them carries the whole width.
    arma::mat lambda(kDfmK, kFavarNState, arma::fill::zeros);
    for (int i = 0; i < kDfmN; ++i)
    {
        lambda(i, i) = 1.0;
    }
    for (int i = kDfmN; i < kDfmK; ++i)
    {
        for (int j = 0; j < kFavarNState; ++j)
        {
            lambda(i, j) = 0.5 + 0.1 * static_cast<double>(i + j);
        }
    }

    arma::mat x = lambda * state;
    for (int t = 0; t < kTT; ++t)
    {
        for (int i = 0; i < kDfmK; ++i)
        {
            x(i, t) += 0.3 * rng.normal();
        }
    }

    // The observed half of the state, tt x n_obs -- one period per row, the
    // layout the file stores and the reader expects.
    obs = arma::trans(state.rows(kDfmN, kFavarNState - 1));
    return x;
}

/// The tree is DfmNormalGamma's with two changes, and both follow from half of
/// the state being data: `/data/train/f_obs` beside the panel, and a Wishart
/// `df`/`scale` where the DFM has a gamma `shape`/`rate` on the factor
/// innovations. The idiosyncratic group is unchanged -- a factor model's
/// idiosyncratic precision is diagonal in every member of the family.
void write_favar_normal_wishart(const ModelFile &file, int h, const arma::mat &x,
                                const arma::mat &obs)
{
    write_attribute<std::string>(file, "/model", "algorithm", "FavarNormalWishart");
    write_attribute<int>(file, "/model", "k", kDfmK);
    write_attribute<int>(file, "/model", "p", kDfmP);
    write_attribute<int>(file, "/model", "n_factors", kDfmN);
    write_attribute<int>(file, "/model", "n_obs_factors", kFavarNObs);
    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", "none");
    write_attribute<bool>(file, "/model", "structural", false);
    write_attribute<std::string>(file, "/model", "error", "wishart");

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/lambda");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/priors/v_sigma");
    ensure_group(file, "/initial");

    // Stored already stacked, one row, the way the other models' response is.
    write_row(file, "/data/train/y", arma::vectorise(x));

    // The observed factors keep their rectangle: tt x n_obs, one period per row.
    write_mat(file, "/data/train/f_obs", obs);

    // The free loadings, in the row-major order the sampler draws them -- the
    // FAVAR count, wider than the DFM's by the observed columns of every row
    // from n_factors on.
    write_row(file, "/priors/lambda/mu", arma::vec(kFavarNLambda, arma::fill::zeros));
    write_mat(file, "/priors/lambda/v_inv",
              arma::eye<arma::mat>(kFavarNLambda, kFavarNLambda) * 0.01);
    write_row(file, "/initial/lambda", arma::vec(kFavarNLambda, arma::fill::value(0.5)));

    write_row(file, "/priors/a/mu", arma::vec(kFavarNA, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(kFavarNA, kFavarNA) * 0.01);
    write_row(file, "/initial/a", arma::vec(kFavarNA, arma::fill::zeros));

    // Diagonal, and written as the diagonal.
    write_row(file, "/priors/u_sigma/shape", arma::vec(kDfmK, arma::fill::value(5.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kDfmK, arma::fill::value(4.0)));
    write_row(file, "/initial/u_sigma_inv", arma::vec(kDfmK, arma::fill::ones));

    // A matrix, and written as a matrix. This pair is the one a file written
    // against a dynamic factor model gets wrong.
    write_int_scalar(file, "/priors/v_sigma/df", kFavarNState + 1);
    write_mat(file, "/priors/v_sigma/scale", arma::eye<arma::mat>(kFavarNState, kFavarNState));
    write_mat(file, "/initial/v_sigma_inv", arma::eye<arma::mat>(kFavarNState, kFavarNState));

    // No /data/forecast: the transition is run on from the last states, so the
    // horizon is the whole of what it needs.
    if (h > 0)
    {
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// The same model with the two gamma priors replaced by two stochastic
/// volatility blocks.
///
/// The tree is DfmNormalGamma's but for the error groups, which now carry what
/// VarNormalStochvol's `/priors/u_sigma` carries -- and carry it twice, once at
/// the width of the observed series and once at the width of the factors. That
/// pair of widths is the point of having this fixture: it is the one thing the
/// in-memory unit test cannot check, because the file is where a host has to get
/// them right.
void write_dfm_normal_stochvol(const ModelFile &file, int h, const arma::mat &x)
{
    write_attribute<std::string>(file, "/model", "algorithm", "DfmNormalStochvol");
    write_attribute<int>(file, "/model", "k", kDfmK);
    write_attribute<int>(file, "/model", "p", kDfmP);
    write_attribute<int>(file, "/model", "n_factors", kDfmN);
    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", "none");
    write_attribute<bool>(file, "/model", "structural", false);
    write_attribute<std::string>(file, "/model", "error", "sv");

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/lambda");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/priors/v_sigma");
    ensure_group(file, "/initial");

    // Stored already stacked, one row, the way the other models' response is.
    write_row(file, "/data/train/y", arma::vectorise(x));

    // The free loadings, in the row-major order the sampler draws them.
    write_row(file, "/priors/lambda/mu", arma::vec(kDfmNLambda, arma::fill::zeros));
    write_mat(file, "/priors/lambda/v_inv", arma::eye<arma::mat>(kDfmNLambda, kDfmNLambda) * 0.01);
    write_row(file, "/initial/lambda", arma::vec(kDfmNLambda, arma::fill::value(0.5)));

    write_row(file, "/priors/a/mu", arma::vec(kDfmNA, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(kDfmNA, kDfmNA) * 0.01);
    write_row(file, "/initial/a", arma::vec(kDfmNA, arma::fill::zeros));

    // The idiosyncratic volatility, one log-volatility per observed series. The
    // offset keeps log(u^2 + offset) finite when a residual lands on zero, and
    // `sigma` is the starting value of the innovation variance rather than a
    // prior -- see dfm_normal_stochvol_io.h.
    write_row(file, "/priors/u_sigma/offset", arma::vec(kDfmK, arma::fill::value(1e-4)));
    write_row(file, "/priors/u_sigma/sigma", arma::vec(kDfmK, arma::fill::value(0.1)));
    write_row(file, "/priors/u_sigma/shape", arma::vec(kDfmK, arma::fill::value(3.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kDfmK, arma::fill::value(0.2)));
    write_row(file, "/priors/u_sigma/mu", arma::vec(kDfmK, arma::fill::zeros));
    write_mat(file, "/priors/u_sigma/v_inv", arma::eye<arma::mat>(kDfmK, kDfmK));

    write_mat(file, "/initial/u_h", arma::mat(kTT, kDfmK, arma::fill::zeros));
    write_row(file, "/initial/u_h_init", arma::vec(kDfmK, arma::fill::zeros));

    // The factor innovations', at the width of the factors and not of the series.
    write_row(file, "/priors/v_sigma/offset", arma::vec(kDfmN, arma::fill::value(1e-4)));
    write_row(file, "/priors/v_sigma/sigma", arma::vec(kDfmN, arma::fill::value(0.1)));
    write_row(file, "/priors/v_sigma/shape", arma::vec(kDfmN, arma::fill::value(3.0)));
    write_row(file, "/priors/v_sigma/rate", arma::vec(kDfmN, arma::fill::value(0.2)));
    write_row(file, "/priors/v_sigma/mu", arma::vec(kDfmN, arma::fill::zeros));
    write_mat(file, "/priors/v_sigma/v_inv", arma::eye<arma::mat>(kDfmN, kDfmN));

    write_mat(file, "/initial/v_h", arma::mat(kTT, kDfmN, arma::fill::zeros));
    write_row(file, "/initial/v_h_init", arma::vec(kDfmN, arma::fill::zeros));

    // No /data/forecast: a DFM forecasts by running its transition on from the
    // last drawn factors, so the horizon is the whole of what it needs.
    if (h > 0)
    {
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// The same model with the two normal priors on the coefficients replaced by
/// two random walks.
///
/// The tree is DfmNormalGamma's but for the two coefficient groups, which now
/// carry what VarTvpGamma's `/priors/a` carries -- an inverse gamma on the state
/// innovations beside a normal on the state before the sample -- and a path
/// rather than a vector as the starting value. That the same group holds both
/// priors is the thing a file is most likely to get wrong here, which is why the
/// fixture writes it rather than only the in-memory unit test covering it.
void write_dfm_tvp_gamma(const ModelFile &file, int h, const arma::mat &x, bool noncentred = false)
{
    write_attribute<std::string>(file, "/model", "algorithm", "DfmTvpGamma");
    write_attribute<int>(file, "/model", "k", kDfmK);
    write_attribute<int>(file, "/model", "p", kDfmP);
    write_attribute<int>(file, "/model", "n_factors", kDfmN);
    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", "none");
    write_attribute<bool>(file, "/model", "structural", false);
    write_attribute<std::string>(file, "/model", "error", "gamma");

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/lambda");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/priors/v_sigma");
    ensure_group(file, "/initial");

    // Stored already stacked, one row, the way the other models' response is.
    write_row(file, "/data/train/y", arma::vectorise(x));

    // The loading paths. The state innovation variances are stored inverted, as
    // in every other time-varying model here, and a tight prior keeps the random
    // walk from wandering far enough to make the smoother ill-conditioned. The
    // path starts where DfmNormalGamma's single vector of loadings does.
    write_mat(file, "/initial/lambda", arma::mat(kDfmNLambda, kTT, arma::fill::value(0.5)));
    write_mat(file, "/initial/lambda_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(kDfmNLambda, arma::fill::value(100.0)))));
    write_row(file, "/initial/lambda_init", arma::vec(kDfmNLambda, arma::fill::value(0.5)));

    if (noncentred)
    {
        write_row(file, "/priors/lambda/omega_v", arma::vec(kDfmNLambda, arma::fill::value(0.005)));
    }
    else
    {
        write_row(file, "/priors/lambda/shape", arma::vec(kDfmNLambda, arma::fill::value(3.0)));
        write_row(file, "/priors/lambda/rate", arma::vec(kDfmNLambda, arma::fill::value(0.01)));
    }
    write_row(file, "/priors/lambda/mu", arma::vec(kDfmNLambda, arma::fill::zeros));
    write_mat(file, "/priors/lambda/v_inv", arma::eye<arma::mat>(kDfmNLambda, kDfmNLambda));

    // The transition path, at the width of vec([A_1 .. A_p]) and not of the
    // loadings -- the pair of widths a file is most likely to get the wrong way
    // round.
    write_mat(file, "/initial/a", arma::mat(kDfmNA, kTT, arma::fill::zeros));
    write_mat(file, "/initial/a_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(kDfmNA, arma::fill::value(100.0)))));
    write_row(file, "/initial/a_init", arma::vec(kDfmNA, arma::fill::zeros));

    if (noncentred)
    {
        write_row(file, "/priors/a/omega_v", arma::vec(kDfmNA, arma::fill::value(0.005)));
    }
    else
    {
        write_row(file, "/priors/a/shape", arma::vec(kDfmNA, arma::fill::value(3.0)));
        write_row(file, "/priors/a/rate", arma::vec(kDfmNA, arma::fill::value(0.01)));
    }
    write_row(file, "/priors/a/mu", arma::vec(kDfmNA, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(kDfmNA, kDfmNA));

    // Both precisions are diagonal and are written as the diagonal. They do not
    // move in this model: it is the coefficients that do.
    write_row(file, "/priors/u_sigma/shape", arma::vec(kDfmK, arma::fill::value(5.0)));
    write_row(file, "/priors/u_sigma/rate", arma::vec(kDfmK, arma::fill::value(4.0)));
    write_row(file, "/initial/u_sigma_inv", arma::vec(kDfmK, arma::fill::ones));

    write_row(file, "/priors/v_sigma/shape", arma::vec(kDfmN, arma::fill::value(5.0)));
    write_row(file, "/priors/v_sigma/rate", arma::vec(kDfmN, arma::fill::value(4.0)));
    write_row(file, "/initial/v_sigma_inv", arma::vec(kDfmN, arma::fill::ones));

    // No /data/forecast: a DFM forecasts by running its transition on from the
    // last drawn factors, so the horizon is the whole of what it needs.
    if (h > 0)
    {
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// The widest of the four: DfmTvpGamma's coefficients over DfmNormalStochvol's
/// errors.
///
/// Its tree is the union of theirs and holds nothing neither has, which is what
/// makes it worth generating rather than only covering in memory: `shape` and
/// `rate` now appear in all four prior groups, at three different widths, and a
/// reader that fetched one group's pair for another would produce a well-formed
/// model of the wrong shape.
void write_dfm_tvp_stochvol(const ModelFile &file, int h, const arma::mat &x, bool noncentred = false)
{
    write_attribute<std::string>(file, "/model", "algorithm", "DfmTvpStochvol");
    write_attribute<int>(file, "/model", "k", kDfmK);
    write_attribute<int>(file, "/model", "p", kDfmP);
    write_attribute<int>(file, "/model", "n_factors", kDfmN);
    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "varsel", "none");
    write_attribute<bool>(file, "/model", "structural", false);
    write_attribute<std::string>(file, "/model", "error", "sv");

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/priors/lambda");
    ensure_group(file, "/priors/a");
    ensure_group(file, "/priors/u_sigma");
    ensure_group(file, "/priors/v_sigma");
    ensure_group(file, "/initial");

    // Stored already stacked, one row, the way the other models' response is.
    write_row(file, "/data/train/y", arma::vectorise(x));

    // The two coefficient paths, exactly as DfmTvpGamma writes them.
    write_mat(file, "/initial/lambda", arma::mat(kDfmNLambda, kTT, arma::fill::value(0.5)));
    write_mat(file, "/initial/lambda_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(kDfmNLambda, arma::fill::value(100.0)))));
    write_row(file, "/initial/lambda_init", arma::vec(kDfmNLambda, arma::fill::value(0.5)));

    if (noncentred)
    {
        write_row(file, "/priors/lambda/omega_v", arma::vec(kDfmNLambda, arma::fill::value(0.005)));
    }
    else
    {
        write_row(file, "/priors/lambda/shape", arma::vec(kDfmNLambda, arma::fill::value(3.0)));
        write_row(file, "/priors/lambda/rate", arma::vec(kDfmNLambda, arma::fill::value(0.01)));
    }
    write_row(file, "/priors/lambda/mu", arma::vec(kDfmNLambda, arma::fill::zeros));
    write_mat(file, "/priors/lambda/v_inv", arma::eye<arma::mat>(kDfmNLambda, kDfmNLambda));

    write_mat(file, "/initial/a", arma::mat(kDfmNA, kTT, arma::fill::zeros));
    write_mat(file, "/initial/a_sigma_inv",
              arma::mat(arma::diagmat(arma::vec(kDfmNA, arma::fill::value(100.0)))));
    write_row(file, "/initial/a_init", arma::vec(kDfmNA, arma::fill::zeros));

    if (noncentred)
    {
        write_row(file, "/priors/a/omega_v", arma::vec(kDfmNA, arma::fill::value(0.005)));
    }
    else
    {
        write_row(file, "/priors/a/shape", arma::vec(kDfmNA, arma::fill::value(3.0)));
        write_row(file, "/priors/a/rate", arma::vec(kDfmNA, arma::fill::value(0.01)));
    }
    write_row(file, "/priors/a/mu", arma::vec(kDfmNA, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(kDfmNA, kDfmNA));

    // And the two volatility blocks, exactly as DfmNormalStochvol writes them:
    // one at the width of the observed series, one at the width of the factors.
    write_row(file, "/priors/u_sigma/offset", arma::vec(kDfmK, arma::fill::value(1e-4)));
    write_row(file, "/priors/u_sigma/sigma", arma::vec(kDfmK, arma::fill::value(0.1)));
    if (noncentred)
    {
        write_row(file, "/priors/u_sigma/omega_v", arma::vec(kDfmK, arma::fill::value(0.1)));
    }
    else
    {
        write_row(file, "/priors/u_sigma/shape", arma::vec(kDfmK, arma::fill::value(3.0)));
        write_row(file, "/priors/u_sigma/rate", arma::vec(kDfmK, arma::fill::value(0.2)));
    }
    write_row(file, "/priors/u_sigma/mu", arma::vec(kDfmK, arma::fill::zeros));
    write_mat(file, "/priors/u_sigma/v_inv", arma::eye<arma::mat>(kDfmK, kDfmK));

    write_mat(file, "/initial/u_h", arma::mat(kTT, kDfmK, arma::fill::zeros));
    write_row(file, "/initial/u_h_init", arma::vec(kDfmK, arma::fill::zeros));

    write_row(file, "/priors/v_sigma/offset", arma::vec(kDfmN, arma::fill::value(1e-4)));
    write_row(file, "/priors/v_sigma/sigma", arma::vec(kDfmN, arma::fill::value(0.1)));
    if (noncentred)
    {
        write_row(file, "/priors/v_sigma/omega_v", arma::vec(kDfmN, arma::fill::value(0.1)));
    }
    else
    {
        write_row(file, "/priors/v_sigma/shape", arma::vec(kDfmN, arma::fill::value(3.0)));
        write_row(file, "/priors/v_sigma/rate", arma::vec(kDfmN, arma::fill::value(0.2)));
    }
    write_row(file, "/priors/v_sigma/mu", arma::vec(kDfmN, arma::fill::zeros));
    write_mat(file, "/priors/v_sigma/v_inv", arma::eye<arma::mat>(kDfmN, kDfmN));

    write_mat(file, "/initial/v_h", arma::mat(kTT, kDfmN, arma::fill::zeros));
    write_row(file, "/initial/v_h_init", arma::vec(kDfmN, arma::fill::zeros));

    // No /data/forecast: a DFM forecasts by running its transition on from the
    // last drawn factors, so the horizon is the whole of what it needs.
    if (h > 0)
    {
        write_attribute<int>(file, "/model", "h", h);
    }
}

/// The discounted VEC: the matrix normal prior of write_var_tvp_discount(),
/// over a design that is `rank` error correction columns wider than the compact
/// regressors, and the cointegration matrix the run conditions on.
///
/// `/initial/beta` is where every VEC here keeps its space and in the same
/// layout, and this model reads it from there -- not as a starting value, of
/// which it has none, but as the space it holds fixed. A file that differs from
/// a VecNormalWishart one in `/model/algorithm`, the two discounts and the shape
/// of the coefficient prior is otherwise the same file.
void write_vec_tvp_discount(const ModelFile &file)
{
    // rank error correction columns in front of the k(p-1) lagged differences.
    const arma::uword n_design = static_cast<arma::uword>(kVecRank + kK * (kVecP - 1));

    write_mat(file, "/priors/a/mean", arma::mat(n_design, kK, arma::fill::zeros));
    write_mat(file, "/priors/a/cov", arma::eye<arma::mat>(n_design, n_design));

    write_row(file, "/initial/beta", vec_initial_beta());

    write_int_scalar(file, "/priors/u_sigma/df", kK);
    write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(kK, kK));
}

/// Dispatches to the seven above. Returns false if the name is not a VEC.
bool write_vec_model(const ModelFile &file, const std::string &model, const std::string &varsel,
                     bool covar, arma::uword nparams, bool noncentred)
{
    if (model == "VecKlgs2010")
    {
        write_vec_klgs_2010(file, nparams);
    }
    else if (model == "VecNormalWishart")
    {
        write_vec_normal_wishart(file, varsel, nparams);
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
        write_vec_tvp_wishart(file, varsel, nparams, noncentred);
    }
    else if (model == "VecTvpGamma")
    {
        write_vec_tvp_gamma(file, varsel, covar, nparams, noncentred);
    }
    else if (model == "VecTvpStochvol")
    {
        write_vec_tvp_stochvol(file, varsel, covar, nparams, noncentred);
    }
    else if (model == "VecTvpDiscount")
    {
        write_vec_tvp_discount(file);
    }
    else
    {
        return false;
    }
    return true;
}

bool is_vec_model(const std::string &model)
{
    return model == "VecKlgs2010" || model == "VecNormalWishart" || model == "VecNormalGamma" ||
           model == "VecNormalStochvol" || model == "VecTvpWishart" || model == "VecTvpGamma" ||
           model == "VecTvpStochvol" || model == "VecTvpDiscount";
}

/// Dimensions of the unobserved components fixtures: two series, each a trend
/// plus a cycle, and a cycle of order two, so that both the cross lags of a VAR
/// cycle and a band wider than the trend's own are exercised.
constexpr int kUcK = 2;
constexpr int kUcP = 2;

/// A sample with trends and cycles in it: random walks with drift -- or, for a
/// smooth trend, an integrated random walk -- under a VAR(2) cycle with a little
/// cross dependence, which the own-lags fixture then ignores.
arma::mat simulate_uc(Lcg &rng, const bool smooth, const bool with_break)
{
    arma::mat tau(kUcK, kTT), cycle(kUcK, kTT, arma::fill::zeros);
    arma::vec last = {10.0, 5.0};
    arma::vec slope = {0.4, 0.2};
    for (int t = 0; t < kTT; ++t)
    {
        const arma::uword col = static_cast<arma::uword>(t);
        for (arma::uword i = 0; i < kUcK; ++i)
        {
            double c = 0.6 * rng.normal();
            if (t >= 1)
            {
                c += 0.9 * cycle(i, col - 1) + 0.1 * cycle(1 - i, col - 1);
            }
            if (t >= 2)
            {
                c -= 0.3 * cycle(i, col - 2);
            }
            cycle(i, col) = c;

            if (smooth)
            {
                slope(i) += 0.05 * rng.normal();
                last(i) += slope(i);
            }
            else
            {
                const double drift = (with_break && t >= kTT / 2) ? 0.1 : 0.5;
                last(i) += drift + 0.3 * rng.normal();
            }
            tau(i, col) = last(i);
        }
    }
    return tau + cycle;
}

/// The drift's deterministic terms over `periods` from `first`: a constant, and
/// with a break a step dummy from the middle of the sample on.
arma::mat uc_deterministic(const int first, const int periods, const int n)
{
    arma::mat d(static_cast<arma::uword>(periods), static_cast<arma::uword>(n), arma::fill::ones);
    if (n > 1)
    {
        for (int t = 0; t < periods; ++t)
        {
            d(static_cast<arma::uword>(t), 1) = (first + t) >= kTT / 2 ? 1.0 : 0.0;
        }
    }
    return d;
}

/// What a UC fixture switches on beyond the model's name.
struct UcOptions
{
    std::string model;
    int h = 0;
    bool covar = false;
    bool smooth = false;
    bool own_cycle = false;
    bool with_break = false;
    bool stationary = false;
    bool stationary_init = false; ///< /model/cycle_init stationary, which needs stationary.
    bool damped = false;          ///< /model/trend damped.
    bool signals = false;         ///< Two signal equations, one per series.
    bool irregular = false;
    bool noncentred = false; ///< UcNormalStochvol: omega_v on the log-volatilities.
    bool ald = false;        ///< UcNormalSkew: the asymmetric Laplace.
    bool skew_trend = false; ///< UcNormalSkew: the trend shock skewed.
    bool skew_index = false; ///< UcNormalSkew: one index variable in the scales.
};

/// The skew index of the --skew-index fixtures over `periods` from `first`: a
/// slow wave, centred, the shape an uncertainty index has.
arma::mat uc_skew_index(const int first, const int periods)
{
    arma::mat z(static_cast<arma::uword>(periods), 1);
    for (int t = 0; t < periods; ++t)
    {
        z(static_cast<arma::uword>(t), 0) = 0.5 * std::sin(static_cast<double>(first + t) / 3.0);
    }
    return z;
}

void write_uc_model(const ModelFile &file, const arma::mat &y, const UcOptions &o)
{
    const int n = o.smooth ? 0 : (o.with_break ? 2 : 1);
    const int order = (o.smooth || o.damped) ? 2 : 1;
    const int n_a = o.own_cycle ? kUcK * kUcP : kUcK * kUcK * kUcP;
    const int shocks = 2 * kUcK;
    const int n_psi = kUcK * (2 * kUcK - 1);
    const bool gamma = o.model == "UcNormalGamma";
    const bool wishart = o.model == "UcNormalWishart";
    const bool stochvol = o.model == "UcNormalStochvol";
    const bool skew = o.model == "UcNormalSkew";

    std::string error = "gamma";
    if (wishart)
    {
        error = "wishart";
    }
    else if (stochvol)
    {
        error = o.covar ? "sv+covar" : "sv";
    }
    else if (skew)
    {
        error = "skew";
    }
    else if (o.covar)
    {
        error = "gamma+covar";
    }

    write_attribute<std::string>(file, "/model", "algorithm", o.model);
    write_attribute<int>(file, "/model", "k", kUcK);
    write_attribute<int>(file, "/model", "p", kUcP);
    write_attribute<int>(file, "/model", "n", n);
    write_attribute<int>(file, "/model", "iterations", kIterations);
    write_attribute<int>(file, "/model", "burnin", kBurnin);
    write_attribute<std::string>(file, "/model", "error", error);
    write_attribute<std::string>(file, "/model", "trend",
                                 o.smooth ? "smooth" : (o.damped ? "damped" : "rw"));
    write_attribute<std::string>(file, "/model", "cycle", o.own_cycle ? "own" : "var");
    if (o.stationary || o.stationary_init)
    {
        write_attribute<bool>(file, "/model", "stationary", true);
    }
    if (o.stationary_init)
    {
        write_attribute<std::string>(file, "/model", "cycle_init", "stationary");
    }
    if (o.irregular)
    {
        write_attribute<bool>(file, "/model", "irregular", true);
    }
    if (skew)
    {
        write_attribute<std::string>(file, "/model", "skew", o.ald ? "ald" : "snd");
        write_attribute<std::string>(file, "/model", "skew_shock", o.skew_trend ? "trend" : "cycle");
    }
    if (o.h > 0)
    {
        write_attribute<int>(file, "/model", "h", o.h);
    }

    ensure_group(file, "/data");
    ensure_group(file, "/data/train");
    ensure_group(file, "/priors");
    ensure_group(file, "/initial");

    // One period per row, one series per column.
    write_mat(file, "/data/train/y", arma::trans(y));
    if (n > 0)
    {
        write_mat(file, "/data/train/x", uc_deterministic(0, kTT, n));
        if (o.h > 0)
        {
            ensure_group(file, "/data/forecast");
            write_mat(file, "/data/forecast/x", uc_deterministic(kTT, o.h, n));
        }
        ensure_group(file, "/priors/c");
        const arma::uword n_c = static_cast<arma::uword>(kUcK * n);
        write_row(file, "/priors/c/mu", arma::vec(n_c, arma::fill::zeros));
        write_mat(file, "/priors/c/v_inv", arma::eye<arma::mat>(n_c, n_c) * 0.1);
        write_row(file, "/initial/c", arma::vec(n_c, arma::fill::zeros));
    }

    ensure_group(file, "/priors/a");
    const arma::uword na = static_cast<arma::uword>(n_a);
    write_row(file, "/priors/a/mu", arma::vec(na, arma::fill::zeros));
    write_mat(file, "/priors/a/v_inv", arma::eye<arma::mat>(na, na) * 0.5);
    write_row(file, "/initial/a", arma::vec(na, arma::fill::zeros));

    // The trend before the sample, centred on the first observation of each
    // series with a standard deviation of ten.
    ensure_group(file, "/priors/tau0");
    const arma::vec first = y.col(0);
    const arma::vec tau0 = order == 2 ? arma::vec(arma::join_vert(first, first)) : first;
    const arma::uword nt = tau0.n_elem;
    write_row(file, "/priors/tau0/mu", tau0);
    write_mat(file, "/priors/tau0/v_inv", arma::eye<arma::mat>(nt, nt) * 0.01);
    write_row(file, "/initial/tau0", tau0);

    const arma::uword np = static_cast<arma::uword>(n_psi);
    if (o.covar && (gamma || stochvol))
    {
        ensure_group(file, "/priors/psi");
        write_row(file, "/priors/psi/mu", arma::vec(np, arma::fill::zeros));
        write_mat(file, "/priors/psi/v_inv", arma::eye<arma::mat>(np, np));
        write_row(file, "/initial/psi", arma::vec(np, arma::fill::zeros));
    }

    ensure_group(file, "/priors/u_sigma");
    const arma::uword ns = static_cast<arma::uword>(shocks);
    const arma::uword kk = static_cast<arma::uword>(kUcK);
    if (gamma)
    {
        write_row(file, "/priors/u_sigma/shape", arma::vec(ns, arma::fill::value(3.0)));
        write_row(file, "/priors/u_sigma/rate", arma::vec(ns, arma::fill::value(1.0)));
        write_row(file, "/initial/u_omega_inv", arma::vec(ns, arma::fill::ones));
    }
    else if (wishart)
    {
        // df two above the dimension, so the prior has a mean; a scale of half
        // the identity puts that mean at a variance of one quarter.
        write_int_scalar(file, "/priors/u_sigma/df", shocks + 2);
        write_mat(file, "/priors/u_sigma/scale", arma::eye<arma::mat>(ns, ns) * 0.5);
        write_mat(file, "/initial/u_sigma_inv", arma::eye<arma::mat>(ns, ns));
    }
    else if (stochvol)
    {
        write_row(file, "/priors/u_sigma/offset", arma::vec(ns, arma::fill::value(1e-4)));
        write_row(file, "/priors/u_sigma/sigma", arma::vec(ns, arma::fill::value(0.1)));
        if (o.noncentred)
        {
            write_row(file, "/priors/u_sigma/omega_v", arma::vec(ns, arma::fill::value(0.1)));
        }
        else
        {
            write_row(file, "/priors/u_sigma/shape", arma::vec(ns, arma::fill::value(3.0)));
            write_row(file, "/priors/u_sigma/rate", arma::vec(ns, arma::fill::value(0.2)));
        }
        write_row(file, "/priors/u_sigma/mu", arma::vec(ns, arma::fill::zeros));
        write_mat(file, "/priors/u_sigma/v_inv", arma::eye<arma::mat>(ns, ns));
        write_mat(file, "/initial/h", arma::mat(kTT, ns, arma::fill::zeros));
        write_row(file, "/initial/h_init", arma::vec(ns, arma::fill::zeros));
    }
    else if (skew)
    {
        // The shock that is not skewed: one gamma prior per series.
        write_row(file, "/priors/u_sigma/shape", arma::vec(kk, arma::fill::value(3.0)));
        write_row(file, "/priors/u_sigma/rate", arma::vec(kk, arma::fill::value(1.0)));
        write_row(file, "/initial/u_omega_inv", arma::vec(kk, arma::fill::ones));

        // The scale coefficients, per series the two log scales each with a
        // constant and the index's slope: Huang, Luo and Wang's priors, a
        // standard deviation of about 0.8 on the constants around log 0.3 and
        // of 3 on the slopes.
        const arma::uword width = o.skew_index ? 2 : 1;
        const arma::uword n_skew = kk * 2 * width;
        arma::vec mu(n_skew, arma::fill::zeros);
        arma::vec var(n_skew, arma::fill::value(9.0));
        for (arma::uword i = 0; i < kk; i++)
        {
            for (arma::uword j = 0; j < 2; j++)
            {
                const arma::uword at = (i * 2 + j) * width;
                mu(at) = std::log(0.3);
                var(at) = 0.7;
            }
        }
        ensure_group(file, "/priors/skew");
        write_row(file, "/priors/skew/mu", mu);
        write_mat(file, "/priors/skew/v_inv", arma::diagmat(1.0 / var));
        write_row(file, "/initial/skew", mu);
        if (o.skew_index)
        {
            write_mat(file, "/data/train/skew_x", uc_skew_index(0, kTT));
            if (o.h > 0)
            {
                ensure_group(file, "/data/forecast");
                write_mat(file, "/data/forecast/skew_x", uc_skew_index(kTT, o.h));
            }
        }
    }

    if (o.irregular)
    {
        ensure_group(file, "/priors/irregular");
        write_row(file, "/priors/irregular/shape", arma::vec(kk, arma::fill::value(3.0)));
        write_row(file, "/priors/irregular/rate", arma::vec(kk, arma::fill::value(0.5)));
        write_row(file, "/initial/irregular_inv", arma::vec(kk, arma::fill::value(4.0)));
    }

    // The damped trend's slope coefficients: a prior around one half with a
    // standard deviation of about 0.3, inside the interval it is truncated to.
    if (o.damped)
    {
        ensure_group(file, "/priors/rho");
        write_row(file, "/priors/rho/mu", arma::vec(kk, arma::fill::value(0.5)));
        write_mat(file, "/priors/rho/v_inv", arma::eye<arma::mat>(kk, kk) * 10.0);
        write_row(file, "/initial/rho", arma::vec(kk, arma::fill::value(0.5)));
    }

    // Two signals, as the European Commission hangs a capacity utilisation
    // indicator on the TFP cycle and a Phillips curve on the unemployment gap:
    // each series' deviation from the line through its first and last value,
    // scaled and shifted, plus noise from a generator of its own -- so the flag
    // moves no number already there. Each loads on a constant and on both
    // cycles at lags zero and one.
    if (o.signals)
    {
        const arma::uword m = kk;
        const arma::uword lags = 1;
        const arma::uword n_z = 1 + kk * (lags + 1);
        write_attribute<int>(file, "/model", "n_signals", static_cast<int>(m));
        write_attribute<int>(file, "/model", "signal_lags", static_cast<int>(lags));
        write_attribute<int>(file, "/model", "n_signal_x", 1);

        Lcg rng(20261010ULL);
        arma::mat s(kTT, m);
        for (arma::uword i = 0; i < m; ++i)
        {
            const double a = y(i, 0);
            const double b = (y(i, kTT - 1) - y(i, 0)) / static_cast<double>(kTT - 1);
            for (int t = 0; t < kTT; ++t)
            {
                const arma::uword col = static_cast<arma::uword>(t);
                s(col, i) = 1.0 + 0.8 * (y(i, col) - a - b * t) + 0.3 * rng.normal();
            }
        }
        write_mat(file, "/data/train/signal", s);
        write_mat(file, "/data/train/signal_x", arma::mat(kTT, 1, arma::fill::ones));

        ensure_group(file, "/priors/signal");
        write_row(file, "/priors/signal/mu", arma::vec(m * n_z, arma::fill::zeros));
        write_mat(file, "/priors/signal/v_inv", arma::eye<arma::mat>(m * n_z, m * n_z) * 0.1);
        write_row(file, "/initial/signal", arma::vec(m * n_z, arma::fill::zeros));
        ensure_group(file, "/priors/signal_noise");
        write_row(file, "/priors/signal_noise/shape", arma::vec(m, arma::fill::value(3.0)));
        write_row(file, "/priors/signal_noise/rate", arma::vec(m, arma::fill::value(0.5)));
        write_row(file, "/initial/signal_noise_inv", arma::vec(m, arma::fill::ones));
    }
}

/// What the horizon realised, for `--score`: the series carried on from its
/// last value by a drift, with noise. From a generator of its own, as
/// write_test_observations() is, so the flag moves no number already there --
/// and not that function itself, whose pull towards zero would score a
/// trending series against a collapse.
void write_uc_test_observations(const ModelFile &file, const arma::vec &last, const int h)
{
    Lcg rng(20260920ULL);
    arma::mat realised(static_cast<arma::uword>(h), last.n_elem);
    arma::vec previous = last;
    for (int t = 0; t < h; ++t)
    {
        for (arma::uword i = 0; i < previous.n_elem; ++i)
        {
            previous(i) += 0.3 + 0.5 * rng.normal();
        }
        realised.row(static_cast<arma::uword>(t)) = arma::trans(previous);
    }
    ensure_group(file, "/data/test");
    write_mat(file, "/data/test/y", realised);
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <dest.h5> <model> <none|ssvs|bvs> <covar 0|1> <structural 0|1> <h>"
                     " [group] [append] [--coint-rho] [--coint-p-tau] [--hold-states] [--noncentred]"
                     " [--score]\n";
        return 2;
    }

    const std::filesystem::path dest = argv[1];
    const std::string model = argv[2];
    const std::string varsel = argv[3];
    const bool covar = std::string(argv[4]) != "0";
    const bool structural = std::string(argv[5]) != "0";
    const int h = std::stoi(argv[6]);

    // What follows the six required arguments, in the order the two bare words
    // have always come in: the group, then `append`. A word starting with two
    // dashes is a flag instead and may sit anywhere among them, so a fixture
    // that wants one does not have to name a group it does not use.
    std::string group_argument;
    bool append = false;
    bool coint_rho = false;
    bool coint_p_tau = false;
    bool hold_states = false;
    bool score = false;
    bool noncentred = false;
    bool deterministic = false;
    bool iid = false;
    bool mixed = false;
    bool missing = false;
    bool partial_test = false;
    std::string shrinkage;
    bool stationary = false;
    bool draw_theta = false;
    bool quantile_grid = false;
    bool steady_state = false;
    bool volatility_break = false;
    bool soft = false;
    bool scenario = false;
    bool smooth = false;
    bool own_cycle = false;
    bool stationary_init = false;
    bool uc_damped = false;
    bool uc_signals = false;
    bool uc_irregular = false;
    bool uc_ald = false;
    bool skew_trend = false;
    bool skew_index = false;
    bool bare_group_seen = false;

    for (int i = 7; i < argc; i++)
    {
        const std::string token = argv[i];

        if (token == "--coint-rho")
        {
            coint_rho = true;
        }
        else if (token == "--hold-states")
        {
            // /model/forecast_states = "hold": the forecast from the last
            // in-sample states that every row forecast before simulating them
            // forward became the default.
            hold_states = true;
        }
        else if (token == "--coint-p-tau")
        {
            coint_p_tau = true;
        }
        else if (token == "--score")
        {
            // /data/test/y: what the horizon realised. It is what turns
            // `bayests forecasts` into a scoring run, so it is what puts a
            // fixture through the predictive density code at all.
            score = true;
        }
        else if (token == "--mixed")
        {
            // A quarterly series among monthly ones: /data/train/constraints.
            mixed = true;
        }
        else if (token.rfind("--shrinkage=", 0) == 0)
        {
            // An adaptive prior on `a`: /model/shrinkage and /priors/a/shrinkage.
            shrinkage = token.substr(12);
        }
        else if (token == "--volatility-break")
        {
            // A known break in the volatility: /model/volatility_break.
            volatility_break = true;
        }
        else if (token == "--steady-state")
        {
            // The prior on the unconditional mean: /model/steady_state.
            steady_state = true;
        }
        else if (token == "--quantile-grid")
        {
            // A grid of quantiles: /model/quantiles.
            quantile_grid = true;
        }
        else if (token == "--draw-theta")
        {
            // theta of the normal-gamma prior drawn rather than fixed.
            draw_theta = true;
        }
        else if (token == "--stationary")
        {
            // Every kept draw of `a` stationary: /model/stationary.
            stationary = true;
        }
        else if (token == "--soft")
        {
            // With --mixed: the quarterly averages soft, their variance drawn.
            soft = true;
        }
        else if (token == "--partial-test")
        {
            // A horizon realised in part: /data/test/constraints.
            partial_test = true;
        }
        else if (token == "--scenario")
        {
            // A conditional forecast: /data/forecast/constraints.
            scenario = true;
        }
        else if (token == "--missing")
        {
            // Missing values and a ragged edge: /data/train/constraints.
            missing = true;
        }
        else if (token == "--deterministic")
        {
            // Deterministic terms in a factor model: /model/n and /data/train/x.
            deterministic = true;
        }
        else if (token == "--smooth")
        {
            // An unobserved components trend whose second difference is the shock.
            smooth = true;
        }
        else if (token == "--own-cycle")
        {
            // An unobserved components cycle in its own lags only.
            own_cycle = true;
        }
        else if (token == "--stationary-init")
        {
            // An unobserved components cycle started from its stationary
            // distribution rather than from zero.
            stationary_init = true;
        }
        else if (token == "--damped")
        {
            // An unobserved components trend whose slope is a stationary AR(1).
            uc_damped = true;
        }
        else if (token == "--signals")
        {
            // Signal equations on an unobserved components cycle.
            uc_signals = true;
        }
        else if (token == "--irregular")
        {
            // An unobserved components model's irregular component.
            uc_irregular = true;
        }
        else if (token == "--ald")
        {
            // UcNormalSkew's asymmetric Laplace in place of the skew normal.
            uc_ald = true;
        }
        else if (token == "--skew-trend")
        {
            // UcNormalSkew with the trend shock skewed rather than the cycle's.
            skew_trend = true;
        }
        else if (token == "--skew-index")
        {
            // UcNormalSkew's scales moving with an index.
            skew_index = true;
        }
        else if (token == "--iid")
        {
            // The first equation carries no coefficients at all: /model/n_iid.
            iid = true;
        }
        else if (token == "--noncentred")
        {
            // omega_v in place of shape and rate in every prior on how far a
            // random walk moves: the non-centred parameterisation, and with it
            // the Savage-Dickey ordinates under /posterior.
            noncentred = true;
        }
        else if (token == "append")
        {
            // Add this model to a file that already holds one, rather than
            // starting a new file -- the only way to build the
            // several-models-in-one-file case --all-groups exists for. Every
            // model still needs its own group, since two at the root would be
            // the same model written twice.
            append = true;
        }
        else if (!bare_group_seen && !token.empty() && token.rfind("--", 0) != 0)
        {
            group_argument = token;
            bare_group_seen = true;
        }
        else
        {
            std::cerr << "Error: unexpected argument '" << token << "'\n";
            return 2;
        }
    }

    // The group the model is written under, normalized the same way the reader
    // normalizes --group, so the two agree on what "/models/3/" means.
    std::string group;
    if (bare_group_seen)
    {
        try
        {
            group = normalize_hdf5_group(group_argument);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return 2;
        }
    }

    // For the messages below: "dest.h5" or "dest.h5:/models/3".
    const std::string group_suffix = group.empty() ? "" : ":" + group;

    // The unobserved components models, handled before everything below: none
    // of the VAR, VEC or factor flags applies to them but --noncentred, which
    // UcNormalStochvol reads, and they take six of their own.
    const bool is_uc = model == "UcNormalGamma" || model == "UcNormalWishart" ||
                       model == "UcNormalStochvol" || model == "UcNormalSkew";
    if (is_uc)
    {
        const bool covar_model = model == "UcNormalGamma" || model == "UcNormalStochvol";
        const bool skew_model = model == "UcNormalSkew";
        if (varsel != "none" || structural || coint_rho || coint_p_tau ||
            (hold_states && model != "UcNormalStochvol") ||
            (noncentred && model != "UcNormalStochvol") || iid || mixed || missing ||
            partial_test || !shrinkage.empty() || draw_theta || quantile_grid || steady_state ||
            soft || scenario || (smooth && deterministic) || (score && h <= 0) ||
            (covar && !covar_model) || ((uc_ald || skew_trend || skew_index) && !skew_model) ||
            (score && skew_model))
        {
            std::cerr << "The unobserved components models take covar (UcNormalGamma and "
                         "UcNormalStochvol), h, a group, --score (with h > 0, and not on "
                         "UcNormalSkew), --stationary, --deterministic, --smooth, --own-cycle, "
                         "--irregular, --noncentred and --hold-states (UcNormalStochvol), and --ald, --skew-trend "
                         "and --skew-index (UcNormalSkew); and not --smooth with "
                         "--deterministic: a smooth trend has no drift here\n";
            return 2;
        }
        try
        {
            if (!dest.parent_path().empty())
            {
                std::filesystem::create_directories(dest.parent_path());
            }
            if (append && group.empty())
            {
                std::cerr << "Error: appending needs a group to write the model under\n";
                return 2;
            }
            const bool add_to_existing = append && std::filesystem::exists(dest);
            if (!add_to_existing)
            {
                std::filesystem::remove(dest);
            }
            HighFive::File h5(dest.string(), add_to_existing ? HighFive::File::ReadWrite
                                                             : HighFive::File::Create);
            const ModelFile file(h5, group);

            Lcg rng(20260808ULL);
            const arma::mat y = simulate_uc(rng, smooth, deterministic);

            UcOptions options;
            options.model = model;
            options.h = h;
            options.covar = covar;
            options.smooth = smooth;
            options.own_cycle = own_cycle;
            options.with_break = deterministic;
            options.stationary = stationary;
            options.stationary_init = stationary_init;
            options.damped = uc_damped;
            options.signals = uc_signals;
            options.irregular = uc_irregular;
            options.noncentred = noncentred;
            options.ald = uc_ald;
            options.skew_trend = skew_trend;
            options.skew_index = skew_index;
            write_uc_model(file, y, options);
            if (hold_states)
            {
                write_attribute<std::string>(file, "/model", "forecast_states", "hold");
            }
            if (score)
            {
                write_uc_test_observations(file, y.col(kTT - 1), h);
            }

            std::cout << "wrote " << dest.string() << group_suffix << " (" << model
                      << ", covar=" << covar << ", h=" << h << ", k=" << kUcK << ", tt=" << kTT
                      << ", p=" << kUcP
                      << ", trend=" << (smooth ? "smooth" : (uc_damped ? "damped" : "rw"))
                      << ", cycle=" << (own_cycle ? "own" : "var")
                      << (deterministic ? ", break=1" : "") << (stationary ? ", stationary=1" : "")
                      << (stationary_init ? ", cycle_init=stationary" : "")
                      << (uc_irregular ? ", irregular=1" : "")
                      << (uc_signals ? ", signals=2" : "")
                      << (noncentred ? ", noncentred=1" : "") << (uc_ald ? ", skew=ald" : "")
                      << (skew_trend ? ", skew_shock=trend" : "")
                      << (skew_index ? ", skew_index=1" : "") << (score ? ", score=1" : "")
                      << ")\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << '\n';
            return 1;
        }
        return 0;
    }
    if (smooth || own_cycle || stationary_init || uc_damped || uc_signals || uc_irregular ||
        uc_ald || skew_trend || skew_index)
    {
        std::cerr << "--smooth, --own-cycle, --stationary-init, --damped, --signals, --irregular, "
                     "--ald, --skew-trend and --skew-index are the unobserved components models'\n";
        return 2;
    }

    const bool is_vec = is_vec_model(model);
    const bool is_dfm = model == "DfmNormalGamma" || model == "DfmNormalStochvol" ||
                        model == "DfmTvpGamma" || model == "DfmTvpStochvol";
    const bool is_favar = model == "FavarNormalWishart";

    const bool is_ald = model == "VarNormalAld" || model == "VarTvpAld";
    const bool is_discount = model == "VarTvpDiscount" || model == "VecTvpDiscount";

    if (model != "VarNormalWishart" && model != "VarNormalGamma" &&
        model != "VarNormalStochvol" && model != "VarTvpGamma" && model != "VarTvpWishart" &&
        model != "VarTvpStochvol" && model != "VarTvpDiscount" && !is_vec && !is_dfm &&
        !is_favar && !is_ald)
    {
        std::cerr << "Unknown model: " << model << '\n';
        return 2;
    }
    if (is_discount && (varsel != "none" || covar || structural))
    {
        std::cerr << "The discounted models take no variable selection -- there are no draws "
                     "for an inclusion indicator to be drawn alongside -- no covariance block "
                     "and no contemporaneous coefficients, an inverse Wishart posterior leaving "
                     "the error covariance unrestricted: see validate() in "
                     "src/core/models/var_tvp_discount.cpp\n";
        return 2;
    }
    if (is_dfm && (varsel != "none" || covar || structural))
    {
        std::cerr << "A dynamic factor model takes no variable selection, no covariance block "
                     "and no contemporaneous coefficients: see validate_dfm_shape() in "
                     "src/core/inputs.cpp\n";
        return 2;
    }
    if (score && h <= 0)
    {
        std::cerr << "--score writes the observations a horizon realised, so it needs a "
                     "horizon to realise: pass h > 0" << '\n';
        return 2;
    }
    if (score && is_ald)
    {
        std::cerr << "--score is not available for " << model
                  << ": the quantile models have no forecast to score -- see can_be_scored in "
                     "src/check.cpp for the twenty that are" << '\n';
        return 2;
    }
    if (quantile_grid && (model != "VarNormalAld" || (h != 0 && !structural)))
    {
        std::cerr << "--quantile-grid is written on VarNormalAld only, and forecasts only with a "
                     "structural form: the simulation draws each variable given the ones before "
                     "it\n";
        return 2;
    }
    if (is_ald && (covar || (h != 0 && !quantile_grid)))
    {
        std::cerr << "A quantile regression model takes no covariance block -- rotating the "
                     "equations into each other leaves a residual whose quantile is not the one "
                     "asked for -- and no forecast horizon, because the h step quantile is not "
                     "the quantile of the iterated one step quantiles: see validate_ald_spec() "
                     "in src/core/inputs.cpp\n";
        return 2;
    }
    if (varsel != "none" && varsel != "ssvs" && varsel != "bvs")
    {
        std::cerr << "Unknown variable selection scheme: " << varsel << '\n';
        return 2;
    }
    const bool scores_panels = model == "VarNormalWishart" || model == "VarNormalGamma";
    const bool completes = scores_panels || model == "VarNormalStochvol" ||
                           model == "VarTvpWishart" || model == "VarTvpGamma" ||
                           model == "VarTvpStochvol";
    if ((mixed || missing) && (!completes || mixed == missing || structural ||
                               (score && !scores_panels)))
    {
        std::cerr << "--mixed and --missing are written on the six Gaussian VARs only, one at a "
                     "time, without a structural form, and with --score only on "
                     "VarNormalWishart and VarNormalGamma: the combinations "
                     "require_completion_spec() refuses\n";
        return 2;
    }
    const bool adapts = model == "VarNormalWishart" || model == "VarNormalGamma" ||
                        model == "VarNormalStochvol";
    if ((!shrinkage.empty() || stationary || steady_state) &&
        (!adapts || varsel != "none" || iid || structural || (steady_state && !shrinkage.empty()) ||
                                               (!shrinkage.empty() && shrinkage != "minnesota" &&
                                                shrinkage != "horseshoe" && shrinkage != "normal_gamma")))
    {
        std::cerr << "--shrinkage=minnesota|horseshoe|normal_gamma and --stationary are written on "
                     "VarNormalWishart, VarNormalGamma and VarNormalStochvol only, without "
                     "variable selection, --iid or a structural form\n";
        return 2;
    }
    if (volatility_break &&
        (model != "VarNormalWishart" || steady_state || mixed || missing || partial_test || soft))
    {
        std::cerr << "--volatility-break is written on VarNormalWishart only, without "
                     "--steady-state and on a panel and horizon observed whole\n";
        return 2;
    }
    if (draw_theta && shrinkage != "normal_gamma")
    {
        std::cerr << "--draw-theta needs --shrinkage=normal_gamma\n";
        return 2;
    }
    if (soft && !mixed)
    {
        std::cerr << "--soft makes the quarterly averages of --mixed soft, so it needs --mixed\n";
        return 2;
    }
    if ((partial_test || scenario) &&
        (!scores_panels || iid || structural || h < 3 || (partial_test && !score) ||
         (scenario && score)))
    {
        std::cerr << "--partial-test and --scenario are written on VarNormalWishart and "
                     "VarNormalGamma only, with "
                     "h >= 3 and without --iid or a structural form; --partial-test needs "
                     "--score, and --scenario refuses it: a conditional forecast is not scored\n";
        return 2;
    }
    if (iid && model != "VarNormalWishart")
    {
        std::cerr << "--iid is written on VarNormalWishart only. The other three constant VARs "
                     "read n_iid as well and unit.iid_block covers them; every remaining "
                     "algorithm refuses it, which is what require_supported_iid_block() is for\n";
        return 2;
    }
    if (iid && (varsel != "none" || structural))
    {
        std::cerr << "--iid cannot be combined with variable selection or a structural form: "
                     "both are refused by require_supported_iid_block(), the first because a "
                     "coefficient fixed at zero has nothing left to select over, the second "
                     "because the contemporaneous block is laid out by a different rule\n";
        return 2;
    }
    if (hold_states && model != "VarTvpWishart" && model != "VarTvpGamma" &&
        model != "VarTvpStochvol" && model != "VarNormalStochvol" &&
        model != "VecTvpWishart" && model != "VecTvpGamma" && model != "VecTvpStochvol" &&
        model != "VecNormalStochvol" && model != "DfmNormalStochvol" &&
        model != "DfmTvpGamma" && model != "DfmTvpStochvol")
    {
        std::cerr << "Only the models whose states drift read forecast_states: expected one of "
                     "VarTvpWishart, VarTvpGamma, VarTvpStochvol, VarNormalStochvol, "
                     "VecTvpWishart, VecTvpGamma, VecTvpStochvol, VecNormalStochvol, "
                     "DfmNormalStochvol, DfmTvpGamma, DfmTvpStochvol\n";
        return 2;
    }
    if (noncentred && model != "VarTvpStochvol" && model != "VarTvpGamma" &&
        model != "VarTvpWishart" && model != "VarTvpAld" && model != "VecTvpStochvol" &&
        model != "VecTvpGamma" && model != "VecTvpWishart" && model != "DfmTvpGamma" &&
        model != "DfmTvpStochvol")
    {
        std::cerr << "Only a sampler whose coefficients drift reads the non-centred prior "
                     "omega_v: expected one of VarTvpWishart, VarTvpGamma, VarTvpStochvol, "
                     "VarTvpAld, VecTvpWishart, VecTvpGamma, VecTvpStochvol, DfmTvpGamma, "
                     "DfmTvpStochvol\n";
        return 2;
    }
    if (deterministic && !is_dfm && !is_favar)
    {
        std::cerr << "--deterministic is the factor models' fixture: expected one of "
                     "DfmNormalGamma, DfmNormalStochvol, DfmTvpGamma, DfmTvpStochvol, "
                     "FavarNormalWishart\n";
        return 2;
    }
    if (coint_rho && model != "VecTvpWishart" && model != "VecTvpGamma" &&
        model != "VecTvpStochvol")
    {
        std::cerr << "Only a VEC whose cointegration space moves has an autoregression to put a "
                     "prior on: expected one of VecTvpWishart, VecTvpGamma, VecTvpStochvol\n";
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
        if (append && group.empty())
        {
            std::cerr << "Error: appending needs a group to write the model under\n";
            return 2;
        }

        const bool add_to_existing = append && std::filesystem::exists(dest);
        if (!add_to_existing)
        {
            std::filesystem::remove(dest);
        }

        HighFive::File h5(dest.string(), add_to_existing ? HighFive::File::ReadWrite
                                                         : HighFive::File::Create);

        // Everything below names absolute paths -- "/model", "/data/train/y" --
        // and this is what puts the group in front of them. Empty group, same
        // paths as before.
        const ModelFile file(h5, group);

        Layout layout;
        layout.n_structural = structural ? kK * (kK - 1) / 2 : 0;
        layout.nparams = layout.n_non_structural + layout.n_structural;

        // One generator for the whole file, so every dataset that draws from it
        // is reproducible as a set rather than individually.
        Lcg rng(20260808ULL);

        if (is_favar)
        {
            arma::mat obs;
            arma::mat x = simulate_favar(rng, obs);
            if (deterministic)
            {
                const arma::mat d = fixture_deterministic(0, kTT);
                x += fixture_deterministic_effects(kDfmK) * arma::trans(d);
                obs += d * arma::trans(fixture_deterministic_effects(kFavarNObs));
            }
            write_favar_normal_wishart(file, h, x, obs);
            if (deterministic)
            {
                write_factor_deterministic(file, h, {{"c", kDfmK}, {"c_obs", kFavarNObs}});
            }
            if (score)
            {
                // The panel and then the observed factors, as the forecast holds
                // them per period.
                write_test_observations(
                    file, arma::join_vert(x.col(kTT - 1), arma::trans(obs.row(kTT - 1))), h);
            }

            std::cout << "wrote " << dest.string() << group_suffix << " (" << model
                      << ", h=" << h << ", k=" << kDfmK << ", tt=" << kTT
                      << ", n_factors=" << kDfmN << ", n_obs_factors=" << kFavarNObs
                      << ", p=" << kDfmP << ", n_lambda=" << kFavarNLambda
                      << ", n_a=" << kFavarNA << (deterministic ? ", n=3" : "") << ")\n";
            return 0;
        }

        if (is_dfm)
        {
            arma::mat x = simulate_dfm(rng);
            if (deterministic)
            {
                x += fixture_deterministic_effects(kDfmK) * arma::trans(fixture_deterministic(0, kTT));
            }
            if (model == "DfmNormalStochvol")
            {
                write_dfm_normal_stochvol(file, h, x);
            }
            else if (model == "DfmTvpGamma")
            {
                write_dfm_tvp_gamma(file, h, x, noncentred);
            }
            else if (model == "DfmTvpStochvol")
            {
                write_dfm_tvp_stochvol(file, h, x, noncentred);
            }
            else
            {
                write_dfm_normal_gamma(file, h, x);
            }

            if (hold_states)
            {
                write_attribute<std::string>(file, "/model", "forecast_states", "hold");
            }

            if (deterministic)
            {
                write_factor_deterministic(file, h, {{"c", kDfmK}});
            }

            if (score)
            {
                write_test_observations(file, x.col(kTT - 1), h);
            }

            std::cout << "wrote " << dest.string() << group_suffix << " (" << model << ", h=" << h
                      << ", k=" << kDfmK << ", tt=" << kTT << ", n_factors=" << kDfmN
                      << ", p=" << kDfmP << ", n_lambda=" << kDfmNLambda << ", n_a=" << kDfmNA
                      << (hold_states ? ", forecast_states=hold" : "")
                      << (score ? ", score=1" : "") << (deterministic ? ", n=3" : "") << ")\n";
            return 0;
        }

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
            write_vec_model(file, model, varsel, covar, nparams, noncentred);

            if (coint_rho)
            {
                write_vec_tvp_coint_rho_prior(file);
            }

            if (coint_p_tau)
            {
                write_vec_tvp_coint_p_tau(file);
            }

            if (hold_states)
            {
                write_attribute<std::string>(file, "/model", "forecast_states", "hold");
            }

            // Levels, the parameterisation a VEC forecasts and is scored in --
            // the same series /data/forecast/x carries the lags of, not the
            // differences /data/train/y holds.
            if (score)
            {
                write_test_observations(file, levels.col(kTT + kVecP - 1), h);
            }

            // coint_p_tau only when set, so the line every existing fixture
            // prints stays the one its recorded fingerprints were taken from.
            std::cout << "wrote " << dest.string() << group_suffix << " (" << model << ", varsel=" << varsel
                      << ", covar=" << covar << ", structural=" << structural << ", h=" << h
                      << ", k=" << kK << ", tt=" << kTT << ", rank=" << kVecRank
                      << ", nparams=" << nparams << ", coint_rho=" << coint_rho
                      << (coint_p_tau ? ", coint_p_tau=1" : "")
                      << (hold_states ? ", forecast_states=hold" : "")
                      << (score ? ", score=1" : "") << ")\n";
            return 0;
        }

        const arma::mat series = simulate_series(rng);
        const arma::mat z_train = build_train_regressors(series, layout, rng);

        write_common(file, model, varsel, covar, structural, h, layout, series, z_train,
                     iid ? 1 : 0);

        if (hold_states)
        {
            write_attribute<std::string>(file, "/model", "forecast_states", "hold");
        }

        if (score)
        {
            const arma::mat realised = write_test_observations(file, series.col(kTT + kP - 1), h);
            if (partial_test)
            {
                write_partial_test(file, realised);
            }
        }
        if (scenario)
        {
            write_scenario(file);
        }

        if (mixed || missing)
        {
            write_incomplete_panel(file, series, mixed, soft);
        }
        if (!shrinkage.empty())
        {
            write_shrinkage(file, shrinkage, layout, draw_theta);
        }
        if (stationary)
        {
            write_attribute<bool>(file, "/model", "stationary", true);
        }
        if (steady_state)
        {
            // A prior centred on zero with a standard deviation of one per
            // series, the chain starting at zero.
            write_attribute<bool>(file, "/model", "steady_state", true);
            ensure_group(file, "/priors/mu");
            write_row(file, "/priors/mu/mu", arma::vec(kK, arma::fill::zeros));
            write_mat(file, "/priors/mu/v_inv", arma::eye<arma::mat>(kK, kK));
            write_row(file, "/initial/mu", arma::vec(kK, arma::fill::zeros));
        }

        if (volatility_break)
        {
            // The break five periods from the end: three free scales and two
            // periods of decay inside the sample. The hyperprior of Lenza and
            // Primiceri (2022): Pareto(1, 1) on the scales, a beta with mode
            // 0.8 and standard deviation 0.2 on the decay.
            write_attribute<int>(file, "/model", "volatility_break", kTT - 4);
            write_attribute<int>(file, "/model", "volatility_scales", 3);
            ensure_group(file, "/priors/volatility");
            write_dataset_double(file, "/priors/volatility/shape", 1.0);
            write_dataset_double(file, "/priors/volatility/rho_shape1", 3.0357);
            write_dataset_double(file, "/priors/volatility/rho_shape2", 1.5089);
            write_row(file, "/initial/volatility", arma::vec({2.0, 2.0, 2.0, 0.5}));
        }

        if (model == "VarTvpDiscount")
        {
            write_var_tvp_discount(file, layout);
        }
        else if (model == "VarNormalWishart")
        {
            write_var_normal_wishart(file, varsel, layout);
        }
        else if (model == "VarNormalGamma")
        {
            write_var_normal_gamma(file, varsel, covar, layout);
        }
        else if (model == "VarNormalStochvol")
        {
            write_var_normal_stochvol(file, varsel, covar, layout);
        }
        else if (model == "VarNormalAld")
        {
            write_var_normal_ald(file, varsel, layout);
            if (quantile_grid)
            {
                write_vector_attribute(file, "/model", "quantiles",
                                       std::vector<double>{0.1, 0.25, 0.5, 0.75, 0.9});
            }
        }
        else if (model == "VarTvpAld")
        {
            write_var_tvp_ald(file, varsel, layout, noncentred);
        }
        else if (model == "VarTvpWishart")
        {
            write_var_tvp_wishart(file, varsel, layout, noncentred);
        }
        else if (model == "VarTvpStochvol")
        {
            write_var_tvp_stochvol(file, varsel, covar, layout, noncentred);
        }
        else
        {
            write_var_tvp_gamma(file, varsel, covar, layout, noncentred);
        }

        std::cout << "wrote " << dest.string() << group_suffix << " (" << model << ", varsel=" << varsel
                  << ", covar=" << covar << ", structural=" << structural << ", h=" << h
                  << ", k=" << kK << ", tt=" << kTT << ", nparams=" << layout.nparams
                  << (hold_states ? ", forecast_states=hold" : "")
                  << (score ? ", score=1" : "") << (noncentred ? ", noncentred=1" : "")
                  << (mixed ? ", mixed=1" : "") << (missing ? ", missing=1" : "")
                  << (partial_test ? ", partial_test=1" : "") << (scenario ? ", scenario=1" : "")
                  << (soft ? ", soft=1" : "")
                  << (shrinkage.empty() ? "" : ", shrinkage=" + shrinkage)
                  << (stationary ? ", stationary=1" : "")
                  << (draw_theta ? ", draw_theta=1" : "")
                  << (steady_state ? ", steady_state=1" : "")
                  << (volatility_break ? ", volatility_break=1" : "")
                  << ")\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
