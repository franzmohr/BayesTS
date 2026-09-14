// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

// That the constant VECs sample the cointegration matrix of a real country model
// of a global VEC, and mix while they do.
//
// A country model of a global VEC carries its weakly exogenous variables in the
// cointegration term, so k_beta exceeds k by their number, and by one more for a
// restricted trend. That is where the draw of Koop, Leon-Gonzalez and Strachan
// (2010) needs the auxiliary loadings of augment_loadings(). The correction it
// replaced, a Metropolis-Hastings step on the normal draw, was right in
// distribution and kept so few proposals on these models that the chains
// barely moved: on the 26 country models of Dees, di Mauro, Pesaran and Smith
// (2007) the median model kept the previous beta in 92% of its draws, and one
// never left its starting values in 10,000.
//
// The model here is the US one of that paper, as the bgvars package sets it up
// (test/dees2007_us.h): six endogenous variables, three foreign ones and a
// restricted trend, so k = 6, k_beta = 10 and rank 2, over 97 quarters, with
// the flat priors of the bgvars GVEC vignette. Under the Metropolis-Hastings
// step it was one of the better mixing ones -- 62% of its draws repeated beta,
// and the median effective sample size of the elements of Pi = alpha beta' was
// 682 of 10,000 by the estimator below; with the auxiliary loadings it is about
// 1,000 of 5,000.
//
// What the auxiliary loadings do not change is the slowest element. Pi's column
// on the restricted trend mixes slowly under either correction -- its smallest
// effective sample size was 30 of 10,000 before and is 15 to 45 of 5,000 now,
// with a first-order autocorrelation of 0.9 -- so that is the Gibbs split of Pi
// into alpha and beta on this posterior rather than something a test here could
// hold the draw of beta to.
//
// Checked, for VecNormalWishart and VecKlgs2010:
//
// 1. beta moves in every draw, as a Gibbs step does.
// 2. The median effective sample size of the 60 elements of Pi is at least a
//    tenth of the draws.
// 3. The two samplers, run from different seeds, give posterior means of Pi
//    within five Monte Carlo standard errors of each other.
// 4. The paper's contemporaneous effects of foreign output and inflation on
//    their US counterparts, 0.54 and 0.06 in its Table B11, lie inside the 90%
//    posterior intervals of those coefficients.

#include "dees2007_us.h"

#include "bayests/vec_klgs_2010.h"
#include "bayests/vec_normal_wishart.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace
{

using namespace dees2007_us;

constexpr int kNAlpha = kK * kRank;
constexpr int kNA = kNAlpha + kK * kNX;
constexpr int kNBeta = kKBeta * kRank;
constexpr int kIterations = 5000;
constexpr int kBurnin = 1000;
constexpr unsigned long long kSeed = 20070101ULL;

/// Positions in `a` of the coefficients on the current difference of foreign
/// output in the output equation and of foreign inflation in the inflation
/// equation: after the loadings, k coefficients per column of x, whose seventh
/// and eighth columns are d.y.s.00 and d.Dp.s.00.
constexpr arma::uword kPosForeignY = kNAlpha + 6 * kK + 0;
constexpr arma::uword kPosForeignDp = kNAlpha + 7 * kK + 1;

int failures = 0;

void check(const std::string &label, bool condition, const std::string &detail = "")
{
    failures += condition ? 0 : 1;
    std::cout << (condition ? "  ok   " : "  FAIL ") << label << "\n";
    if (!detail.empty())
    {
        std::cout << "         " << detail << "\n";
    }
}

/// A tt x n matrix from a row-major array.
arma::mat rows_of(const double *values, int tt, int n)
{
    return arma::trans(arma::mat(values, n, tt));
}

struct Data
{
    arma::mat y, w, x, z;
};

Data make_data()
{
    Data d;
    d.y = rows_of(kY, kTT, kK);
    d.w = rows_of(kW, kTT, kKBeta);
    d.x = rows_of(kX, kTT, kNX);

    // The SUR reading of the same regressors, with the loading columns built
    // from the starting beta. Every VEC rebuilds them from its own draw.
    const arma::mat diag_k = arma::eye<arma::mat>(kK, kK);
    const arma::mat beta0 = arma::reshape(arma::vec(kInitialBeta, kNBeta), kKBeta, kRank);
    d.z = arma::mat(kTT * kK, kNA, arma::fill::zeros);
    for (int t = 0; t < kTT; ++t)
    {
        d.z.submat(t * kK, 0, (t + 1) * kK - 1, kNAlpha - 1) =
            arma::kron(arma::trans(arma::trans(beta0) * arma::trans(d.w.row(t))), diag_k);
        d.z.submat(t * kK, kNAlpha, (t + 1) * kK - 1, kNA - 1) = arma::kron(d.x.row(t), diag_k);
    }
    return d;
}

bayests::VarSpec make_spec()
{
    bayests::VarSpec spec;
    spec.k = kK;
    spec.p = kP;
    spec.m = 3;            // foreign output, inflation and exchange rate
    spec.s = 2;            // their current difference and one lag of it
    spec.n = 1;            // the unrestricted constant
    spec.n_restricted = 1; // the trend
    spec.rank = kRank;
    spec.k_beta = kKBeta;
    spec.iterations = kIterations;
    spec.burnin = kBurnin;
    return spec;
}

/// The priors of the bgvars GVEC vignette: flat on the coefficients and on the
/// cointegration space, and a Wishart with 3 + rank degrees of freedom and a
/// scale of 1e-4 on the error precision. Starting values are the maximum
/// likelihood estimates.
template <typename Input> void fill_common(Input &in, const Data &d)
{
    in.spec = make_spec();
    in.train.y = d.y;
    in.train.w = d.w;
    in.a_prior.mu = arma::vec(kNA, arma::fill::zeros);
    in.a_prior.v_inv = arma::mat(kNA, kNA, arma::fill::zeros);
    in.beta_prior.v_inv = 0.0;
    in.beta_prior.p_tau_inv = arma::eye<arma::mat>(kKBeta, kKBeta);
    in.u_sigma_prior.df = 3.0 + kRank;
    in.u_sigma_prior.scale = 1e-4 * arma::eye<arma::mat>(kK, kK);
    in.initial.a = arma::vec(kInitialA, kNA);
    in.initial.beta = arma::vec(kInitialBeta, kNBeta);
    in.initial.u_sigma_inv = arma::reshape(arma::vec(kInitialSigmaInv, kK * kK), kK, kK);
}

bayests::VecNormalWishartInput make_sur(const Data &d)
{
    bayests::VecNormalWishartInput in;
    fill_common(in, d);
    in.train.z = d.z;
    return in;
}

bayests::VecKlgs2010Input make_compact(const Data &d)
{
    bayests::VecKlgs2010Input in;
    fill_common(in, d);
    in.train.x = d.x;
    return in;
}

/// Effective sample size of a chain from Geyer's initial positive sequence: the
/// autocorrelations are summed in pairs until a pair is no longer positive.
/// Unlike batch means, it sees a chain that repeats its values for long
/// stretches for what it is.
double effective_size(const arma::rowvec &chain)
{
    const arma::uword n = chain.n_elem;
    const arma::vec centred = arma::trans(chain) - arma::mean(chain);
    const double gamma0 = arma::dot(centred, centred) / static_cast<double>(n);
    if (!(gamma0 > 0.0))
    {
        return 0.0;
    }
    const auto rho = [&](arma::uword lag) {
        return arma::dot(centred.head(n - lag), centred.tail(n - lag)) /
               (static_cast<double>(n) * gamma0);
    };
    double tau = -1.0;
    for (arma::uword lag = 0; lag + 1 < n / 2; lag += 2)
    {
        const double pair = rho(lag) + rho(lag + 1);
        if (pair <= 0.0)
        {
            break;
        }
        tau += 2.0 * pair;
    }
    return static_cast<double>(n) / tau;
}

struct Summary
{
    arma::vec pi_mean;
    arma::vec pi_se;
};

Summary summarise(const std::string &model, const arma::mat &a, const arma::mat &beta)
{
    const arma::uword draws = a.n_cols;

    // 1. beta moves in every draw.
    arma::uword repeated = 0;
    for (arma::uword i = 1; i < draws; ++i)
    {
        repeated += arma::approx_equal(beta.col(i), beta.col(i - 1), "absdiff", 0.0) ? 1 : 0;
    }
    check(model + ": beta moves in every draw", repeated == 0,
          std::to_string(repeated) + " of " + std::to_string(draws - 1) + " draws repeat beta");

    // 2. Pi mixes.
    arma::mat pi(kK * kKBeta, draws);
    for (arma::uword i = 0; i < draws; ++i)
    {
        const arma::mat alpha = arma::reshape(a.col(i).head(kNAlpha), kK, kRank);
        const arma::mat b = arma::reshape(beta.col(i), kKBeta, kRank);
        pi.col(i) = arma::vectorise(alpha * arma::trans(b));
    }
    arma::vec ess(pi.n_rows);
    for (arma::uword j = 0; j < pi.n_rows; ++j)
    {
        ess(j) = effective_size(pi.row(j));
    }
    check(model + ": the median effective sample size of Pi is at least a tenth of the draws",
          arma::median(ess) >= draws / 10.0,
          "median " + std::to_string(arma::median(ess)) + ", smallest " + std::to_string(ess.min()) +
              " of " + std::to_string(draws));

    // 4. Table B11 inside the 90% intervals.
    const auto inside = [&](arma::uword pos, double paper, const std::string &what) {
        arma::rowvec coef = arma::sort(a.row(pos));
        const double lower = coef(static_cast<arma::uword>(0.05 * (draws - 1)));
        const double upper = coef(static_cast<arma::uword>(0.95 * (draws - 1)));
        check(model + ": Table B11's effect of " + what + " is inside the 90% interval",
              paper >= lower && paper <= upper,
              std::to_string(paper) + " against [" + std::to_string(lower) + ", " +
                  std::to_string(upper) + "]");
    };
    inside(kPosForeignY, 0.54, "foreign output");
    inside(kPosForeignDp, 0.06, "foreign inflation");

    Summary out;
    out.pi_mean = arma::mean(pi, 1);
    out.pi_se = arma::stddev(pi, 0, 1) / arma::sqrt(ess);
    return out;
}

} // namespace

int main()
{
    const Data data = make_data();
    bayests::NullReporter reporter;

    std::cout << "VecNormalWishart, the US model of Dees et al. (2007):\n";
    arma::arma_rng::set_seed(kSeed);
    const bayests::VecNormalWishartDraws sur =
        bayests::VecNormalWishartSampler{}.draw_coefficients(make_sur(data), reporter);
    const Summary s_sur = summarise("VecNormalWishart", sur.a, sur.beta);

    std::cout << "VecKlgs2010, the same model:\n";
    arma::arma_rng::set_seed(kSeed + 1);
    const bayests::VecKlgs2010Draws compact =
        bayests::VecKlgs2010Sampler{}.draw_coefficients(make_compact(data), reporter);
    const Summary s_compact = summarise("VecKlgs2010", compact.a, compact.beta);

    // 3. The same posterior from both.
    std::cout << "both samplers:\n";
    const arma::vec z =
        (s_sur.pi_mean - s_compact.pi_mean) / arma::sqrt(arma::square(s_sur.pi_se) + arma::square(s_compact.pi_se));
    check("posterior means of Pi agree within five Monte Carlo standard errors", arma::abs(z).max() < 5.0,
          "largest |z| " + std::to_string(arma::abs(z).max()));

    std::cout << (failures == 0 ? "all as expected\n" : "SOMETHING IS WRONG\n");
    return failures == 0 ? 0 : 1;
}
