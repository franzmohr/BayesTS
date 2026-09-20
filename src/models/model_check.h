// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_MODELS_MODEL_CHECK_H
#define BAYESTS_MODELS_MODEL_CHECK_H

// What `bayests check` runs for one model: the reader and the validate() a run
// would use, with nothing drawn and nothing written.
//
// The point is that it is the same code. A checker with its own idea of what
// each model needs would drift from the readers the way prose does, and a file
// it passed would still be refused -- or, worse, still be read as something
// else. So each front-end hands its own read_input() to one of the four
// templates at the bottom, and the only rules restated here are the ones a run
// meets after the sampler has finished: the forecast's checks on its
// regressors, which the samplers can only make once there are draws to compare
// against. The check.* tests in
// test/CMakeLists.txt run this over every fixture a golden.* run accepts, so a
// rule here that is stricter than the run fails there.

#include "models/models.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/model_io_common.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace bayests::model_check_detail
{

/// Whether a model's input has a selection scheme of its own for the covariance
/// block, which is what reading /model/priors/psi's `varsel` fills in.
template <typename Input>
struct has_psi_varsel
{
    static constexpr bool value = requires(const Input &input) { input.psi_varsel; };
};

/// One block's selection prior against the coefficient prior it selects over,
/// for `bvs` and for a prior that has a single variance to read.
///
/// The `if constexpr` is the whole of the scope rule: a time-varying block's
/// prior is a RandomWalkPrior and simply does not match, so the four families
/// share this call and only the constant-coefficient ones do anything with it.
/// See bayests::flat_selection_prior() for why that is the right scope rather
/// than a gap.
template <typename Prior>
void collect_flat_selection(ModelCheck &check, const char *block, const VarSelection scheme,
                            const VarSelPrior &varsel, const Prior &prior)
{
    if (scheme != VarSelection::bvs)
    {
        return;
    }

    if constexpr (std::is_same_v<Prior, NormalPrior>)
    {
        const FlatSelectionPrior report = flat_selection_prior(varsel, prior.v_inv);
        if (report.flat > 0)
        {
            check.flat_selection.push_back({block, report});
        }
    }
}

/// Where a fitted posterior is looked for. Every sampler here writes an error
/// precision, so its presence is what says the model has been run; the
/// discounted pair writes a covariance and no draws at all, and is probed at
/// the first dataset of its closed form instead.
constexpr const char *kDefaultPosteriorProbe = "/posterior/u_sigma_inv/coeffs";

/// Everything the check reports that does not depend on which model it is.
template <typename Input>
ModelCheck inspect(const ModelFile &file, const Input &input,
                   const char *posterior_probe = kDefaultPosteriorProbe)
{
    ModelCheck check;
    check.spec = input.spec;

    // tt is never stored; it is what the reader's validate() recovers too.
    const arma::uword k = input.spec.k > 0 ? static_cast<arma::uword>(input.spec.k) : 0;
    check.periods = k > 0 ? input.train.y.n_elem / k : 0;
    check.z_columns = input.train.z.n_cols;
    check.test_periods = input.test.y.n_rows;

    // Taken before anything below opens a dataset, so that the list is what
    // the reader asked for and nothing else.
    const std::vector<std::string> read = file.datasets_read();

    check.error_attribute = hdf5_io::optional_attribute_string(file, "/model", "error", "");
    check.has_posterior = dataset_has_data(file, posterior_probe);

    // Not the reader's -- the command line seeds the generator, see
    // src/model_seed.cpp -- but a seed a run would refuse, the check refuses.
    check.seed = hdf5_io::read_model_seed(file);

    for (const std::string &name : list_model_datasets(file))
    {
        if (!std::binary_search(read.begin(), read.end(), name))
        {
            check.unread.push_back(name);
        }
    }

    for (const std::string &name : list_attribute_names(file, "/model"))
    {
        if (!hdf5_io::is_model_attribute(name))
        {
            check.unknown_attributes.push_back(name);
        }
    }

    // The covariance block's own selection scheme. A model reads it only if its
    // input has somewhere to put it -- the four time-varying models with a
    // covariance block -- and only with that block switched on. Anywhere else
    // the file asks for a selection nothing performs.
    check.psi_varsel_unread = attribute_exists(file, "/model/priors/psi", "varsel") &&
                              !(has_psi_varsel<Input>::value && input.spec.uses_covar());

    // Whether `bvs` has a prior it can select against, per block. The two
    // spellings of the coefficient block's selection prior are the Wishart
    // models' `varsel_prior` and everyone else's `a_varsel_prior`; the
    // covariance block's is read only with that block switched on, and under
    // its own scheme where the model has one.
    if constexpr (requires { input.varsel_prior; })
    {
        collect_flat_selection(check, "a", input.spec.varsel, input.varsel_prior, input.a_prior);
    }
    if constexpr (requires { input.a_varsel_prior; })
    {
        collect_flat_selection(check, "a", input.spec.varsel, input.a_varsel_prior, input.a_prior);
    }
    if constexpr (requires { input.psi_varsel_prior; })
    {
        if (input.spec.uses_covar())
        {
            VarSelection psi_scheme = input.spec.varsel;
            if constexpr (has_psi_varsel<Input>::value)
            {
                psi_scheme = input.psi_varsel;
            }
            collect_flat_selection(check, "psi", psi_scheme, input.psi_varsel_prior,
                                   input.psi_prior);
        }
    }

    return check;
}

/// The checks a VAR's forecast() makes on its regressors, made without draws:
/// require_forecast_regressors(), one row per horizon for update_forecast_lags()
/// to write into, and the width test against `a`. A draw of `a` has one row per
/// column of /data/train/z, so z less the contemporaneous terms the forecast
/// splits off is the width that test will see.
template <typename Input>
void require_var_forecast_regressors(const Input &input)
{
    const VarSpec &spec = input.spec;
    if (spec.h <= 0 || spec.k <= 0)
    {
        return;
    }

    const arma::uword k = static_cast<arma::uword>(spec.k);
    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::mat &x = input.forecast.x;
    const arma::uword n_a = input.train.z.n_cols;
    const arma::uword n_structural = static_cast<arma::uword>(spec.n_structural());

    if (spec.nparams_per_period() > 0 && x.n_elem == 0)
    {
        throw std::invalid_argument(
            "h = " + std::to_string(spec.h) +
            " asks for a forecast, but the file has no forecast regressors: /data/forecast/x is "
            "missing, and a forecast without them would be drawn from the errors alone");
    }

    // The sampler estimates from z, and the forecast writes the lags of its
    // simulated path into the regressors by p. A z of another width either
    // overruns them or, where the widths happen to fit, feeds a lag block into
    // the wrong coefficients.
    if (n_a > 0 && n_a != static_cast<arma::uword>(spec.nparams_per_period()))
    {
        throw std::invalid_argument(
            "/data/train/z has " + std::to_string(n_a) + " columns, but k = " +
            std::to_string(spec.k) + ", p = " + std::to_string(spec.p) + ", m = " +
            std::to_string(spec.m) + ", s = " + std::to_string(spec.s) + ", n = " +
            std::to_string(spec.n) + (spec.structural ? " and structural" : "") + " make " +
            std::to_string(spec.nparams_per_period()) +
            " coefficients per period. The chain would run on z, and the forecast then places "
            "the lags by p");
    }

    if (x.n_elem == 0)
    {
        return;
    }
    if (x.n_rows != h)
    {
        throw std::invalid_argument(
            "/data/forecast/x holds " + std::to_string(x.n_rows) + " horizons, and h = " +
            std::to_string(spec.h) + " needs exactly one row per horizon");
    }
    if (n_a > n_structural && x.n_cols * k != n_a - n_structural)
    {
        throw std::invalid_argument(
            "/data/forecast/x has " + std::to_string(x.n_cols) +
            " regressors per horizon, but /data/train/z's " + std::to_string(n_a) +
            " columns over k = " + std::to_string(k) + " equations" +
            (n_structural > 0 ? ", less " + std::to_string(n_structural) + " contemporaneous terms," : "") +
            " need " + std::to_string((n_a - n_structural) / k));
    }
}

/// The same for a VEC. Its forecast rewrites the draws as the level VAR they
/// imply and hands /data/forecast/x to that VAR's forecast, so the regressors
/// are in the level layout vec_to_var_spec() describes -- p lag blocks of the
/// endogenous variables, the exogenous ones with s lags, and the deterministic
/// terms of both kinds -- and not the differenced one /data/train/z uses. A
/// VEC's z against its dimensions is validate()'s business already.
template <typename Input>
void require_vec_forecast_regressors(const Input &input)
{
    const VarSpec &spec = input.spec;
    if (spec.h <= 0 || spec.k <= 0)
    {
        return;
    }

    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::uword width = static_cast<arma::uword>(
        spec.k * spec.p + spec.m * (spec.s + 1) + spec.n + spec.n_restricted);
    const arma::mat &x = input.forecast.x;

    if (width > 0 && x.n_elem == 0)
    {
        throw std::invalid_argument(
            "h = " + std::to_string(spec.h) +
            " asks for a forecast, but the file has no forecast regressors: /data/forecast/x is "
            "missing. A VEC forecasts in levels, so it needs " + std::to_string(width) +
            " level regressors per horizon");
    }
    if (x.n_elem == 0)
    {
        return;
    }
    if (x.n_rows != h)
    {
        throw std::invalid_argument(
            "/data/forecast/x holds " + std::to_string(x.n_rows) + " horizons, and h = " +
            std::to_string(spec.h) + " needs exactly one row per horizon");
    }
    if (x.n_cols != width)
    {
        throw std::invalid_argument(
            "/data/forecast/x has " + std::to_string(x.n_cols) +
            " regressors per horizon, but a VEC forecasts in levels and needs k*p + m*(s+1) + "
            "n + n_restricted = " + std::to_string(width) +
            " -- the level layout, not the differenced one /data/train/z is in");
    }
}

/// The same for a discounted model, whose regressors are the compact
/// `/data/train/x` and whose forecast applies one k x n_x coefficient matrix
/// per horizon to `/data/forecast/x` in the same layout.
///
/// Counted off the spec throughout. The VAR template above measures the width
/// against `/data/train/z`, which these models do not read and which a file
/// written for one does not carry, so every one of its tests would pass
/// vacuously on a forecast that is about to overrun its regressors.
template <typename Input>
void require_discount_forecast_regressors(const Input &input)
{
    const VarSpec &spec = input.spec;
    if (spec.h <= 0 || spec.k <= 0)
    {
        return;
    }

    const arma::uword h = static_cast<arma::uword>(spec.h);
    const arma::uword width = static_cast<arma::uword>(spec.n_x());
    const arma::mat &x = input.forecast.x;

    if (width > 0 && x.n_elem == 0)
    {
        throw std::invalid_argument(
            "h = " + std::to_string(spec.h) +
            " asks for a forecast, but the file has no forecast regressors: /data/forecast/x is "
            "missing, and a forecast without them would be drawn from the errors alone");
    }
    if (x.n_elem == 0)
    {
        return;
    }
    if (x.n_rows != h)
    {
        throw std::invalid_argument(
            "/data/forecast/x holds " + std::to_string(x.n_rows) + " horizons, and h = " +
            std::to_string(spec.h) + " needs exactly one row per horizon");
    }
    if (x.n_cols != width)
    {
        throw std::invalid_argument(
            "/data/forecast/x has " + std::to_string(x.n_cols) +
            " regressors per horizon, but k = " + std::to_string(spec.k) + ", p = " +
            std::to_string(spec.p) + ", m = " + std::to_string(spec.m) + ", s = " +
            std::to_string(spec.s) + ", n = " + std::to_string(spec.n) + " make " +
            std::to_string(width) + " -- the compact layout /data/train/x is in");
    }
}

/// Opens the model read-only, with the reads recorded, and runs its reader and
/// validate(). `extra` is the stage-specific part.
template <typename ReadInput, typename Extra>
ModelCheck run(const ModelLocation &location, ReadInput read_input, Extra extra,
               const char *posterior_probe = kDefaultPosteriorProbe)
{
    HighFive::File h5 = open_hdf5_file(location.file);
    ModelFile file(h5, location.group);
    file.record_reads();

    const auto input = read_input(file);
    input.validate();
    extra(file, input);

    return inspect(file, input, posterior_probe);
}

} // namespace bayests::model_check_detail

/// A VAR, whose forecast needs regressors in the layout of /data/train/z.
template <typename ReadInput>
ModelCheck check_var_model(const ModelLocation &location, ReadInput read_input)
{
    return bayests::model_check_detail::run(
        location, read_input, [](const ModelFile &, const auto &input) {
            bayests::model_check_detail::require_var_forecast_regressors(input);
        });
}

/// A VEC, whose forecast needs regressors in the layout of its level VAR.
template <typename ReadInput>
ModelCheck check_vec_model(const ModelLocation &location, ReadInput read_input)
{
    return bayests::model_check_detail::run(
        location, read_input, [](const ModelFile &, const auto &input) {
            bayests::model_check_detail::require_vec_forecast_regressors(input);
        });
}

/// A discounted VAR: the compact regressor layout, and a posterior that is a
/// closed form rather than draws, so neither of the two things the VAR template
/// above looks at is where this model keeps it.
template <typename ReadInput>
ModelCheck check_var_discount_model(const ModelLocation &location, ReadInput read_input,
                                    const char *posterior_probe)
{
    return bayests::model_check_detail::run(
        location, read_input,
        [](const ModelFile &, const auto &input) {
            bayests::model_check_detail::require_discount_forecast_regressors(input);
        },
        posterior_probe);
}

/// A discounted VEC: the level forecast layout every VEC needs, against the
/// same closed-form posterior.
template <typename ReadInput>
ModelCheck check_vec_discount_model(const ModelLocation &location, ReadInput read_input,
                                    const char *posterior_probe)
{
    return bayests::model_check_detail::run(
        location, read_input,
        [](const ModelFile &, const auto &input) {
            bayests::model_check_detail::require_vec_forecast_regressors(input);
        },
        posterior_probe);
}

/// A factor model, whose forecast runs on the horizon alone, so there is
/// nothing past validate() to check.
template <typename ReadInput>
ModelCheck check_factor_model(const ModelLocation &location, ReadInput read_input)
{
    return bayests::model_check_detail::run(location, read_input,
                                            [](const ModelFile &, const auto &) {});
}

/// A quantile model: validate() refuses a horizon, and forecast() does nothing
/// whatever /model says, so there is nothing past validate() to check.
template <typename ReadInput>
ModelCheck check_quantile_model(const ModelLocation &location, ReadInput read_input)
{
    return bayests::model_check_detail::run(location, read_input,
                                            [](const ModelFile &, const auto &) {});
}

#endif // BAYESTS_MODELS_MODEL_CHECK_H
