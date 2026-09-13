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
// against, and the front-ends' reading of the horizon. The check.* tests in
// test/CMakeLists.txt run this over every fixture a golden.* run accepts, so a
// rule here that is stricter than the run fails there.

#include "models/models.h"

#include "io/hdf5/hdf5_and_armadillo.h"
#include "io/hdf5/model_io_common.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace bayests::model_check_detail
{

/// Everything the check reports that does not depend on which model it is.
template <typename Input>
ModelCheck inspect(const ModelFile &file, const Input &input)
{
    ModelCheck check;
    check.spec = input.spec;

    // tt is never stored; it is what the reader's validate() recovers too.
    const arma::uword k = input.spec.k > 0 ? static_cast<arma::uword>(input.spec.k) : 0;
    check.periods = k > 0 ? input.train.y.n_elem / k : 0;
    check.z_columns = input.train.z.n_cols;

    // Taken before anything below opens a dataset, so that the list is what
    // the reader asked for and nothing else.
    const std::vector<std::string> read = file.datasets_read();

    check.error_attribute = hdf5_io::optional_attribute_string(file, "/model", "error", "");
    check.has_posterior = dataset_has_data(file, "/posterior/u_sigma_inv/coeffs");

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

    return check;
}

/// The front-ends skip a forecast when /model has no `h` attribute, and hand
/// one that is there to the sampler, which refuses a horizon below one. So a
/// written h = 0 runs the whole chain and then fails -- which is what a file
/// written as "no forecast, please" by anyone who spells that as zero gets. Not
/// the quantile models: their forecast() is a no-op whatever the attribute says.
inline void require_written_horizon(const ModelFile &file, const VarSpec &spec)
{
    if (spec.h <= 0 && attribute_exists(file, "/model", "h"))
    {
        throw std::invalid_argument(
            "h = " + std::to_string(spec.h) +
            " is written, and a written h asks for a forecast: the forecasts stage would refuse "
            "the horizon after the chain has run. To ask for no forecast, leave the attribute out");
    }
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
    if (x.n_rows < h)
    {
        throw std::invalid_argument(
            "/data/forecast/x holds " + std::to_string(x.n_rows) + " horizons, and h = " +
            std::to_string(spec.h) + " needs one per horizon");
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
    if (x.n_rows < h)
    {
        throw std::invalid_argument(
            "/data/forecast/x holds " + std::to_string(x.n_rows) + " horizons, and h = " +
            std::to_string(spec.h) + " needs one per horizon");
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

/// Opens the model read-only, with the reads recorded, and runs its reader and
/// validate(). `extra` is the stage-specific part.
template <typename ReadInput, typename Extra>
ModelCheck run(const ModelLocation &location, ReadInput read_input, Extra extra)
{
    HighFive::File h5 = open_hdf5_file(location.file);
    ModelFile file(h5, location.group);
    file.record_reads();

    const auto input = read_input(file);
    input.validate();
    extra(file, input);

    return inspect(file, input);
}

} // namespace bayests::model_check_detail

/// A VAR, whose forecast needs regressors in the layout of /data/train/z.
template <typename ReadInput>
ModelCheck check_var_model(const ModelLocation &location, ReadInput read_input)
{
    return bayests::model_check_detail::run(
        location, read_input, [](const ModelFile &file, const auto &input) {
            bayests::model_check_detail::require_written_horizon(file, input.spec);
            bayests::model_check_detail::require_var_forecast_regressors(input);
        });
}

/// A VEC, whose forecast needs regressors in the layout of its level VAR.
template <typename ReadInput>
ModelCheck check_vec_model(const ModelLocation &location, ReadInput read_input)
{
    return bayests::model_check_detail::run(
        location, read_input, [](const ModelFile &file, const auto &input) {
            bayests::model_check_detail::require_written_horizon(file, input.spec);
            bayests::model_check_detail::require_vec_forecast_regressors(input);
        });
}

/// A factor model, whose forecast runs on the horizon alone.
template <typename ReadInput>
ModelCheck check_factor_model(const ModelLocation &location, ReadInput read_input)
{
    return bayests::model_check_detail::run(
        location, read_input, [](const ModelFile &file, const auto &input) {
            bayests::model_check_detail::require_written_horizon(file, input.spec);
        });
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
