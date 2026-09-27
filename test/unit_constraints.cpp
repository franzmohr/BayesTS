// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

/// @file unit_constraints.cpp
/// @brief Checks what a constraint set is refused for, and that every algorithm
/// refuses one.
///
/// A constraint set says what was observed of a panel that is not observed
/// whole: each row is a linear combination of entries of `y`, holding exactly
/// (group zero) or up to an error shared by a soft group. It is the basis of
/// mixed-frequency estimation, where a quarterly figure is one row over the
/// months it aggregates.
///
/// Every algorithm that does not read one has to refuse a file that carries it:
/// the entries a panel does not observe hold placeholders, and a sampler reading
/// `y` whole would estimate a model from them and report it as a fit. That
/// refusal is checked for each of those input types by name. The six Gaussian
/// VARs read a training set, and VarNormalWishart and VarNormalGamma the other
/// two as well; unit.var_mixed_frequency and check.pinned_panel cover what they
/// do with them.
///
/// The rest is validate_constraints(), which a sampler that does read the set
/// relies on, one refusal at a time against a set that is otherwise valid.

#include "bayests/inputs.h"
#include "core/models/constraint_support.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using bayests::Constraints;
using bayests::core::require_supported_constraints;
using bayests::core::validate_constraints;

int failures = 0;

void check(bool condition, const std::string &what)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
        return;
    }
    std::cout << "ok: " << what << '\n';
}

constexpr arma::uword kVariables = 2;
constexpr arma::uword kPeriods = 6;

/// Six months of two series, one row per period: a monthly one observed every
/// month and a quarterly one whose monthly values are placeholders.
arma::mat panel()
{
    arma::mat y(kPeriods, kVariables);
    for (arma::uword t = 0; t < kPeriods; t++)
    {
        y(t, 0) = 1.5 + static_cast<double>(t);
        y(t, 1) = -7.0; // never observed
    }
    return y;
}

/// Variable by period, the layout validate_constraints() is given.
arma::mat by_period() { return arma::trans(panel()); }

/// The monthly series pinned in every month, the quarterly one observed as the
/// average of each quarter's three months, and one soft row: a noisy reading
/// of the first month of the quarterly series.
Constraints valid()
{
    Constraints c;
    const arma::mat y = panel();
    std::vector<double> value, weight;
    std::vector<arma::uword> group, row, period, variable;

    for (arma::uword t = 0; t < kPeriods; t++)
    {
        row.push_back(value.size());
        period.push_back(t);
        variable.push_back(0);
        weight.push_back(1.0);
        value.push_back(y(t, 0));
        group.push_back(0);
    }
    for (arma::uword q = 0; q < 2; q++)
    {
        for (arma::uword m = 0; m < 3; m++)
        {
            row.push_back(value.size());
            period.push_back(3 * q + m);
            variable.push_back(1);
            weight.push_back(1.0 / 3.0);
        }
        value.push_back(0.25 * static_cast<double>(q + 1));
        group.push_back(0);
    }
    row.push_back(value.size());
    period.push_back(0);
    variable.push_back(1);
    weight.push_back(1.0);
    value.push_back(0.1);
    group.push_back(1);

    c.value = arma::vec(value);
    c.weight = arma::vec(weight);
    c.group = arma::conv_to<arma::uvec>::from(group);
    c.row = arma::conv_to<arma::uvec>::from(row);
    c.period = arma::conv_to<arma::uvec>::from(period);
    c.variable = arma::conv_to<arma::uvec>::from(variable);
    return c;
}

bool refused(const Constraints &c, const std::string &contains = "")
{
    try
    {
        validate_constraints(c, by_period(), "/data/train/constraints");
    }
    catch (const std::invalid_argument &e)
    {
        if (!contains.empty() && std::string(e.what()).find(contains) == std::string::npos)
        {
            std::cerr << "  refused, but the message does not say '" << contains
                      << "': " << e.what() << '\n';
            return false;
        }
        return true;
    }
    return false;
}

void test_valid_set_passes()
{
    check(!refused(valid()), "a monthly series, a quarterly one and a soft row pass");

    Constraints none;
    check(none.empty(), "a default set is empty, which is a panel observed whole");
    check(!valid().empty(), "and a set with rows is not");
}

void test_shapes()
{
    Constraints c = valid();
    c.group.shed_row(0);
    check(refused(c, "/group"), "a group per row, or refused");

    c = valid();
    c.weight.shed_row(0);
    check(refused(c, "one element per entry"), "a weight per entry, or refused");

    c = valid();
    c.value(0) = arma::datum::nan;
    check(refused(c, "finite"), "a NaN value is refused as not finite");

    c = valid();
    c.weight(0) = arma::datum::inf;
    check(refused(c, "finite"), "an infinite weight is refused as not finite");
}

void test_ranges()
{
    Constraints c = valid();
    c.row(0) = c.value.n_elem;
    check(refused(c, "/row holds"), "an entry of a row past the last is refused");

    // The messages count from one, as the file does: index 6 is period 7.
    c = valid();
    c.period(6) = kPeriods;
    check(refused(c, "/period holds 7"), "a period past the sample is refused, counted from one");

    c = valid();
    c.variable(6) = kVariables;
    check(refused(c, "/variable holds 3"), "a variable past the panel is refused, counted from one");
}

void test_rows_that_observe_nothing()
{
    // A row with no entries: the soft row's one entry moved to another row.
    Constraints c = valid();
    c.row(c.row.n_elem - 1) = 6;
    check(refused(c, "observes nothing"), "a row with no entry is refused");

    c = valid();
    c.weight.subvec(6, 8).zeros();
    check(refused(c, "observes nothing"), "a row whose weights are all zero is refused");
}

void test_duplicates()
{
    Constraints c = valid();
    c.period(7) = c.period(6);
    check(refused(c, "twice"), "one entry named twice in a row is refused");
}

void test_groups()
{
    Constraints c = valid();
    c.group(c.group.n_elem - 1) = 2;
    check(refused(c, "none left out"), "soft groups with a gap are refused");

    c = valid();
    c.group(6) = 1;
    check(!refused(c), "a quarterly row may be soft, sharing the group of another");
}

void test_pins()
{
    Constraints c = valid();
    c.value(2) += 1.0;
    check(refused(c, "wrong period"), "a pinned value that disagrees with y is refused");

    c = valid();
    c.value(2) *= 1.0 + 1e-13;
    check(!refused(c), "and one that agrees to a rounding error is not");

    c = valid();
    c.weight(2) = 2.0;
    c.value(2) *= 2.0;
    check(!refused(c), "a pin is value over weight, so a weight other than one is read as such");

    // Two hard rows on one entry: the soft row made hard and moved onto a
    // monthly entry, which the first six rows already pin.
    c = valid();
    const arma::uword soft = c.value.n_elem - 1;
    c.group(soft) = 0;
    c.variable(c.row.n_elem - 1) = 0;
    c.value(soft) = panel()(0, 0);
    check(refused(c, "two hard rows"), "an entry pinned twice is refused");

    // A hard aggregation over pinned entries only: the first quarter's row
    // pointed at the monthly series.
    c = valid();
    c.variable.subvec(6, 8).zeros();
    check(refused(c, "implied by them or contradicts them"),
          "a hard row reaching only pinned entries is refused");

    // The same row soft is a reading with an error, and stays.
    c.group(6) = 1;
    check(!refused(c), "and the same row soft is not");
}

void test_supported_checks_both_layouts()
{
    bayests::VarSpec spec;
    spec.k = static_cast<int>(kVariables);

    bayests::TrainData train;
    train.y = arma::vectorise(arma::trans(panel())); // the stacked spelling
    train.constraints = valid();
    bayests::TestData test;

    bool ok = true;
    try
    {
        require_supported_constraints(spec, train, test, true, "a sampler that reads them");
    }
    catch (const std::exception &e)
    {
        std::cerr << "  " << e.what() << '\n';
        ok = false;
    }
    check(ok, "a supported set is checked against y stacked, and passes");

    train.constraints.value(0) += 1.0;
    bool caught = false;
    try
    {
        require_supported_constraints(spec, train, test, true, "a sampler that reads them");
    }
    catch (const std::invalid_argument &)
    {
        caught = true;
    }
    check(caught, "and is refused when a pin disagrees with it");

    // The test constraints against /data/test/y, one row per period.
    train.constraints = Constraints();
    test.y = panel();
    test.constraints = valid();
    test.constraints.value(0) += 1.0;
    caught = false;
    try
    {
        require_supported_constraints(spec, train, test, true, "a sampler that reads them");
    }
    catch (const std::invalid_argument &e)
    {
        caught = std::string(e.what()).find("/data/test/constraints") != std::string::npos;
    }
    check(caught, "the test set is checked against /data/test/y, and named as the test set");
}

/// validate() refuses a constraint set before it reads anything else, so a
/// default input carrying one is enough: whatever else it lacks is never
/// reached.
template <typename Input>
void check_refuses(const std::string &algorithm)
{
    for (const bool on_test : {false, true})
    {
        Input input;
        input.spec.k = static_cast<int>(kVariables);
        input.train.y = panel();
        if (on_test)
        {
            input.test.y = panel();
            input.test.constraints = valid();
        }
        else
        {
            input.train.constraints = valid();
        }

        std::string message;
        try
        {
            input.validate();
        }
        catch (const std::exception &e)
        {
            message = e.what();
        }
        check(message.find(algorithm + " does not read /data/train/constraints") !=
                  std::string::npos,
              algorithm + " refuses " + (on_test ? "test" : "training") + " constraints by name");
    }
}

/// A scenario, /data/forecast/constraints, on an algorithm that does not
/// condition its forecast: one pin on the first period of a two-period horizon.
/// Only the inputs that carry forecast data can be handed one; a factor model's
/// reader refuses the group instead, which unit.constraints_io covers.
template <typename Input>
void check_refuses_scenario(const std::string &algorithm)
{
    Input input;
    input.spec.k = static_cast<int>(kVariables);
    input.spec.h = 2;
    input.train.y = panel();
    Constraints scenario;
    scenario.value = {0.5};
    scenario.group = {0};
    scenario.row = {0};
    scenario.period = {0};
    scenario.variable = {0};
    scenario.weight = {1.0};
    input.forecast.constraints = scenario;

    std::string message;
    try
    {
        input.validate();
    }
    catch (const std::exception &e)
    {
        message = e.what();
    }
    check(message.find(algorithm + " does not read /data/forecast/constraints") != std::string::npos,
          algorithm + " refuses a scenario by name");
}

void test_every_algorithm_refuses()
{
    check_refuses<bayests::VarNormalAldInput>("VarNormalAld");
    check_refuses<bayests::VarTvpAldInput>("VarTvpAld");
    check_refuses<bayests::VarTvpDiscountInput>("VarTvpDiscount");
    check_refuses<bayests::VecNormalWishartInput>("VecNormalWishart");
    check_refuses<bayests::VecNormalGammaInput>("VecNormalGamma");
    check_refuses<bayests::VecNormalStochvolInput>("VecNormalStochvol");
    check_refuses<bayests::VecTvpWishartInput>("VecTvpWishart");
    check_refuses<bayests::VecTvpGammaInput>("VecTvpGamma");
    check_refuses<bayests::VecTvpStochvolInput>("VecTvpStochvol");
    check_refuses<bayests::VecTvpDiscountInput>("VecTvpDiscount");
    check_refuses<bayests::VecKlgs2010Input>("VecKlgs2010");
    check_refuses<bayests::DfmNormalGammaInput>("DfmNormalGamma");
    check_refuses<bayests::DfmNormalStochvolInput>("DfmNormalStochvol");
    check_refuses<bayests::DfmTvpGammaInput>("DfmTvpGamma");
    check_refuses<bayests::DfmTvpStochvolInput>("DfmTvpStochvol");
    check_refuses<bayests::FavarNormalWishartInput>("FavarNormalWishart");

    check_refuses_scenario<bayests::VarNormalStochvolInput>("VarNormalStochvol");
    check_refuses_scenario<bayests::VarNormalAldInput>("VarNormalAld");
    check_refuses_scenario<bayests::VarTvpWishartInput>("VarTvpWishart");
    check_refuses_scenario<bayests::VarTvpGammaInput>("VarTvpGamma");
    check_refuses_scenario<bayests::VarTvpStochvolInput>("VarTvpStochvol");
    check_refuses_scenario<bayests::VarTvpAldInput>("VarTvpAld");
    check_refuses_scenario<bayests::VarTvpDiscountInput>("VarTvpDiscount");
    check_refuses_scenario<bayests::VecNormalWishartInput>("VecNormalWishart");
    check_refuses_scenario<bayests::VecNormalGammaInput>("VecNormalGamma");
    check_refuses_scenario<bayests::VecNormalStochvolInput>("VecNormalStochvol");
    check_refuses_scenario<bayests::VecTvpWishartInput>("VecTvpWishart");
    check_refuses_scenario<bayests::VecTvpGammaInput>("VecTvpGamma");
    check_refuses_scenario<bayests::VecTvpStochvolInput>("VecTvpStochvol");
    check_refuses_scenario<bayests::VecTvpDiscountInput>("VecTvpDiscount");
    check_refuses_scenario<bayests::VecKlgs2010Input>("VecKlgs2010");
}

} // namespace

int main()
{
    try
    {
        test_valid_set_passes();
        test_shapes();
        test_ranges();
        test_rows_that_observe_nothing();
        test_duplicates();
        test_groups();
        test_pins();
        test_supported_checks_both_layouts();
        test_every_algorithm_refuses();
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: threw: " << e.what() << '\n';
        ++failures;
    }

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "all checks passed\n";
    return 0;
}
