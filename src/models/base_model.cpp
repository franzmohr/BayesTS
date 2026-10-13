// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "models.h"
#include "reporters/console_reporter.h"

bayests::Reporter &BaseModel::stage_reporter()
{
    if (host_reporter_ != nullptr)
    {
        return *host_reporter_;
    }

    // A fresh bar per stage, as each front-end used to construct its own.
    own_reporter_ = std::make_unique<bayests::ConsoleReporter>();
    return *own_reporter_;
}

bayests::Reporter &BaseModel::quiet_reporter()
{
    if (host_reporter_ != nullptr)
    {
        return *host_reporter_;
    }

    own_reporter_ = std::make_unique<bayests::NullReporter>();
    return *own_reporter_;
}
