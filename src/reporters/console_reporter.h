// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#ifndef BAYESTS_REPORTERS_CONSOLE_REPORTER_H
#define BAYESTS_REPORTERS_CONSOLE_REPORTER_H

#include "bayests/reporter.h"

namespace bayests
{

/// Draws the command line's progress bar on stdout.
///
/// This is the half of the old sampler that could not travel: an embedded host
/// supplies its own Reporter and never links this. Ctrl-C is left to the
/// terminal, so check_interrupt() stays a no-op.
class ConsoleReporter final : public Reporter
{
public:
    void message(const std::string &text) override;
    void progress(long long done, long long total) override;
    void finish() override;

private:
    bool drew_anything_ = false;
};

} // namespace bayests

#endif // BAYESTS_REPORTERS_CONSOLE_REPORTER_H
