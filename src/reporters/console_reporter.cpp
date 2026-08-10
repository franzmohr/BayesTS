// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Franz X. Mohr

#include "reporters/console_reporter.h"

#include <cstdio>
#include <string>

namespace bayests
{

void ConsoleReporter::message(const std::string &text)
{
    printf("%s\n", text.c_str());
    fflush(stdout);
}

void ConsoleReporter::progress(long long done, long long total)
{
    if (total <= 0)
    {
        return;
    }

    // Redraw at most once per percent. The samplers call this every draw, and
    // a chain of 12000 draws that repaints the bar 12000 times spends more
    // time in the terminal than in the sampler.
    const bool at_percent_boundary = (total < 100) || (done % (total / 100) == 0);
    if (!at_percent_boundary && done != total)
    {
        return;
    }

    constexpr int bar_width = 50;
    const int percent = static_cast<int>((done * 100) / total);
    const int pos = (bar_width * percent) / 100;

    std::string bar(bar_width, ' ');
    for (int j = 0; j < pos; ++j)
    {
        bar[j] = '=';
    }
    if (pos < bar_width)
    {
        bar[pos] = '>';
    }

    printf("\rProgress: [%s] %3d%% (%lld/%lld)", bar.c_str(), percent, done, total);
    fflush(stdout);
    drew_anything_ = true;
}

void ConsoleReporter::finish()
{
    if (drew_anything_)
    {
        printf("\n");
        fflush(stdout);
        drew_anything_ = false;
    }
}

} // namespace bayests
