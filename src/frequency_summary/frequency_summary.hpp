#pragma once

#include <cstdint>

class FrequencySummary
{
public:
    virtual ~FrequencySummary() = default;

    virtual void update(uint64_t item) = 0;

    virtual double estimate(uint64_t item) const = 0;

protected:
    FrequencySummary() = default;
};
