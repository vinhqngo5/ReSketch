#pragma once

#include <cstdint>
#include <vector>

class QuantileSummary;

class QuantileSummary {
  public:
    virtual ~QuantileSummary() = default;

    virtual void update(uint64_t item) = 0;

    virtual void merge(const QuantileSummary &other) = 0;

    virtual double get_rank(uint64_t value) const = 0;

  protected:
    QuantileSummary() = default;
};