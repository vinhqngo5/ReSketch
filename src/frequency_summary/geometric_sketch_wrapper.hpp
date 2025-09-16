#pragma once

#include "frequency_summary.hpp"
#include "frequency_summary_config.hpp"
#include "geometric_sketch/cpp/include/GeometricSketch.h"
#include <memory>
#include <stdexcept>

class GeometricSketchWrapper : public FrequencySummary {
  public:
    explicit GeometricSketchWrapper(const GeometricSketchConfig &config)
        : m_config(config), m_sketch(std::make_unique<GeometricSketch>(config.width, config.depth, config.branching_factor)) {}

    void update(uint64_t item) override { m_sketch->update(static_cast<uint32_t>(item), 1); }

    double estimate(uint64_t item) const override { return static_cast<double>(m_sketch->query(static_cast<uint32_t>(item))); }

    void expand(uint32_t n) { m_sketch->expand(n); }

    void shrink(uint32_t n) { m_sketch->shrink(n); }

    uint64_t get_max_memory_usage() const { return m_sketch->getMemoryUsage(); }

  private:
    GeometricSketchConfig m_config;
    mutable std::unique_ptr<GeometricSketch> m_sketch;
};