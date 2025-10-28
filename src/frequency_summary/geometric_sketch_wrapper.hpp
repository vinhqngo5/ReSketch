#pragma once

#include "frequency_summary.hpp"
#include "frequency_summary_config.hpp"
#include "geometric_sketch/cpp/include/GeometricSketch.h"
#include <memory>
#include <stdexcept>

class GeometricSketchWrapper : public FrequencySummary {
  public:
    explicit GeometricSketchWrapper(const GeometricSketchConfig &config)
        : m_config(config), m_virtual_width(config.width), m_sketch(std::make_unique<GeometricSketch>(config.width, config.depth, config.branching_factor)) {}

    void update(uint64_t item) override { m_sketch->update(static_cast<uint32_t>(item), 1); }

    double estimate(uint64_t item) const override { return static_cast<double>(m_sketch->query(static_cast<uint32_t>(item))); }

    void expand(uint32_t new_width) {
        if (new_width <= m_virtual_width) { throw std::invalid_argument("New width must be larger than current width."); }
        uint32_t width_increment = new_width - m_virtual_width;
        uint32_t counters_to_add = width_increment * m_config.depth;
        m_sketch->expand(counters_to_add);
        m_virtual_width = new_width;
    }

    void shrink(uint32_t new_width) {
        if (new_width >= m_virtual_width) { throw std::invalid_argument("New width must be smaller than current width."); }
        uint32_t width_decrement = m_virtual_width - new_width;
        uint32_t counters_to_remove = width_decrement * m_config.depth;
        m_sketch->shrink(counters_to_remove);
        m_virtual_width = new_width;
    }

    uint64_t get_max_memory_usage() const { return m_sketch->getMemoryUsage(); }

    // Note: This is identical to CountMinSketch::calculate_max_width. Only use this for calculating initial width.
    static uint32_t calculate_max_width(uint32_t total_memory_bytes, uint32_t depth) {
        if (depth == 0) return 0;

        uint32_t max_counters = total_memory_bytes / sizeof(uint32_t);
        return static_cast<uint32_t>(max_counters / depth);
    }

  private:
    GeometricSketchConfig m_config;
    uint32_t m_virtual_width;
    mutable std::unique_ptr<GeometricSketch> m_sketch;
};