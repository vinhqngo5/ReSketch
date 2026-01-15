#pragma once

#include "frequency_summary_config.hpp"

#include "frequency_summary.hpp"

#include "geometric_sketch/cpp/include/DynamicSketch.h"

#include <memory>
#include <stdexcept>

class DynamicSketchWrapper : public FrequencySummary
{
public:
    explicit DynamicSketchWrapper(const DynamicSketchConfig &config)
        : m_config(config), m_virtual_width(config.width), m_sketch(std::make_unique<DynamicSketch>(config.width, config.depth, config.is_same_seed))
    {
    }

    void update(uint64_t item) override { m_sketch->update(static_cast<uint32_t>(item), 1); }

    double estimate(uint64_t item) const override { return static_cast<double>(m_sketch->query(static_cast<uint32_t>(item))); }

    void expand(uint32_t new_width)
    {
        if (new_width <= m_virtual_width) { throw std::invalid_argument("New width must be larger than current width."); }
        uint32_t width_to_add = new_width - m_virtual_width;
        m_sketch->expand(width_to_add);
        m_virtual_width = new_width;
    }

    void shrink(uint32_t new_width)
    {
        if (new_width >= m_virtual_width) { throw std::invalid_argument("New width must be smaller than current width."); }
        uint32_t n = m_virtual_width - new_width;
        m_sketch->shrink(n);
        m_virtual_width = new_width;
    }

    uint64_t get_max_memory_usage() const { return m_sketch->getMemoryUsage(); }

    // Note: This is identical to CountMinSketch::calculate_max_width. Only use this for calculating initial width.
    static uint32_t calculate_max_width(uint32_t total_memory_bytes, uint32_t depth)
    {
        if (depth == 0) return 0;

        uint32_t max_counters = total_memory_bytes / sizeof(uint32_t);
        return static_cast<uint32_t>(max_counters / depth);
    }

private:
    DynamicSketchConfig m_config;
    uint32_t m_virtual_width;
    mutable std::unique_ptr<DynamicSketch> m_sketch;
};
