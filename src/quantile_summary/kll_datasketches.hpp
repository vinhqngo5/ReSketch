#pragma once

#include "quantile_summary_config.hpp"

#include "frequency_summary/frequency_summary.hpp"
#include "quantile_summary.hpp"

#include <kll/kll_sketch.hpp>

#include <functional>

// Adapter class that inherits from both QuantileSummary and FrequencySummary and uses Apache DataSketches KLL internally
class KLL : public QuantileSummary, public FrequencySummary
{
public:
    explicit KLL(const KLLConfig &config) : m_config(config), m_sketch(static_cast<uint16_t>(config.k)) {}

    KLL() : m_config({30}), m_sketch(30) {}

    // Copy constructor
    KLL(const KLL &other) : m_config(other.m_config), m_sketch(other.m_sketch) {}

    KLL &operator=(const KLL &other)
    {
        if (this == &other) return *this;
        m_config = other.m_config;
        m_sketch = other.m_sketch;
        return *this;
    }

    // Move constructor and assignment
    KLL(KLL &&other) noexcept = default;
    KLL &operator=(KLL &&other) noexcept = default;

    // QuantileSummary interface
    void update(uint64_t item) override { m_sketch.update(item); }

    void merge(const QuantileSummary &other) override
    {
        const auto *other_kll = dynamic_cast<const KLL *>(&other);
        if (!other_kll) { throw std::invalid_argument("Can only merge KLL with another KLL."); }
        merge(*other_kll);
    }

    void merge(const KLL &other_kll)
    {
        if (m_config.k != other_kll.m_config.k) { throw std::invalid_argument("KLL sketches must have the same k parameter to be merged."); }
        m_sketch.merge(other_kll.m_sketch);
    }

    double get_rank(uint64_t value) const override
    {
        if (m_sketch.is_empty()) return 0.0;
        return m_sketch.get_rank(value);
    }

    // FrequencySummary interface
    double estimate(uint64_t item) const override { return m_sketch.estimate(item); }

    // Additional methods
    const KLLConfig &get_config() const { return m_config; }

    double get_count_in_range(uint64_t start_h, uint64_t end_h) const { return m_sketch.get_count_in_range(start_h, end_h); }

    KLL rebuild(uint64_t start_h, uint64_t end_h) const
    {
        KLL new_kll(m_config);
        new_kll.m_sketch = m_sketch.rebuild(start_h, end_h);
        return new_kll;
    }

    void for_each_summarized_item(const std::function<void(uint64_t item, uint64_t weight)> &func) const { m_sketch.for_each_summarized_item(func); }

    // Construct KLL from weighted items without intermediate compaction
    static KLL construct_from_weighted_items(const std::vector<std::pair<uint64_t, uint64_t>> &weighted_items, const KLLConfig &config)
    {
        KLL result(config);
        result.m_sketch = datasketches::kll_sketch<uint64_t>::construct_from_weighted_items(weighted_items, static_cast<uint16_t>(config.k));
        return result;
    }

    // NOTE: actually apache datasketches KLL does not provide a way to set c explicitly
    uint32_t get_max_memory_usage() const
    {
        // The total number of items stored across all compactors is bounded by ~3*k.
        // This comes from the sum of the geometric series of capacities: k / (1 - c).
        // For c = 2/3, this is k / (1/3) = 3k.
        uint32_t max_stored_items = static_cast<uint32_t>(std::ceil(m_config.k / (1.0 - 2.0 / 3.0)));
        return max_stored_items * sizeof(uint32_t);   // Assuming each item is stored as a 32-bit integer without changing all the types
    }

    static uint32_t calculate_max_k(uint32_t total_memory_bytes, double c = 2.0 / 3.0)
    {
        const uint32_t item_size = sizeof(uint32_t);
        if (total_memory_bytes < item_size || (1.0 - c) <= 0) { return 0; }

        uint32_t max_storable_items = total_memory_bytes / item_size;
        double k = static_cast<double>(max_storable_items) * (1.0 - c);

        return static_cast<uint32_t>(std::floor(k));
    }

    // Access to underlying sketch
    const datasketches::kll_sketch<uint64_t> &get_sketch() const { return m_sketch; }
    datasketches::kll_sketch<uint64_t> &get_sketch() { return m_sketch; }

    bool is_empty() const { return m_sketch.is_empty(); }
    uint64_t get_n() const { return m_sketch.get_n(); }
    uint32_t get_k() const { return m_sketch.get_k(); }
    uint32_t get_num_retained() const { return m_sketch.get_num_retained(); }
    uint8_t get_num_levels() const { return m_sketch.get_num_levels(); }

    friend std::ostream &operator<<(std::ostream &os, const KLL &kll)
    {
        os << "KLL Sketch (Apache DataSketches):" << std::endl;
        os << "  k: " << kll.m_config.k << std::endl;
        os << "  count: " << kll.m_sketch.get_n() << std::endl;
        os << "  num_levels: " << static_cast<int>(kll.m_sketch.get_num_levels()) << std::endl;
        return os;
    }

    // For debugging
    int count_compress = 0;
    double total_compress_time = 0.0;

private:
    KLLConfig m_config;
    datasketches::kll_sketch<uint64_t> m_sketch;
};
