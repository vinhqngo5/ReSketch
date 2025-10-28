#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "frequency_summary/frequency_summary.hpp"
#include "quantile_summary.hpp"
#include "quantile_summary_config.hpp"

class KLL : public QuantileSummary, public FrequencySummary {
  public:
    explicit KLL(const KLLConfig &config) : m_config(config), m_n(0), m_c(2.0 / 3.0), m_rng(std::random_device{}()), m_dist(0.5) { m_compactors.emplace_back(); }

    KLL() : m_n(0), m_c(2.0 / 3.0), m_rng(std::random_device{}()), m_dist(0.5) { m_config.k = 0; }

    // Copy constructor and assignment
    KLL(const KLL &other) : m_config(other.m_config), m_n(other.m_n), m_c(other.m_c), m_compactors(other.m_compactors), m_rng(std::random_device{}()), m_dist(other.m_dist) {}
    KLL &operator=(const KLL &other) {
        if (this == &other) return *this;
        m_config = other.m_config;
        m_n = other.m_n;
        m_compactors = other.m_compactors;
        m_dist = other.m_dist;
        return *this;
    }

    // Move constructor and assignment
    KLL(KLL &&other) noexcept = default;
    KLL &operator=(KLL &&other) noexcept {
        if (this != &other) {
            m_config = std::move(other.m_config);
            m_n = std::move(other.m_n);
            m_compactors = std::move(other.m_compactors);
            m_rng = std::move(other.m_rng);
            m_dist = std::move(other.m_dist);
        }
        return *this;
    }

    void update(uint64_t item) override {
        m_compactors[0].push_back(item);
        m_n++;
        if (m_compactors[0].size() >= _get_level_capacity(0)) { _compress(0); }
    }

    void merge(const QuantileSummary &other) override {
        const auto *other_kll = dynamic_cast<const KLL *>(&other);
        if (!other_kll) { throw std::invalid_argument("Can only merge KLL with another KLL."); }
        merge(*other_kll);
    }

    void merge(const KLL &other_kll) {
        if (m_config.k != other_kll.m_config.k) { throw std::invalid_argument("KLL sketches must have the same k parameter to be merged."); }
        m_n += other_kll.m_n;
        uint32_t max_level = std::max(m_compactors.size(), other_kll.m_compactors.size());
        if (m_compactors.size() < max_level) m_compactors.resize(max_level);

        for (uint32_t i = 0; i < other_kll.m_compactors.size(); ++i) {
            m_compactors[i].insert(m_compactors[i].end(), other_kll.m_compactors[i].begin(), other_kll.m_compactors[i].end());
        }

        for (uint32_t i = 0; i < m_compactors.size(); ++i) {
            if (m_compactors[i].size() >= _get_level_capacity(i)) { _compress(i); }
        }
    }

    double get_rank(uint64_t value) const override {
        double rank = 0.0;
        for (uint32_t i = 0; i < m_compactors.size(); ++i) {
            uint64_t weight = 1ULL << i;
            uint64_t count_le = std::count_if(m_compactors[i].begin(), m_compactors[i].end(), [value](uint64_t item) { return item <= value; });
            rank += static_cast<double>(count_le) * weight;
        }
        return rank;
    }

    double estimate(uint64_t item) const override {
        double estimated_count = 0.0;
        for (uint32_t i = 0; i < m_compactors.size(); ++i) {
            uint64_t weight = 1ULL << i;
            uint64_t count_eq = std::count(m_compactors[i].begin(), m_compactors[i].end(), item);
            estimated_count += static_cast<double>(count_eq) * weight;
        }
        return estimated_count;
    }

    const KLLConfig &get_config() const { return m_config; }

    double get_count_in_range(uint64_t start_h, uint64_t end_h) const {
        double estimated_count = 0.0;
        for (uint32_t i = 0; i < m_compactors.size(); ++i) {
            uint64_t weight = 1ULL << i;
            uint64_t count_in_range = std::count_if(m_compactors[i].begin(), m_compactors[i].end(), [start_h, end_h](uint64_t h) { return h > start_h && h <= end_h; });
            estimated_count += static_cast<double>(count_in_range) * weight;
        }
        return estimated_count;
    }

    KLL rebuild(uint64_t start_h, uint64_t end_h) const {
        KLL new_sketch(m_config);
        if (!m_compactors.empty()) new_sketch.m_compactors.resize(m_compactors.size());

        for (uint32_t i = 0; i < m_compactors.size(); ++i) {
            uint64_t weight = 1ULL << i;
            for (const auto &item : m_compactors[i]) {
                if (item > start_h && item <= end_h) {
                    new_sketch.m_compactors[i].push_back(item);
                    new_sketch.m_n += weight;
                }
            }
        }
        return new_sketch;
    }

    // Iterate over all summarized items (used for Split())
    void for_each_summarized_item(const std::function<void(uint64_t item, uint64_t weight)> &func) const {
        for (uint32_t i = 0; i < m_compactors.size(); ++i) {
            if (m_compactors[i].empty()) continue;

            uint64_t weight = 1ULL << i;
            for (const auto &item_value : m_compactors[i]) { func(item_value, weight); }
        }
    }

    // Update the sketch with a new item and its weight -> Need to double check the correctness later but it should be correct
    // compress = False is used in Split() to avoid repeated compressions
    void update(uint64_t item, uint64_t weight, bool compress = true) {
        if (weight == 0) return;

        m_n += weight;
        uint32_t level = 0;
        while (weight > 0) {
            if (weight & 1) {
                if (level >= m_compactors.size()) { m_compactors.resize(level + 1); }
                m_compactors[level].push_back(item);
            }
            weight >>= 1;
            level++;
        }

        // After adding, check all levels for potential compressions
        // Need to compress multiple levels since the first level might not be overflowed but higher levels might be
        if (compress) {
            for (uint32_t i = 0; i < m_compactors.size(); ++i) {
                if (m_compactors[i].size() >= _get_level_capacity(i)) { _compress(i); }
            }
        }
    }

    uint32_t get_max_memory_usage() const {
        // The total number of items stored across all compactors is bounded by ~3*k.
        // This comes from the sum of the geometric series of capacities: k / (1 - c).
        // For c = 2/3, this is k / (1/3) = 3k.
        uint32_t max_stored_items = static_cast<uint32_t>(std::ceil(m_config.k / (1.0 - m_c)));

        // return max_stored_items * sizeof(uint64_t);
        return max_stored_items * sizeof(uint32_t);   // Assuming each item is stored as a 32-bit integer without changing all the types
    }

    friend std::ostream &operator<<(std::ostream &os, const KLL &kll) {
        os << "KLL Sketch:" << std::endl;
        os << "  k: " << kll.m_config.k << std::endl;
        os << "  count: " << kll.m_n << std::endl;
        os << "  levels: " << kll.m_compactors.size() << std::endl;

        for (uint32_t i = 0; i < kll.m_compactors.size(); ++i) {
            os << "  Level " << i << ":" << std::endl;
            os << "    capacity: " << kll._get_level_capacity(i) << std::endl;
            os << "    size: " << kll.m_compactors[i].size() << std::endl;

            // Print items (limited to first few if there are many)
            if (!kll.m_compactors[i].empty()) {
                os << "    items: ";
                const uint32_t max_items_to_show = 10;
                for (uint32_t j = 0; j < std::min<std::uint32_t>(kll.m_compactors[i].size(), max_items_to_show); ++j) { os << kll.m_compactors[i][j] << " "; }
                if (kll.m_compactors[i].size() > max_items_to_show) { os << "... (" << kll.m_compactors[i].size() - max_items_to_show << " more)"; }
                os << std::endl;
            }
        }

        return os;
    }

    int count_compress = 0;   // For debugging purposes, count how many times compression has been called
                              // calculate time taken for compression
    double total_compress_time = 0.0;

  private:
    uint32_t _get_level_capacity(uint32_t level) const {
        if (m_config.k == 0) return std::numeric_limits<uint32_t>::max();
        return static_cast<uint32_t>(std::ceil(m_config.k * std::pow(m_c, static_cast<double>(m_compactors.size() - level - 1))));
    }

    void _compress_old(uint32_t level) {
        count_compress++;
        auto start_time = std::chrono::high_resolution_clock::now();
        if (level >= m_compactors.size() || m_compactors[level].size() < _get_level_capacity(level)) { return; }

        std::sort(m_compactors[level].begin(), m_compactors[level].end());

        std::vector<uint64_t> keepers;
        keepers.reserve(m_compactors[level].size() / 2 + 1);

        for (const auto &item : m_compactors[level]) {
            if (m_dist(m_rng)) { keepers.push_back(item); }
        }

        m_compactors[level].clear();

        if (level + 1 >= m_compactors.size()) { m_compactors.emplace_back(); }
        m_compactors[level + 1].insert(m_compactors[level + 1].end(), keepers.begin(), keepers.end());

        auto end_time = std::chrono::high_resolution_clock::now();
        total_compress_time += std::chrono::duration<double>(end_time - start_time).count();

        if (m_compactors[level + 1].size() >= _get_level_capacity(level + 1)) { _compress(level + 1); }
    }

    void _compress(uint32_t level) {
        count_compress++;
        auto start_time = std::chrono::high_resolution_clock::now();

        // The level to compress must exist and be full.
        if (level >= m_compactors.size() || m_compactors[level].size() < _get_level_capacity(level)) { return; }

        if (level + 1 >= m_compactors.size()) { m_compactors.emplace_back(); }

        std::sort(m_compactors[level].begin(), m_compactors[level].end());

        auto &source_compactor = m_compactors[level];

        // Randomly selecting -> need to double check the guarantee later (I think it should be more correct)
        uint32_t rand_offset = m_dist(m_rng) ? 1 : 0;

        // inplace compacting
        uint32_t keeper_count = 0;
        for (uint32_t i = rand_offset; i < source_compactor.size(); i += 2) {
            if (i != keeper_count) { source_compactor[keeper_count] = source_compactor[i]; }
            keeper_count++;
        }
        source_compactor.resize(keeper_count);

        auto &target_compactor = m_compactors[level + 1];

        target_compactor.insert(target_compactor.end(), source_compactor.begin(), source_compactor.end());

        source_compactor.clear();

        auto end_time = std::chrono::high_resolution_clock::now();
        total_compress_time += std::chrono::duration<double>(end_time - start_time).count();

        if (m_compactors[level + 1].size() >= _get_level_capacity(level + 1)) { _compress(level + 1); }
    }

    KLLConfig m_config;
    uint64_t m_n;
    const double m_c;
    std::vector<std::vector<uint64_t>> m_compactors;

    mutable std::mt19937 m_rng;
    std::bernoulli_distribution m_dist;
};