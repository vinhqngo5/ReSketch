#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "frequency_summary.hpp"
#include "frequency_summary_config.hpp"
#include "hash/xxhash64.hpp"

#define LONG_PRIME 2147483647

class CountMinSketch : public FrequencySummary {
  public:
    explicit CountMinSketch(const CountMinConfig &config) : m_config(config) { _initialize_from_config(); }

    void update(uint64_t item) override {
        for (uint32_t i = 0; i < m_depth; ++i) {
            uint64_t hash_val = _hash(item, i);
            m_table[i][hash_val % m_width]++;
        }
    }

    double estimate(uint64_t item) const override {
        uint32_t min_count = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0; i < m_depth; ++i) {
            uint64_t hash_val = _hash(item, i);
            min_count = std::min(min_count, m_table[i][hash_val % m_width]);
        }
        return static_cast<double>(min_count);
    }

    void merge(const CountMinSketch &other) {
        if (m_width != other.m_width || m_depth != other.m_depth) { throw std::invalid_argument("Cannot merge Count-Min sketches with different dimensions."); }

        for (unsigned int i = 0; i < m_depth; ++i) {
            for (unsigned int j = 0; j < m_width; ++j) { m_table[i][j] += other.m_table[i][j]; }
        }
    }

    uint32_t get_max_memory_usage() const {
        uint32_t table_memory = m_depth * m_width * sizeof(uint32_t);

        return table_memory;
    }

    static uint32_t calculate_max_width(uint32_t total_memory_bytes, uint32_t depth) {
        if (depth == 0) return 0;

        uint32_t max_counters = total_memory_bytes / sizeof(uint32_t);
        return static_cast<uint32_t>(max_counters / depth);
    }

  private:
    void _initialize_from_config() {
        if (m_config.calculate_from == "EPSILON_DELTA") {
            m_width = static_cast<uint32_t>(std::ceil(M_E / m_config.epsilon));
            m_depth = static_cast<uint32_t>(std::ceil(std::log(1.0 / m_config.delta)));
        } else if (m_config.calculate_from == "WIDTH_DEPTH") {
            m_width = m_config.width;
            m_depth = m_config.depth;
        } else {
            throw std::invalid_argument("Invalid 'calculate_from' value in CountMinConfig.");
        }

        m_table.assign(m_depth, std::vector<uint32_t>(m_width, 0));

        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint32_t> dist;
        m_hash_a.resize(m_depth);
        m_hash_b.resize(m_depth);
        for (uint32_t i = 0; i < m_depth; ++i) {
            // For a*x + b, 'a' must be non-zero (and ideally odd).
            m_hash_a[i] = dist(rng) | 1;   // Ensure 'a' is odd and non-zero
            m_hash_b[i] = dist(rng);
        }
    }

    // pair-wise hash functions
    uint64_t _hash(uint64_t item, uint32_t row_index) const { return (m_hash_a[row_index] * item + m_hash_b[row_index]) % LONG_PRIME; }

    CountMinConfig m_config;
    uint32_t m_width;
    uint32_t m_depth;
    std::vector<std::vector<uint32_t>> m_table;

    // for pair-wise hash functions
    std::vector<uint64_t> m_hash_a;
    std::vector<uint64_t> m_hash_b;
};