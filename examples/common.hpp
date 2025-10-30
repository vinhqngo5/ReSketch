#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

// JSON Library
#include "json/json.hpp"

// Sketch Headers
#include "frequency_summary/count_min_sketch.hpp"
#include "frequency_summary/resketchv2.hpp"

using json = nlohmann::json;

// Timer class to measure execution time
class Timer {
  public:
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    double stop_s() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - m_start).count();
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

// Data generation functions
std::vector<uint64_t> generate_zipf_data(uint64_t size, uint64_t diversity, double a);
std::vector<uint64_t> read_caida_data(const std::string &path, uint64_t max_items);

// Accuracy calculation functions
template <typename SketchType> double calculate_are_all_items(const SketchType &sketch, const std::map<uint64_t, uint64_t> &true_freqs) {
    if (true_freqs.empty()) return 0.0;
    double total_rel_error = 0.0;
    for (const auto &[item, true_freq] : true_freqs) {
        double est_freq = sketch.estimate(item);
        if (true_freq > 0) { total_rel_error += std::abs(est_freq - true_freq) / true_freq; }
    }
    return total_rel_error / true_freqs.size();
}

template <typename SketchType> double calculate_aae_all_items(const SketchType &sketch, const std::map<uint64_t, uint64_t> &true_freqs) {
    if (true_freqs.empty()) return 0.0;
    double total_abs_error = 0.0;
    for (const auto &[item, true_freq] : true_freqs) {
        double est_freq = sketch.estimate(item);
        total_abs_error += std::abs(est_freq - true_freq);
    }
    return total_abs_error / true_freqs.size();
}

// Memory calculation functions
uint32_t calculate_width_from_memory_cm(uint64_t memory_bytes, uint32_t depth);
uint32_t calculate_width_from_memory_resketch(uint64_t memory_bytes, uint32_t depth, uint32_t kll_k);
uint32_t calculate_width_from_memory_geometric(uint64_t memory_bytes, uint32_t depth);
uint32_t calculate_width_from_memory_dynamic(uint64_t memory_bytes, uint32_t depth);

// File utilities
void create_directory(const std::string &path);
