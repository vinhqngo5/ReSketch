#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
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

// Frequency analysis functions
std::map<uint64_t, uint64_t> get_true_freqs(const std::vector<uint64_t> &data);
std::vector<uint64_t> get_top_k_items(const std::map<uint64_t, uint64_t> &freqs, int k);
std::vector<uint64_t> get_random_items(const std::map<uint64_t, uint64_t> &freqs, int count);

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

// Variance calculation functions
template <typename SketchType> double calculate_are_variance(const SketchType &sketch, const std::map<uint64_t, uint64_t> &true_freqs, double mean_are) {
    if (true_freqs.empty()) return 0.0;
    double sum_sq = 0.0;
    for (const auto &[item, true_freq] : true_freqs) {
        double est_freq = sketch.estimate(item);
        double rel_error = (true_freq > 0) ? (std::abs(est_freq - true_freq) / true_freq) : 0.0;
        sum_sq += (rel_error - mean_are) * (rel_error - mean_are);
    }
    return sum_sq / true_freqs.size();
}

template <typename SketchType> double calculate_aae_variance(const SketchType &sketch, const std::map<uint64_t, uint64_t> &true_freqs, double mean_aae) {
    if (true_freqs.empty()) return 0.0;
    double sum_sq = 0.0;
    for (const auto &[item, true_freq] : true_freqs) {
        double est_freq = sketch.estimate(item);
        double abs_error = std::abs(est_freq - true_freq);
        sum_sq += (abs_error - mean_aae) * (abs_error - mean_aae);
    }
    return sum_sq / true_freqs.size();
}

// Frequency comparison printing
// Each sketch is passed with a name, and the function will print a comparison table with true frequency and estimated frequency from each sketch for the given items
template <typename... SketchTypes>
void print_frequency_comparison(const std::string &title, const std::vector<uint64_t> &items, const std::map<uint64_t, uint64_t> &true_freqs,
                                const std::vector<std::string> &sketch_names, const SketchTypes &...sketches);

// Helper for the print_frequency_comparison variadic template
template <typename SketchType, typename... RestSketchTypes>
void print_frequency_comparison_impl(uint64_t item, const std::map<uint64_t, uint64_t> &true_freqs, const std::vector<std::string> &sketch_names, size_t idx,
                                     const SketchType &sketch, const RestSketchTypes &...rest);

void print_frequency_comparison_impl(uint64_t item, const std::map<uint64_t, uint64_t> &true_freqs, const std::vector<std::string> &sketch_names, size_t idx);

// Memory calculation functions
uint32_t calculate_width_from_memory_cm(uint64_t memory_bytes, uint32_t depth);
uint32_t calculate_width_from_memory_resketch(uint64_t memory_bytes, uint32_t depth, uint32_t kll_k);
uint32_t calculate_width_from_memory_geometric(uint64_t memory_bytes, uint32_t depth);
uint32_t calculate_width_from_memory_dynamic(uint64_t memory_bytes, uint32_t depth);

// File utilities
void create_directory(const std::string &path);

// Template implementations
template <typename SketchType, typename... RestSketchTypes>
void print_frequency_comparison_impl(uint64_t item, const std::map<uint64_t, uint64_t> &true_freqs, const std::vector<std::string> &sketch_names, size_t idx,
                                     const SketchType &sketch, const RestSketchTypes &...rest) {
    // Print estimate for current sketch and current item
    double est_freq = sketch.estimate(item);
    std::cout << " | " << std::fixed << std::setprecision(0) << std::setw(10) << est_freq;

    // Recurse for remaining sketches
    print_frequency_comparison_impl(item, true_freqs, sketch_names, idx + 1, rest...);
}

template <typename... SketchTypes>
void print_frequency_comparison(const std::string &title, const std::vector<uint64_t> &items, const std::map<uint64_t, uint64_t> &true_freqs,
                                const std::vector<std::string> &sketch_names, const SketchTypes &...sketches) {
    std::cout << "\n--- " << title << " ---\n\n";

    // Print header
    std::cout << "+------+--------------";
    for (size_t i = 0; i < sketch_names.size(); ++i) { std::cout << "+------------"; }
    std::cout << "+" << std::endl;

    std::cout << "| Rank | True Freq    ";
    for (const auto &name : sketch_names) { std::cout << "| " << std::left << std::setw(10) << name << " "; }
    std::cout << "|" << std::endl;

    std::cout << "+------+--------------";
    for (size_t i = 0; i < sketch_names.size(); ++i) { std::cout << "+------------"; }
    std::cout << "+" << std::endl;

    // Print data rows
    for (size_t i = 0; i < items.size(); ++i) {
        uint64_t item = items[i];
        auto it = true_freqs.find(item);
        uint64_t true_freq = (it != true_freqs.end()) ? it->second : 0;

        std::cout << "| " << std::right << std::setw(4) << (i + 1) << " | " << std::setw(12) << true_freq;
        print_frequency_comparison_impl(item, true_freqs, sketch_names, 0, sketches...);
        std::cout << " |" << std::endl;
    }

    std::cout << "+------+--------------";
    for (size_t i = 0; i < sketch_names.size(); ++i) { std::cout << "+------------"; }
    std::cout << "+" << std::endl;
}
