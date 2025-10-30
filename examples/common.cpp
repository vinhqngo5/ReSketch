#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/stat.h>

#include "frequency_summary/dynamic_sketch_wrapper.hpp"
#include "frequency_summary/geometric_sketch_wrapper.hpp"

std::vector<uint64_t> generate_zipf_data(uint64_t size, uint64_t diversity, double a) {
    std::vector<double> pdf(diversity);
    double sum = 0.0;
    for (uint64_t i = 1; i <= diversity; ++i) {
        pdf[i - 1] = 1.0 / std::pow(static_cast<double>(i), a);
        sum += pdf[i - 1];
    }
    for (uint64_t i = 0; i < diversity; ++i) { pdf[i] /= sum; }
    std::discrete_distribution<uint64_t> dist(pdf.begin(), pdf.end());
    std::mt19937_64 rng(std::random_device{}());
    std::vector<uint64_t> data;
    data.reserve(size);
    for (uint64_t i = 0; i < size; ++i) { data.push_back(dist(rng)); }
    return data;
}

std::vector<uint64_t> read_caida_data(const std::string &path, uint64_t max_items) {
    std::vector<uint64_t> data;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open CAIDA file: " << path << std::endl;
        return data;
    }

    std::string line;
    uint64_t count = 0;
    while (std::getline(file, line) && count < max_items) {
        uint64_t ip_as_int = 0;

        unsigned int a, b, c, d;
        if (sscanf(line.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            ip_as_int = ((uint64_t) a << 24) | ((uint64_t) b << 16) | ((uint64_t) c << 8) | (uint64_t) d;
            data.push_back(ip_as_int);
            count++;
        } else {
            std::stringstream ss(line);
            if (ss >> ip_as_int) {
                data.push_back(ip_as_int);
                count++;
            }
        }
    }
    file.close();
    std::cout << "Read " << data.size() << " items from CAIDA file." << std::endl;
    return data;
}

uint32_t calculate_width_from_memory_cm(uint64_t memory_bytes, uint32_t depth) {
    if (depth == 0) return 0;
    return CountMinSketch::calculate_max_width(memory_bytes, depth);
}

uint32_t calculate_width_from_memory_resketch(uint64_t memory_bytes, uint32_t depth, uint32_t kll_k) {
    if (depth == 0) return 0;
    return ReSketchV2::calculate_max_width(memory_bytes, depth, kll_k);
}

uint32_t calculate_width_from_memory_geometric(uint64_t memory_bytes, uint32_t depth) {
    if (depth == 0) return 0;
    return GeometricSketchWrapper::calculate_max_width(memory_bytes, depth);
}

uint32_t calculate_width_from_memory_dynamic(uint64_t memory_bytes, uint32_t depth) {
    if (depth == 0) return 0;
    return DynamicSketchWrapper::calculate_max_width(memory_bytes, depth);
}

void create_directory(const std::string &path) {
    uint32_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path.substr(0, pos);
        mkdir(dir.c_str(), 0755);
    }
}
