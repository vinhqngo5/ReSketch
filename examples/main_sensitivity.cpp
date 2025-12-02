#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

// JSON Library
#include "json/json.hpp"

// Utils
#include "utils/ConfigParser.hpp"

// Sketch Headers
#include "frequency_summary/count_min_sketch.hpp"
#include "frequency_summary/resketchv2.hpp"

// Config Headers
#include "frequency_summary/frequency_summary_config.hpp"

// Common utilities
#include "common.hpp"

using namespace std;
using json = nlohmann::json;

// Sensitivity Experiment Config
struct SensitivityConfig {
    uint32_t memory_budget_kb = 32;
    uint32_t repetitions = 5;
    string dataset_type = "zipf";   // "zipf" or "caida"
    string caida_path = "data/CAIDA/only_ip";
    uint64_t total_items = 10000000;
    uint64_t stream_size = 10000000;
    uint64_t stream_diversity = 10000;
    double zipf_param = 1.1;
    string output_file = "output/sensitivity_results.json";

    // k values to test for ReSketch
    vector<uint32_t> k_values = {8, 10, 30, 50, 100};
    // depth values to test for ReSketch
    vector<uint32_t> depth_values = {1, 2, 3, 4, 5, 6, 7, 8};

    static void add_params_to_config_parser(SensitivityConfig &config, ConfigParser &parser) {
        parser.AddParameter(new UnsignedInt32Parameter("app.memory_budget_kb", "32", &config.memory_budget_kb, false, "Memory budget in KB"));
        parser.AddParameter(new UnsignedInt32Parameter("app.repetitions", "5", &config.repetitions, false, "Number of experiment repetitions"));
        parser.AddParameter(new StringParameter("app.dataset_type", "zipf", &config.dataset_type, false, "Dataset type: zipf or caida"));
        parser.AddParameter(new StringParameter("app.caida_path", "data/CAIDA/only_ip", &config.caida_path, false, "Path to CAIDA data file"));
        parser.AddParameter(new UnsignedInt64Parameter("app.total_items", "10000000", &config.total_items, false, "Total items to process"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", "10000000", &config.stream_size, false, "Dataset size for zipf generation"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", "1000000", &config.stream_diversity, false, "Unique items in stream (zipf)"));
        parser.AddParameter(new FloatParameter("app.zipf", "1.1", reinterpret_cast<float *>(&config.zipf_param), false, "Zipfian param 'a'"));
        parser.AddParameter(new StringParameter("app.output_file", "output/sensitivity_results.json", &config.output_file, false, "Output JSON file path"));
    }

    friend std::ostream &operator<<(std::ostream &os, const SensitivityConfig &config) {
        os << "\n=== Sensitivity Experiment Configuration ===\n";
        os << "Memory Budget: " << config.memory_budget_kb << " KB\n";
        os << "Repetitions: " << config.repetitions << "\n";
        os << "Dataset: " << config.dataset_type << "\n";
        if (config.dataset_type == "caida") { os << "CAIDA Path: " << config.caida_path << "\n"; }
        os << "Total Items: " << config.total_items << "\n";
        os << "Dataset Size: " << config.stream_size << "\n";
        if (config.dataset_type == "zipf") {
            os << "Stream Diversity: " << config.stream_diversity << "\n";
            os << "Zipf Parameter: " << config.zipf_param << "\n";
        }
        os << "ReSketch K values: ";
        for (uint32_t i = 0; i < config.k_values.size(); ++i) {
            os << config.k_values[i];
            if (i < config.k_values.size() - 1) os << ", ";
        }
        os << "\n";
        os << "ReSketch Depth values: ";
        for (uint32_t i = 0; i < config.depth_values.size(); ++i) {
            os << config.depth_values[i];
            if (i < config.depth_values.size() - 1) os << ", ";
        }
        os << "\n";
        os << "Output File: " << config.output_file << "\n";
        return os;
    }
};

// Result for a single configuration
struct SensitivityResult {
    string algorithm;
    uint32_t k_value;   // Only relevant for ReSketch
    uint32_t width;
    uint32_t depth;
    uint64_t memory_bytes;
    double throughput_mops;
    double query_throughput_mops;
    double are;
    double aae;
    double are_within_var;
    double aae_within_var;
};

void export_to_json(const string &filename, const SensitivityConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config,
                    const map<string, vector<vector<SensitivityResult>>> &all_results) {
    create_directory(filename);

    // Build JSON object
    json j;

    // Metadata section
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    gmtime_r(&now_time_t, &tm_now);
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%SZ");

    j["metadata"] = {{"experiment_type", "sensitivity"}, {"timestamp", timestamp.str()}};

    // Config section
    j["config"]["experiment"] = {
        {"memory_budget_kb", config.memory_budget_kb}, {"repetitions", config.repetitions},           {"dataset_type", config.dataset_type}, {"total_items", config.total_items},
        {"stream_size", config.stream_size},           {"stream_diversity", config.stream_diversity}, {"zipf_param", config.zipf_param}};

    j["config"]["base_sketch_config"]["countmin"] = {{"depth", cm_config.depth}};
    j["config"]["base_sketch_config"]["resketch"] = {{"depth", rs_config.depth}};

    j["config"]["sensitivity_params"] = {{"k_values", config.k_values}, {"depth_values", config.depth_values}};

    // Results section
    json results_json;
    for (const auto &[config_name, repetitions] : all_results) {
        json config_reps = json::array();

        for (uint32_t rep = 0; rep < repetitions.size(); ++rep) {
            json rep_json;
            rep_json["repetition_id"] = rep;

            json results_array = json::array();
            for (const auto &result : repetitions[rep]) {
                json result_json = {{"algorithm", result.algorithm},
                                    {"k_value", result.k_value},
                                    {"width", result.width},
                                    {"depth", result.depth},
                                    {"memory_bytes", result.memory_bytes},
                                    {"throughput_mops", result.throughput_mops},
                                    {"query_throughput_mops", result.query_throughput_mops},
                                    {"are", result.are},
                                    {"aae", result.aae},
                                    {"are_within_var", result.are_within_var},
                                    {"aae_within_var", result.aae_within_var}};
                results_array.push_back(result_json);
            }

            rep_json["results"] = results_array;
            config_reps.push_back(rep_json);
        }

        results_json[config_name] = config_reps;
    }

    j["results"] = results_json;

    // Write to file
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: Cannot open output file: " << filename << endl;
        return;
    }

    out << j.dump(2);
    out.close();

    cout << "\nResults exported to: " << filename << endl;
}

void run_sensitivity_experiment(const SensitivityConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config) {
    cout << config << endl;
    cout << cm_config << endl;
    cout << rs_config << endl;

    // Results storage: config_name -> repetitions -> results
    map<string, vector<vector<SensitivityResult>>> all_results;

    // Configuration names
    all_results["CountMin"] = vector<vector<SensitivityResult>>(config.repetitions);
    for (auto depth : config.depth_values) {
        for (auto k : config.k_values) {
            string config_name = "ReSketch_d" + to_string(depth) + "_k" + to_string(k);
            all_results[config_name] = vector<vector<SensitivityResult>>(config.repetitions);
        }
    }

    uint64_t memory_budget_bytes = (uint64_t) config.memory_budget_kb * 1024;

    for (uint32_t rep = 0; rep < config.repetitions; ++rep) {
        cout << "\n=== Repetition " << (rep + 1) << "/" << config.repetitions << " ===" << endl;

        // Generate or load dataset
        vector<uint64_t> data;
        if (config.dataset_type == "zipf") {
            cout << "Generating Zipf data..." << endl;
            data = generate_zipf_data(config.stream_size, config.stream_diversity, config.zipf_param);
        } else if (config.dataset_type == "caida") {
            cout << "Reading CAIDA data..." << endl;
            data = read_caida_data(config.caida_path, config.stream_size);
            if (data.empty()) {
                cerr << "Error: Failed to read CAIDA data. Skipping repetition." << endl;
                continue;
            }
        } else {
            cerr << "Error: Unknown dataset type: " << config.dataset_type << ". Skipping repetition." << endl;
            continue;
        }

        // Limit to total_items
        if (data.size() > config.total_items) {
            data.resize(config.total_items);
        } else if (data.size() < config.total_items) {
            // Repeat data to reach total_items
            vector<uint64_t> extended_data;
            extended_data.reserve(config.total_items);
            for (uint64_t i = 0; i < config.total_items; ++i) { extended_data.push_back(data[i % data.size()]); }
            data = std::move(extended_data);
        }

        cout << "Processing " << data.size() << " items" << endl;

        // Calculate true frequencies
        map<uint64_t, uint64_t> true_freqs;
        for (const auto &item : data) { true_freqs[item]++; }

        // Prepare query set (all unique items)
        vector<uint64_t> query_items;
        query_items.reserve(true_freqs.size());
        for (const auto &[item, freq] : true_freqs) { query_items.push_back(item); }

        // Test Count-Min Sketch
        {
            uint32_t cm_width = calculate_width_from_memory_cm(memory_budget_bytes, cm_config.depth);
            cout << "\nCount-Min: depth=" << cm_config.depth << ", width=" << cm_width << endl;

            CountMinConfig cm_conf = cm_config;
            cm_conf.width = cm_width;
            cm_conf.calculate_from = "WIDTH_DEPTH";
            CountMinSketch cm_sketch(cm_conf);

            // Measure update throughput
            Timer timer;
            timer.start();
            for (const auto &item : data) { cm_sketch.update(item); }
            double update_duration = timer.stop_s();
            double throughput = (update_duration > 0) ? (data.size() / update_duration / 1e6) : 0;

            // Measure query throughput
            volatile double cm_sum = 0.0;
            timer.start();
            for (const auto &item : query_items) { cm_sum += cm_sketch.estimate(item); }
            double query_duration = timer.stop_s();
            double query_throughput = (query_duration > 0) ? (query_items.size() / query_duration / 1e6) : 0;

            // Calculate accuracy
            double are = calculate_are_all_items(cm_sketch, true_freqs);
            double aae = calculate_aae_all_items(cm_sketch, true_freqs);

            // Compute within-run variance across items for relative and absolute errors
            vector<double> are_errors;
            vector<double> aae_errors;
            are_errors.reserve(true_freqs.size());
            aae_errors.reserve(true_freqs.size());
            for (const auto &p : true_freqs) {
                uint64_t item = p.first;
                uint64_t true_freq = p.second;
                double est_freq = cm_sketch.estimate(item);
                double rel_error = (true_freq > 0) ? (std::abs(est_freq - (double) true_freq) / (double) true_freq) : 0.0;
                are_errors.push_back(rel_error);
                aae_errors.push_back(std::abs(est_freq - (double) true_freq));
            }

            double are_var_within = 0.0;
            double aae_var_within = 0.0;
            if (!are_errors.empty()) {
                double mean_are = are;
                double sum_sq = 0.0;
                for (double v : are_errors) sum_sq += (v - mean_are) * (v - mean_are);
                are_var_within = sum_sq / are_errors.size();
            }
            if (!aae_errors.empty()) {
                double mean_aae = aae;
                double sum_sq = 0.0;
                for (double v : aae_errors) sum_sq += (v - mean_aae) * (v - mean_aae);
                aae_var_within = sum_sq / aae_errors.size();
            }

            SensitivityResult result;
            result.algorithm = "CountMin";
            result.k_value = 0;   // Not applicable
            result.width = cm_width;
            result.depth = cm_config.depth;
            result.memory_bytes = cm_sketch.get_max_memory_usage();
            result.throughput_mops = throughput;
            result.query_throughput_mops = query_throughput;
            result.are = are;
            result.aae = aae;
            result.are_within_var = are_var_within;
            result.aae_within_var = aae_var_within;

            all_results["CountMin"][rep].push_back(result);

            cout << "  Throughput: " << throughput << " Mops/s" << endl;
            cout << "  Query Throughput: " << query_throughput << " Mops/s" << endl;
            cout << "  Memory: " << result.memory_bytes / 1024 << " KB" << endl;
            cout << "  ARE: " << are << ", AAE: " << aae << endl;
        }

        // Test ReSketch with different depth and k values
        for (auto depth : config.depth_values) {
            for (auto k : config.k_values) {
                uint32_t rs_width = calculate_width_from_memory_resketch(memory_budget_bytes, depth, k);
                cout << "\nReSketch: depth=" << depth << ", k=" << k << ", width=" << rs_width << endl;

                ReSketchConfig rs_conf = rs_config;
                rs_conf.depth = depth;
                rs_conf.width = rs_width;
                rs_conf.kll_k = k;
                ReSketchV2 rs_sketch(rs_conf);

                // Measure update throughput
                Timer timer;
                timer.start();
                for (const auto &item : data) { rs_sketch.update(item); }
                double update_duration = timer.stop_s();
                double throughput = (update_duration > 0) ? (data.size() / update_duration / 1e6) : 0;

                // Measure query throughput
                volatile double rs_sum = 0.0;
                timer.start();
                for (const auto &item : query_items) { rs_sum += rs_sketch.estimate(item); }
                double query_duration = timer.stop_s();
                double query_throughput = (query_duration > 0) ? (query_items.size() / query_duration / 1e6) : 0;

                // Calculate accuracy
                double are = calculate_are_all_items(rs_sketch, true_freqs);
                double aae = calculate_aae_all_items(rs_sketch, true_freqs);

                // Compute within-run variance across items for relative and absolute errors
                vector<double> are_errors;
                vector<double> aae_errors;
                are_errors.reserve(true_freqs.size());
                aae_errors.reserve(true_freqs.size());
                for (const auto &p : true_freqs) {
                    uint64_t item = p.first;
                    uint64_t true_freq = p.second;
                    double est_freq = rs_sketch.estimate(item);
                    double rel_error = (true_freq > 0) ? (std::abs(est_freq - (double) true_freq) / (double) true_freq) : 0.0;
                    are_errors.push_back(rel_error);
                    aae_errors.push_back(std::abs(est_freq - (double) true_freq));
                }

                double are_var_within_rs = 0.0;
                double aae_var_within_rs = 0.0;
                if (!are_errors.empty()) {
                    double mean_are = are;
                    double sum_sq = 0.0;
                    for (double v : are_errors) sum_sq += (v - mean_are) * (v - mean_are);
                    are_var_within_rs = sum_sq / are_errors.size();
                }
                if (!aae_errors.empty()) {
                    double mean_aae = aae;
                    double sum_sq = 0.0;
                    for (double v : aae_errors) sum_sq += (v - mean_aae) * (v - mean_aae);
                    aae_var_within_rs = sum_sq / aae_errors.size();
                }

                SensitivityResult result;
                result.algorithm = "ReSketch";
                result.k_value = k;
                result.width = rs_width;
                result.depth = depth;
                result.memory_bytes = rs_sketch.get_max_memory_usage();
                result.throughput_mops = throughput;
                result.query_throughput_mops = query_throughput;
                result.are = are;
                result.aae = aae;
                result.are_within_var = are_var_within_rs;
                result.aae_within_var = aae_var_within_rs;

                string config_name = "ReSketch_d" + to_string(depth) + "_k" + to_string(k);
                all_results[config_name][rep].push_back(result);

                cout << "  Throughput: " << throughput << " Mops/s" << endl;
                cout << "  Query Throughput: " << query_throughput << " Mops/s" << endl;
                cout << "  Memory: " << result.memory_bytes / 1024 << " KB" << endl;
                cout << "  ARE: " << are << ", AAE: " << aae << endl;
            }
        }
    }

    // Add timestamp to output filename
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    string timestamp = timestamp_stream.str();

    // Insert timestamp before file extension
    string output_file = config.output_file;
    uint32_t ext_pos = output_file.find_last_of('.');
    if (ext_pos != string::npos) {
        output_file = output_file.substr(0, ext_pos) + "_" + timestamp + output_file.substr(ext_pos);
    } else {
        output_file += "_" + timestamp;
    }

    export_to_json(output_file, config, cm_config, rs_config, all_results);
}

int main(int argc, char **argv) {
    ConfigParser parser;
    SensitivityConfig sens_config;
    CountMinConfig cm_config;
    ReSketchConfig rs_config;

    SensitivityConfig::add_params_to_config_parser(sens_config, parser);
    CountMinConfig::add_params_to_config_parser(cm_config, parser);
    ReSketchConfig::add_params_to_config_parser(rs_config, parser);

    if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h")) {
        parser.PrintUsage();
        return 0;
    }

    Status s = parser.ParseCommandLine(argc, argv);
    if (!s.IsOK()) {
        fprintf(stderr, "%s\n", s.ToString().c_str());
        return -1;
    }

    run_sensitivity_experiment(sens_config, cm_config, rs_config);

    return 0;
}
