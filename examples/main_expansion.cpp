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
#include "frequency_summary/dynamic_sketch_wrapper.hpp"
#include "frequency_summary/geometric_sketch_wrapper.hpp"
#include "frequency_summary/resketchv2.hpp"

// Config Headers
#include "frequency_summary/frequency_summary_config.hpp"

// Common utilities
#include "common.hpp"

using namespace std;
using json = nlohmann::json;

// Expansion Experiment Config
struct ExpansionConfig {
    uint32_t initial_memory_kb = 32;
    uint32_t expansion_interval = 100000;
    uint32_t memory_increment_kb = 32;
    uint32_t repetitions = 10;
    string dataset_type = "zipf";   // "zipf" or "caida"
    string caida_path = "data/CAIDA/only_ip";
    uint64_t total_items = 10000000;   // Total items to process (may repeat dataset) -> time repeat = total_items / stream_size
    uint64_t stream_size = 10000000;
    uint64_t stream_diversity = 10000;
    float zipf_param = 1.1;
    string output_file = "output/expansion_results.json";

    static void add_params_to_config_parser(ExpansionConfig &config, ConfigParser &parser) {
        parser.AddParameter(new UnsignedInt32Parameter("app.initial_memory_kb", "32", &config.initial_memory_kb, false, "Initial memory budget in KB"));
        parser.AddParameter(new UnsignedInt32Parameter("app.expansion_interval", "100000", &config.expansion_interval, false, "Items between expansions"));
        parser.AddParameter(new UnsignedInt32Parameter("app.memory_increment_kb", "32", &config.memory_increment_kb, false, "Memory increment per expansion in KB"));
        parser.AddParameter(new UnsignedInt32Parameter("app.repetitions", "10", &config.repetitions, false, "Number of experiment repetitions"));
        parser.AddParameter(new StringParameter("app.dataset_type", "zipf", &config.dataset_type, false, "Dataset type: zipf or caida"));
        parser.AddParameter(new StringParameter("app.caida_path", "data/CAIDA/only_ip", &config.caida_path, false, "Path to CAIDA data file"));
        parser.AddParameter(new UnsignedInt64Parameter("app.total_items", "10000000", &config.total_items, false, "Total items to process (will repeat dataset if needed)"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", "10000000", &config.stream_size, false, "Dataset size for zipf generation"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", "1000000", &config.stream_diversity, false, "Unique items in stream (zipf)"));
        parser.AddParameter(new FloatParameter("app.zipf", "1.1", &config.zipf_param, false, "Zipfian param 'a'"));
        parser.AddParameter(new StringParameter("app.output_file", "output/expansion_results.json", &config.output_file, false, "Output JSON file path"));
    }

    friend std::ostream &operator<<(std::ostream &os, const ExpansionConfig &config) {
        os << "\n=== Expansion Experiment Configuration ===\n";
        os << "Initial Memory: " << config.initial_memory_kb << " KB\n";
        os << "Expansion Interval: " << config.expansion_interval << " items\n";
        os << "Memory Increment: " << config.memory_increment_kb << " KB\n";
        os << "Repetitions: " << config.repetitions << "\n";
        os << "Dataset: " << config.dataset_type << "\n";
        if (config.dataset_type == "caida") { os << "CAIDA Path: " << config.caida_path << "\n"; }
        os << "Total Items to Process: " << config.total_items << "\n";
        os << "Dataset Size: " << config.stream_size << "\n";
        if (config.dataset_type == "zipf") {
            os << "Stream Diversity: " << config.stream_diversity << "\n";
            os << "Zipf Parameter: " << config.zipf_param << "\n";
        }
        os << "Output File: " << config.output_file << "\n";
        return os;
    }
};

// Checkpoint data
struct Checkpoint {
    uint64_t items_processed;
    double throughput_mops;
    double query_throughput_mops;
    uint64_t memory_kb;
    double are;
    double aae;
};

void export_to_json(const string &filename, const ExpansionConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config, const GeometricSketchConfig &gs_config,
                    const DynamicSketchConfig &ds_config, const map<string, vector<vector<Checkpoint>>> &all_results) {
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

    j["metadata"] = {{"experiment_type", "expansion"}, {"timestamp", timestamp.str()}};

    // Config section
    j["config"]["experiment"] = {{"initial_memory_kb", config.initial_memory_kb},
                                 {"expansion_interval", config.expansion_interval},
                                 {"memory_increment_kb", config.memory_increment_kb},
                                 {"repetitions", config.repetitions},
                                 {"dataset_type", config.dataset_type},
                                 {"total_items", config.total_items},
                                 {"stream_size", config.stream_size},
                                 {"stream_diversity", config.stream_diversity},
                                 {"zipf_param", config.zipf_param}};

    j["config"]["base_sketch_config"]["countmin"] = {{"depth", cm_config.depth}};
    j["config"]["base_sketch_config"]["resketch"] = {{"depth", rs_config.depth}, {"kll_k", rs_config.kll_k}};
    j["config"]["base_sketch_config"]["geometric"] = {{"depth", gs_config.depth}};
    j["config"]["base_sketch_config"]["dynamic"] = {{"depth", ds_config.depth}};

    // Results section
    json results_json;
    for (const auto &[sketch_name, repetitions] : all_results) {
        json sketch_reps = json::array();

        for (uint32_t rep = 0; rep < repetitions.size(); ++rep) {
            json rep_json;
            rep_json["repetition_id"] = rep;

            json checkpoints_array = json::array();
            for (const auto &cp : repetitions[rep]) {
                json cp_json = {{"items_processed", cp.items_processed},
                                {"memory_bytes", cp.memory_kb * 1024},
                                {"throughput_mops", cp.throughput_mops},
                                {"query_throughput_mops", cp.query_throughput_mops},
                                {"are", cp.are},
                                {"aae", cp.aae}};
                checkpoints_array.push_back(cp_json);
            }

            rep_json["checkpoints"] = checkpoints_array;
            sketch_reps.push_back(rep_json);
        }

        results_json[sketch_name] = sketch_reps;
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

void run_expansion_experiment(const ExpansionConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config, const GeometricSketchConfig &gs_config,
                              const DynamicSketchConfig &ds_config) {
    cout << config << endl;
    cout << cm_config << endl;
    cout << rs_config << endl;
    cout << gs_config << endl;
    cout << ds_config << endl;

    map<string, vector<vector<Checkpoint>>> all_results;
    all_results["CountMin"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["ReSketch"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["DynamicSketch"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["GeometricSketch"] = vector<vector<Checkpoint>>(config.repetitions);

    for (uint32_t rep = 0; rep < config.repetitions; ++rep) {
        cout << "\n=== Repetition " << (rep + 1) << "/" << config.repetitions << " ===" << endl;

        // Generate or load base dataset
        vector<uint64_t> base_data;
        if (config.dataset_type == "zipf") {
            cout << "Generating Zipf data..." << endl;
            base_data = generate_zipf_data(config.stream_size, config.stream_diversity, config.zipf_param);
        } else if (config.dataset_type == "caida") {
            cout << "Reading CAIDA data..." << endl;
            base_data = read_caida_data(config.caida_path, config.stream_size);
            if (base_data.empty()) {
                cerr << "Error: Failed to read CAIDA data. Skipping repetition." << endl;
                continue;
            }
        } else {
            cerr << "Error: Unknown dataset type: " << config.dataset_type << ". Skipping repetition." << endl;
            continue;
        }

        uint64_t num_repeats = (config.total_items + base_data.size() - 1) / base_data.size();

        cout << "Base dataset size: " << base_data.size() << endl;
        cout << "Will process " << config.total_items << " items total (repeating dataset " << num_repeats << " times)" << endl;

        uint64_t initial_memory_bytes = (uint64_t) config.initial_memory_kb * 1024;
        uint64_t memory_increment_bytes = (uint64_t) config.memory_increment_kb * 1024;

        // Initialize sketches with initial memory budget
        uint32_t cm_width = calculate_width_from_memory_cm(initial_memory_bytes, cm_config.depth);
        uint32_t rs_width = calculate_width_from_memory_resketch(initial_memory_bytes, rs_config.depth, rs_config.kll_k);
        uint32_t gs_width = calculate_width_from_memory_geometric(initial_memory_bytes, gs_config.depth);
        uint32_t ds_width = calculate_width_from_memory_dynamic(initial_memory_bytes, ds_config.depth);

        cout << "Initial widths: CM=" << cm_width << ", RS=" << rs_width << ", GS=" << gs_width << ", DS=" << ds_width << endl;

        // Prepare local configs with computed widths and construct sketches using their config constructors
        CountMinConfig cm_conf = cm_config;
        cm_conf.width = cm_width;
        CountMinSketch cm_sketch(cm_conf);

        ReSketchConfig rs_conf = rs_config;
        rs_conf.width = rs_width;
        ReSketchV2 rs_sketch(rs_conf);

        GeometricSketchConfig gs_conf = gs_config;
        gs_conf.width = gs_width;
        GeometricSketchWrapper gs_sketch(gs_conf);

        DynamicSketchConfig ds_conf = ds_config;
        ds_conf.width = ds_width;
        DynamicSketchWrapper ds_sketch(ds_conf);

        // Track DynamicSketch expansion: accumulate budget until we can double
        uint64_t ds_accumulated_budget = 0;
        uint64_t ds_last_expansion_size = initial_memory_bytes;

        Timer timer;
        uint64_t items_processed = 0;
        uint32_t checkpoint_idx = 0;

        while (items_processed < config.total_items) {
            uint64_t chunk_size = min((uint64_t) config.expansion_interval, config.total_items - items_processed);
            uint64_t chunk_start = items_processed;
            uint64_t chunk_end = chunk_start + chunk_size;

            // Process chunk for CountMin
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { cm_sketch.update(base_data[i % base_data.size()]); }
            double cm_duration = timer.stop_s();

            // Process chunk for ReSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { rs_sketch.update(base_data[i % base_data.size()]); }
            double rs_duration = timer.stop_s();

            // Process chunk for GeometricSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { gs_sketch.update(base_data[i % base_data.size()]); }
            double gs_duration = timer.stop_s();

            // Process chunk for DynamicSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { ds_sketch.update(base_data[i % base_data.size()]); }
            double ds_duration = timer.stop_s();

            items_processed += chunk_size;

            // Calculate true frequencies for items processed up to this checkpoint
            map<uint64_t, uint64_t> true_freqs_at_checkpoint;
            for (uint64_t i = 0; i < items_processed; ++i) { true_freqs_at_checkpoint[base_data[i % base_data.size()]]++; }

            // Record checkpoint
            Checkpoint cm_cp, rs_cp, gs_cp, ds_cp;
            cm_cp.items_processed = rs_cp.items_processed = gs_cp.items_processed = ds_cp.items_processed = items_processed;

            cm_cp.throughput_mops = (cm_duration > 0) ? (chunk_size / cm_duration / 1e6) : 0;
            rs_cp.throughput_mops = (rs_duration > 0) ? (chunk_size / rs_duration / 1e6) : 0;
            gs_cp.throughput_mops = (gs_duration > 0) ? (chunk_size / gs_duration / 1e6) : 0;
            ds_cp.throughput_mops = (ds_duration > 0) ? (chunk_size / ds_duration / 1e6) : 0;

            cm_cp.memory_kb = cm_sketch.get_max_memory_usage() / 1024;
            rs_cp.memory_kb = rs_sketch.get_max_memory_usage() / 1024;
            gs_cp.memory_kb = gs_sketch.get_max_memory_usage() / 1024;
            ds_cp.memory_kb = ds_sketch.get_max_memory_usage() / 1024;

            cm_cp.are = calculate_are_all_items(cm_sketch, true_freqs_at_checkpoint);
            rs_cp.are = calculate_are_all_items(rs_sketch, true_freqs_at_checkpoint);
            gs_cp.are = calculate_are_all_items(gs_sketch, true_freqs_at_checkpoint);
            ds_cp.are = calculate_are_all_items(ds_sketch, true_freqs_at_checkpoint);

            cm_cp.aae = calculate_aae_all_items(cm_sketch, true_freqs_at_checkpoint);
            rs_cp.aae = calculate_aae_all_items(rs_sketch, true_freqs_at_checkpoint);
            gs_cp.aae = calculate_aae_all_items(gs_sketch, true_freqs_at_checkpoint);
            ds_cp.aae = calculate_aae_all_items(ds_sketch, true_freqs_at_checkpoint);

            // Measure query throughput on all unique items
            vector<uint64_t> unique_items;
            unique_items.reserve(true_freqs_at_checkpoint.size());
            for (const auto &[item, freq] : true_freqs_at_checkpoint) { unique_items.push_back(item); }
            uint64_t num_queries = unique_items.size();

            // CountMin query throughput
            volatile double cm_sum = 0.0;
            timer.start();
            for (const auto &item : unique_items) { cm_sum += cm_sketch.estimate(item); }
            double cm_query_duration = timer.stop_s();
            cm_cp.query_throughput_mops = (cm_query_duration > 0) ? (num_queries / cm_query_duration / 1e6) : 0;

            // ReSketch query throughput
            volatile double rs_sum = 0.0;
            timer.start();
            for (const auto &item : unique_items) { rs_sum += rs_sketch.estimate(item); }
            double rs_query_duration = timer.stop_s();
            rs_cp.query_throughput_mops = (rs_query_duration > 0) ? (num_queries / rs_query_duration / 1e6) : 0;

            // GeometricSketch query throughput
            volatile double gs_sum = 0.0;
            timer.start();
            for (const auto &item : unique_items) { gs_sum += gs_sketch.estimate(item); }
            double gs_query_duration = timer.stop_s();
            gs_cp.query_throughput_mops = (gs_query_duration > 0) ? (num_queries / gs_query_duration / 1e6) : 0;

            // DynamicSketch query throughput
            volatile double ds_sum = 0.0;
            timer.start();
            for (const auto &item : unique_items) { ds_sum += ds_sketch.estimate(item); }
            double ds_query_duration = timer.stop_s();
            ds_cp.query_throughput_mops = (ds_query_duration > 0) ? (num_queries / ds_query_duration / 1e6) : 0;

            all_results["CountMin"][rep].push_back(cm_cp);
            all_results["ReSketch"][rep].push_back(rs_cp);
            all_results["GeometricSketch"][rep].push_back(gs_cp);
            all_results["DynamicSketch"][rep].push_back(ds_cp);

            cout << "Checkpoint " << (checkpoint_idx + 1) << " at " << items_processed << " items:" << endl;
            cout << "  CM: " << cm_cp.throughput_mops << " Mops, Query: " << cm_cp.query_throughput_mops << " Mops, " << cm_cp.memory_kb << " KB, ARE=" << cm_cp.are
                 << ", AAE=" << cm_cp.aae << endl;
            cout << "  RS: " << rs_cp.throughput_mops << " Mops, Query: " << rs_cp.query_throughput_mops << " Mops, " << rs_cp.memory_kb << " KB, ARE=" << rs_cp.are
                 << ", AAE=" << rs_cp.aae << endl;
            cout << "  GS: " << gs_cp.throughput_mops << " Mops, Query: " << gs_cp.query_throughput_mops << " Mops, " << gs_cp.memory_kb << " KB, ARE=" << gs_cp.are
                 << ", AAE=" << gs_cp.aae << endl;
            cout << "  DS: " << ds_cp.throughput_mops << " Mops, Query: " << ds_cp.query_throughput_mops << " Mops, " << ds_cp.memory_kb << " KB, ARE=" << ds_cp.are
                 << ", AAE=" << ds_cp.aae << endl;

            // Expand sketches if not at end
            if (items_processed < config.total_items) {
                // CountMin: cannot expand (do nothing)

                // ReSketch: expand by memory_increment_kb
                uint32_t rs_new_width = calculate_width_from_memory_resketch(rs_sketch.get_max_memory_usage() + memory_increment_bytes, rs_config.depth, rs_config.kll_k);
                if (rs_new_width > rs_width) {
                    rs_sketch.expand(rs_new_width);
                    rs_width = rs_new_width;
                    cout << "  -> ReSketch expanded to width " << rs_width << endl;
                }

                // GeometricSketch: expand by memory_increment_kb
                uint32_t gs_new_width = calculate_width_from_memory_geometric(gs_sketch.get_max_memory_usage() + memory_increment_bytes, gs_config.depth);
                if (gs_new_width > gs_width) {
                    gs_sketch.expand(gs_new_width);
                    gs_width = gs_new_width;
                    cout << "  -> GeometricSketch expanded to width " << gs_width << endl;
                }

                // DynamicSketch: accumulate budget until we can double
                ds_accumulated_budget += memory_increment_bytes;
                // Check if accumulated budget is enough to double the last expansion size
                if (ds_accumulated_budget >= ds_last_expansion_size) {
                    uint32_t ds_new_width = calculate_width_from_memory_dynamic(ds_sketch.get_max_memory_usage() + ds_last_expansion_size, ds_config.depth);
                    if (ds_new_width > ds_width) {
                        ds_sketch.expand(ds_new_width);
                        cout << "  -> DynamicSketch expanded to width " << ds_new_width << " (added " << (ds_last_expansion_size / 1024)
                             << " KB, accumulated budget: " << (ds_accumulated_budget / 1024) << " KB)" << endl;
                        ds_accumulated_budget -= ds_last_expansion_size;
                        ds_last_expansion_size *= 2;
                        ds_width = ds_new_width;
                    }
                }
            }

            checkpoint_idx++;
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

    export_to_json(output_file, config, cm_config, rs_config, gs_config, ds_config, all_results);
}

int main(int argc, char **argv) {
    ConfigParser parser;
    ExpansionConfig exp_config;
    CountMinConfig cm_config;
    ReSketchConfig rs_config;
    GeometricSketchConfig gs_config;
    DynamicSketchConfig ds_config;

    ExpansionConfig::add_params_to_config_parser(exp_config, parser);
    CountMinConfig::add_params_to_config_parser(cm_config, parser);
    ReSketchConfig::add_params_to_config_parser(rs_config, parser);
    GeometricSketchConfig::add_params_to_config_parser(gs_config, parser);
    DynamicSketchConfig::add_params_to_config_parser(ds_config, parser);

    if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h")) {
        parser.PrintUsage();
        return 0;
    }

    Status s = parser.ParseCommandLine(argc, argv);
    if (!s.IsOK()) {
        fprintf(stderr, "%s\n", s.ToString().c_str());
        return -1;
    }

    run_expansion_experiment(exp_config, cm_config, rs_config, gs_config, ds_config);

    return 0;
}
