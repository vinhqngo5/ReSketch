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
#include "frequency_summary/geometric_sketch_wrapper.hpp"
#include "frequency_summary/resketchv2.hpp"

// Config Headers
#include "frequency_summary/frequency_summary_config.hpp"

// Common utilities
#include "common.hpp"

using namespace std;
using json = nlohmann::json;

// Shrinking Experiment Config
struct ShrinkingConfig {
    uint32_t initial_memory_kb = 160;   // Starting memory (e.g., 1.6MB)
    uint32_t max_memory_kb = 640;       // Maximum memory during warmup (e.g., 6.4MB)
    uint32_t final_memory_kb = 32;
    uint32_t shrinking_interval = 10000 / 8;
    uint32_t memory_decrement_kb = 32 / 8;
    uint32_t repetitions = 10;
    string dataset_type = "zipf";   // "zipf" or "caida"
    string caida_path = "data/CAIDA/only_ip";
    uint64_t total_items = 10000000;   // Total items to process during shrinking phase
    uint64_t stream_size = 10000000;
    uint64_t stream_diversity = 10000000;
    double zipf_param = 1.1;
    string output_file = "output/shrinking_results.json";

    static void add_params_to_config_parser(ShrinkingConfig &config, ConfigParser &parser) {
        parser.AddParameter(new UnsignedInt32Parameter("app.initial_memory_kb", "160", &config.initial_memory_kb, false, "Initial memory budget in KB"));
        parser.AddParameter(new UnsignedInt32Parameter("app.max_memory_kb", "640", &config.max_memory_kb, false, "Maximum memory during warmup in KB"));
        parser.AddParameter(new UnsignedInt32Parameter("app.final_memory_kb", "32", &config.final_memory_kb, false, "Final minimum memory in KB"));
        parser.AddParameter(new UnsignedInt32Parameter("app.shrinking_interval", "1250", &config.shrinking_interval, false, "Items between shrinking operations"));
        parser.AddParameter(new UnsignedInt32Parameter("app.memory_decrement_kb", "4", &config.memory_decrement_kb, false, "Memory decrement per shrinking step in KB"));
        parser.AddParameter(new UnsignedInt32Parameter("app.repetitions", "10", &config.repetitions, false, "Number of experiment repetitions"));
        parser.AddParameter(new StringParameter("app.dataset_type", "zipf", &config.dataset_type, false, "Dataset type: zipf or caida"));
        parser.AddParameter(new StringParameter("app.caida_path", "data/CAIDA/only_ip", &config.caida_path, false, "Path to CAIDA data file"));
        parser.AddParameter(new UnsignedInt64Parameter("app.total_items", "10000000", &config.total_items, false, "Total items to process during shrinking phase"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", "10000000", &config.stream_size, false, "Dataset size for zipf generation"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", "1000000", &config.stream_diversity, false, "Unique items in stream (zipf)"));
        parser.AddParameter(new FloatParameter("app.zipf", "1.1", reinterpret_cast<float *>(&config.zipf_param), false, "Zipfian param 'a'"));
        parser.AddParameter(new StringParameter("app.output_file", "output/shrinking_results.json", &config.output_file, false, "Output JSON file path"));
    }

    friend std::ostream &operator<<(std::ostream &os, const ShrinkingConfig &config) {
        os << "\n=== Shrinking Experiment Configuration ===\n";
        os << "Initial Memory: " << config.initial_memory_kb << " KB\n";
        os << "Max Memory (Warmup): " << config.max_memory_kb << " KB\n";
        os << "Final Memory: " << config.final_memory_kb << " KB\n";
        os << "Shrinking Interval: " << config.shrinking_interval << " items\n";
        os << "Memory Decrement: " << config.memory_decrement_kb << " KB\n";
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
    // As of now, is_warmup is not really used but reserved for future use if ingesting while warming up is needed
    bool is_warmup;                 // True during warmup expansion phase
    bool geometric_cannot_shrink;   // True when GeometricSketch can't shrink anymore
};

void export_to_json(const string &filename, const ShrinkingConfig &config, const ReSketchConfig &rs_config, const GeometricSketchConfig &gs_config,
                    const map<string, vector<vector<Checkpoint>>> &all_results) {
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

    j["metadata"] = {{"experiment_type", "shrinking"}, {"timestamp", timestamp.str()}};

    // Config section
    j["config"]["experiment"] = {{"initial_memory_kb", config.initial_memory_kb},
                                 {"max_memory_kb", config.max_memory_kb},
                                 {"final_memory_kb", config.final_memory_kb},
                                 {"shrinking_interval", config.shrinking_interval},
                                 {"memory_decrement_kb", config.memory_decrement_kb},
                                 {"repetitions", config.repetitions},
                                 {"dataset_type", config.dataset_type},
                                 {"total_items", config.total_items},
                                 {"stream_size", config.stream_size},
                                 {"stream_diversity", config.stream_diversity},
                                 {"zipf_param", config.zipf_param}};

    j["config"]["base_sketch_config"]["resketch"] = {{"depth", rs_config.depth}, {"kll_k", rs_config.kll_k}};
    j["config"]["base_sketch_config"]["geometric"] = {{"depth", gs_config.depth}};

    // Results section
    json results_json;
    for (const auto &[sketch_name, repetitions] : all_results) {
        json sketch_reps = json::array();

        for (uint32_t rep = 0; rep < repetitions.size(); ++rep) {
            json rep_json;
            rep_json["repetition_id"] = rep;

            json checkpoints_array = json::array();
            for (const auto &cp : repetitions[rep]) {
                checkpoints_array.push_back({{"items_processed", cp.items_processed},
                                             {"throughput_mops", cp.throughput_mops},
                                             {"query_throughput_mops", cp.query_throughput_mops},
                                             {"memory_kb", cp.memory_kb},
                                             {"are", cp.are},
                                             {"aae", cp.aae},
                                             {"is_warmup", cp.is_warmup},
                                             {"geometric_cannot_shrink", cp.geometric_cannot_shrink}});
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

void run_shrinking_experiment(const ShrinkingConfig &config, const ReSketchConfig &rs_config, const GeometricSketchConfig &gs_config) {
    cout << config << endl;
    cout << rs_config << endl;
    cout << gs_config << endl;

    map<string, vector<vector<Checkpoint>>> all_results;
    all_results["ReSketch"] = vector<vector<Checkpoint>>(config.repetitions);
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
                cerr << "Error: Failed to read CAIDA data from " << config.caida_path << endl;
                continue;
            }
        } else {
            cerr << "Error: Unknown dataset type: " << config.dataset_type << endl;
            continue;
        }

        uint64_t num_repeats = (config.total_items + base_data.size() - 1) / base_data.size();

        cout << "Base dataset size: " << base_data.size() << endl;
        cout << "Will process " << config.total_items << " items total (repeating dataset " << num_repeats << " times)" << endl;

        // Phase 1: Warmup expansion (initial -> max memory) - NO DATA PROCESSING
        cout << "\n--- Phase 1: Warmup Expansion (" << config.initial_memory_kb << " KB -> " << config.max_memory_kb << " KB) ---" << endl;

        uint64_t initial_memory_bytes = (uint64_t) config.initial_memory_kb * 1024;
        uint64_t max_memory_bytes = (uint64_t) config.max_memory_kb * 1024;

        // Initialize sketches at INITIAL width
        uint32_t rs_initial_width = calculate_width_from_memory_resketch(initial_memory_bytes, rs_config.depth, rs_config.kll_k);
        uint32_t gs_initial_width = calculate_width_from_memory_geometric(initial_memory_bytes, gs_config.depth);

        cout << "Initial widths: RS=" << rs_initial_width << ", GS=" << gs_initial_width << endl;

        ReSketchConfig rs_conf = rs_config;
        rs_conf.width = rs_initial_width;
        ReSketchV2 rs_sketch(rs_conf);

        GeometricSketchConfig gs_conf = gs_config;
        gs_conf.width = gs_initial_width;
        GeometricSketchWrapper gs_sketch(gs_conf);

        uint64_t rs_initial_memory_kb = rs_sketch.get_max_memory_usage() / 1024;
        uint64_t gs_initial_memory_kb = gs_sketch.get_max_memory_usage() / 1024;
        cout << "Actual initial memory: RS=" << rs_initial_memory_kb << " KB (target: " << config.initial_memory_kb << " KB), GS=" << gs_initial_memory_kb << " KB" << endl;

        // Expand to max width (warmup phase)
        uint32_t rs_max_width = calculate_width_from_memory_resketch(max_memory_bytes, rs_config.depth, rs_config.kll_k);
        uint32_t gs_max_width = calculate_width_from_memory_geometric(max_memory_bytes, gs_config.depth);

        cout << "Expanding to max widths: RS=" << rs_max_width << ", GS=" << gs_max_width << endl;

        rs_sketch.expand(rs_max_width);
        rs_conf.width = rs_max_width;

        gs_sketch.expand(gs_max_width);
        gs_conf.width = gs_max_width;

        uint64_t rs_current_memory_kb = rs_sketch.get_max_memory_usage() / 1024;
        uint64_t gs_current_memory_kb = gs_sketch.get_max_memory_usage() / 1024;

        cout << "Sketches expanded to max memory: RS=" << rs_current_memory_kb << " KB, GS=" << gs_current_memory_kb << " KB" << endl;

        // Phase 2: Shrinking while processing data
        cout << "\n--- Phase 2: Shrinking While Processing Data ---" << endl;

        Timer timer;
        uint64_t items_processed = 0;

        uint64_t final_memory_bytes = (uint64_t) config.final_memory_kb * 1024;
        uint64_t memory_decrement_bytes = (uint64_t) config.memory_decrement_kb * 1024;

        uint64_t rs_target_memory_bytes = max_memory_bytes;
        uint64_t gs_target_memory_bytes = max_memory_bytes;

        bool gs_cannot_shrink = false;

        while (items_processed < config.total_items) {
            uint64_t chunk_size = min((uint64_t) config.shrinking_interval, config.total_items - items_processed);
            uint64_t chunk_start = items_processed;
            uint64_t chunk_end = chunk_start + chunk_size;

            // Process chunk for ReSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { rs_sketch.update(base_data[i % base_data.size()]); }
            double rs_duration = timer.stop_s();

            // Process chunk for GeometricSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { gs_sketch.update(base_data[i % base_data.size()]); }
            double gs_duration = timer.stop_s();

            items_processed += chunk_size;

            // Calculate true frequencies for items processed up to this checkpoint
            map<uint64_t, uint64_t> true_freqs_at_checkpoint;
            for (uint64_t i = 0; i < items_processed; ++i) { true_freqs_at_checkpoint[base_data[i % base_data.size()]]++; }

            // Record checkpoint
            Checkpoint rs_cp, gs_cp;
            rs_cp.items_processed = gs_cp.items_processed = items_processed;

            rs_cp.throughput_mops = (rs_duration > 0) ? (chunk_size / rs_duration / 1e6) : 0;
            gs_cp.throughput_mops = (gs_duration > 0) ? (chunk_size / gs_duration / 1e6) : 0;

            rs_cp.memory_kb = rs_sketch.get_max_memory_usage() / 1024;
            gs_cp.memory_kb = gs_sketch.get_max_memory_usage() / 1024;

            rs_cp.are = calculate_are_all_items(rs_sketch, true_freqs_at_checkpoint);
            gs_cp.are = calculate_are_all_items(gs_sketch, true_freqs_at_checkpoint);

            rs_cp.aae = calculate_aae_all_items(rs_sketch, true_freqs_at_checkpoint);
            gs_cp.aae = calculate_aae_all_items(gs_sketch, true_freqs_at_checkpoint);

            rs_cp.is_warmup = false;
            gs_cp.is_warmup = false;

            rs_cp.geometric_cannot_shrink = false;
            gs_cp.geometric_cannot_shrink = gs_cannot_shrink;

            // Measure query throughput on all unique items
            vector<uint64_t> unique_items;
            unique_items.reserve(true_freqs_at_checkpoint.size());
            for (const auto &[item, freq] : true_freqs_at_checkpoint) { unique_items.push_back(item); }
            uint64_t num_queries = unique_items.size();

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

            all_results["ReSketch"][rep].push_back(rs_cp);
            all_results["GeometricSketch"][rep].push_back(gs_cp);

            // Print checkpoint summary
            cout << "Checkpoint at " << items_processed << " items:" << endl;
            cout << "  ReSketch:        Memory=" << rs_cp.memory_kb << " KB, ARE=" << rs_cp.are << ", AAE=" << rs_cp.aae << endl;
            cout << "  GeometricSketch: Memory=" << gs_cp.memory_kb << " KB, ARE=" << gs_cp.are << ", AAE=" << gs_cp.aae;
            if (gs_cannot_shrink) { cout << " [Cannot shrink further]"; }
            cout << endl;

            // Shrink sketches for next iteration (if not at minimum)
            if (items_processed < config.total_items) {
                // Shrink ReSketch
                rs_target_memory_bytes = max((uint64_t) final_memory_bytes, rs_target_memory_bytes - memory_decrement_bytes);
                uint32_t rs_new_width = calculate_width_from_memory_resketch(rs_target_memory_bytes, rs_config.depth, rs_config.kll_k);

                if (rs_new_width < rs_conf.width) {
                    cout << "  Shrinking ReSketch to " << rs_new_width << " width (target: " << (rs_target_memory_bytes / 1024) << " KB)" << endl;
                    rs_sketch.shrink(rs_new_width);
                    rs_conf.width = rs_new_width;
                }

                // Shrink GeometricSketch (only if it can still shrink)
                if (!gs_cannot_shrink) {
                    gs_target_memory_bytes = max((uint64_t) initial_memory_bytes, gs_target_memory_bytes - memory_decrement_bytes);
                    uint32_t gs_new_width = calculate_width_from_memory_geometric(gs_target_memory_bytes, gs_config.depth);

                    if (gs_new_width < gs_conf.width) {
                        cout << "  Shrinking GeometricSketch to " << gs_new_width << " width (target: " << (gs_target_memory_bytes / 1024) << " KB)" << endl;
                        gs_sketch.shrink(gs_new_width);
                        gs_conf.width = gs_new_width;
                    }

                    // Check if GeometricSketch reached its limit (initial memory)
                    if (gs_target_memory_bytes <= initial_memory_bytes) {
                        cout << "  GeometricSketch reached initial memory (" << config.initial_memory_kb << " KB) and cannot shrink further!" << endl;
                        gs_cannot_shrink = true;
                    }
                }
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

    export_to_json(output_file, config, rs_config, gs_config, all_results);
}

int main(int argc, char **argv) {
    ConfigParser parser;
    ShrinkingConfig shrink_config;
    ReSketchConfig rs_config;
    GeometricSketchConfig gs_config;

    ShrinkingConfig::add_params_to_config_parser(shrink_config, parser);
    ReSketchConfig::add_params_to_config_parser(rs_config, parser);
    GeometricSketchConfig::add_params_to_config_parser(gs_config, parser);

    if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h")) {
        parser.PrintUsage();
        return 0;
    }

    Status s = parser.ParseCommandLine(argc, argv);
    if (!s.IsOK()) {
        fprintf(stderr, "%s\n", s.ToString().c_str());
        return -1;
    }

    run_shrinking_experiment(shrink_config, rs_config, gs_config);

    return 0;
}
