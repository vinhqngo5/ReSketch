#define DOCTEST_CONFIG_IMPLEMENT
#include "frequency_summary/frequency_summary_config.hpp"

#include "frequency_summary/count_min_sketch.hpp"
#include "frequency_summary/dynamic_sketch_wrapper.hpp"
#include "frequency_summary/geometric_sketch_wrapper.hpp"
#include "frequency_summary/resketchv2.hpp"
#include "common.hpp"

#include "utils/ConfigParser.hpp"

#include <json/json.hpp>

#include "doctest.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace std;
using json = nlohmann::json;

// Expansion-Shrinking Experiment Config
struct ExpansionShrinkingConfig
{
    // Memory parameters
    uint32_t m0_kb = 32;   // Starting memory for expansion, minimum for GS shrinking
    uint32_t m2_kb = 16;   // Final memory for ReSketch shrinking target
    // M1: Auto-calculated after expansion (actual memory reached after expansion)

    // Expansion phase
    uint32_t expansion_interval = 100000;
    uint32_t memory_increment_kb = 32;

    // Shrinking phase
    uint64_t shrinking_items = 2000000;   // Items to process during shrinking phases
    // Note: Shrinking checkpoints and intervals auto-calculated from M1, M2, shrinking_items

    // General
    uint32_t repetitions = 10;
    string dataset_type = "zipf";
    string caida_path = "data/CAIDA/only_ip";
    uint64_t expansion_items = 10000000;
    uint64_t stream_size = 10000000;
    uint64_t stream_diversity = 10000;
    float zipf_param = 1.1;
    string output_file = "output/expansion_shrinking_results.json";

    static void add_params_to_config_parser(ExpansionShrinkingConfig &config, ConfigParser &parser)
    {
        parser.AddParameter(new UnsignedInt32Parameter("app.m0_kb", "32", &config.m0_kb, false, "M0: Starting memory for expansion, minimum for GS"));
        parser.AddParameter(new UnsignedInt32Parameter("app.m2_kb", "16", &config.m2_kb, false, "M2: Final ReSketch shrinking target"));
        parser.AddParameter(new UnsignedInt32Parameter("app.expansion_interval", "100000", &config.expansion_interval, false, "Items between expansions"));
        parser.AddParameter(new UnsignedInt32Parameter("app.memory_increment_kb", "32", &config.memory_increment_kb, false, "Memory increment per expansion in KB"));
        parser.AddParameter(
            new UnsignedInt64Parameter("app.shrinking_items", "2000000", &config.shrinking_items, false, "Total items to process during shrinking (checkpoints auto-calculated)"));
        parser.AddParameter(new UnsignedInt32Parameter("app.repetitions", "10", &config.repetitions, false, "Number of experiment repetitions"));
        parser.AddParameter(new StringParameter("app.dataset_type", "zipf", &config.dataset_type, false, "Dataset type: zipf or caida"));
        parser.AddParameter(new StringParameter("app.caida_path", "data/CAIDA/only_ip", &config.caida_path, false, "Path to CAIDA data file"));
        parser.AddParameter(new UnsignedInt64Parameter("app.expansion_items", "10000000", &config.expansion_items, false, "Total items for expansion phase"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", "10000000", &config.stream_size, false, "Dataset size for zipf generation"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", "1000000", &config.stream_diversity, false, "Unique items in stream (zipf)"));
        parser.AddParameter(new FloatParameter("app.zipf", "1.1", &config.zipf_param, false, "Zipfian param 'a'"));
        parser.AddParameter(new StringParameter("app.output_file", "output/expansion_shrinking_results.json", &config.output_file, false, "Output JSON file path"));
    }

    friend std::ostream &operator<<(std::ostream &os, const ExpansionShrinkingConfig &config)
    {
        os << "\n=== Expansion-Shrinking Experiment Configuration ===\n";
        os << "Memory M0 (start expansion, min GS): " << config.m0_kb << " KB\n";
        os << "Memory M1 (end expansion, start shrinking): auto-calculated\n";
        os << "Memory M2 (final ReSketch): " << config.m2_kb << " KB\n";
        os << "Expansion Interval: " << config.expansion_interval << " items\n";
        os << "Memory Increment: " << config.memory_increment_kb << " KB\n";
        os << "Shrinking Items: " << config.shrinking_items << " (checkpoints auto-calculated)\n";
        os << "Repetitions: " << config.repetitions << "\n";
        os << "Dataset: " << config.dataset_type << "\n";
        if (config.dataset_type == "caida") { os << "CAIDA Path: " << config.caida_path << "\n"; }
        os << "Expansion Items: " << config.expansion_items << "\n";
        os << "Dataset Size: " << config.stream_size << "\n";
        if (config.dataset_type == "zipf")
        {
            os << "Stream Diversity: " << config.stream_diversity << "\n";
            os << "Zipf Parameter: " << config.zipf_param << "\n";
        }
        os << "Output File: " << config.output_file << "\n";
        return os;
    }
};

// Checkpoint data
struct Checkpoint
{
    string phase;   // "expansion", "shrinking_no_data", "shrinking_with_data"
    uint64_t items_processed;
    uint64_t items_in_phase;   // Items processed within current phase
    double throughput_mops;
    double query_throughput_mops;
    uint64_t memory_kb;
    double are;
    double aae;
    double are_variance;
    double aae_variance;
    bool geometric_cannot_shrink;   // True when GeometricSketch reached M0 limit
};

void export_to_json(
    const string &filename, const ExpansionShrinkingConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config, const GeometricSketchConfig &gs_config,
    const DynamicSketchConfig &ds_config, const map<string, vector<vector<Checkpoint>>> &all_results)
{
    create_directory(filename);

    json j;

    // Metadata section
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    gmtime_r(&now_time_t, &tm_now);
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%SZ");

    j["metadata"] = {{"experiment_type", "expansion_shrinking"}, {"timestamp", timestamp.str()}};

    // Config section
    j["config"]["experiment"] = {
        {"m0_kb", config.m0_kb},
        {"m2_kb", config.m2_kb},
        {"expansion_interval", config.expansion_interval},
        {"memory_increment_kb", config.memory_increment_kb},
        {"shrinking_items", config.shrinking_items},
        {"repetitions", config.repetitions},
        {"dataset_type", config.dataset_type},
        {"expansion_items", config.expansion_items},
        {"stream_size", config.stream_size},
        {"stream_diversity", config.stream_diversity},
        {"zipf_param", config.zipf_param}};

    j["config"]["base_sketch_config"]["countmin"] = {{"depth", cm_config.depth}};
    j["config"]["base_sketch_config"]["resketch"] = {{"depth", rs_config.depth}, {"kll_k", rs_config.kll_k}};
    j["config"]["base_sketch_config"]["geometric"] = {{"depth", gs_config.depth}};
    j["config"]["base_sketch_config"]["dynamic"] = {{"depth", ds_config.depth}};

    // Results section
    json results_json;
    for (const auto &[sketch_name, repetitions] : all_results)
    {
        json sketch_reps = json::array();

        for (uint32_t rep = 0; rep < repetitions.size(); ++rep)
        {
            json rep_json;
            rep_json["repetition_id"] = rep;

            json checkpoints_array = json::array();
            for (const auto &cp : repetitions[rep])
            {
                json cp_json;
                cp_json["phase"] = cp.phase;
                cp_json["items_processed"] = cp.items_processed;
                cp_json["items_in_phase"] = cp.items_in_phase;
                cp_json["throughput_mops"] = cp.throughput_mops;
                cp_json["query_throughput_mops"] = cp.query_throughput_mops;
                cp_json["memory_kb"] = cp.memory_kb;
                cp_json["are"] = cp.are;
                cp_json["aae"] = cp.aae;
                cp_json["are_variance"] = cp.are_variance;
                cp_json["aae_variance"] = cp.aae_variance;
                cp_json["geometric_cannot_shrink"] = cp.geometric_cannot_shrink;
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
    if (!out.is_open())
    {
        cerr << "Error: Cannot open output file: " << filename << endl;
        return;
    }

    out << j.dump(2);
    out.close();

    cout << "\nResults exported to: " << filename << endl;
}

// Calculate memory checkpoints for shrinking: M1_rounded -> power-of-2 divisions -> M2
vector<uint64_t> calculate_shrinking_memory_checkpoints(uint64_t m1_bytes, uint64_t m2_bytes)
{
    vector<uint64_t> checkpoints;

    // Find next power-of-2 below M1 (floor of log2)
    uint64_t m1_log2 = (uint64_t) floor(log2((double) m1_bytes));
    uint64_t m1_power_of_2 = 1ULL << m1_log2;   // 2^m1_log2

    // Generate power-of-2 checkpoints by halving: next_pow2 -> next_pow2/2 -> ... -> M2
    uint64_t current = m1_power_of_2;
    while (current >= m2_bytes)
    {
        checkpoints.push_back(current);
        current /= 2;
    }

    return checkpoints;
}

// Calculate items per interval using geometric distribution -> Each interval gets half the items of the previous interval
vector<uint64_t> calculate_geometric_item_intervals(uint64_t total_items, uint32_t num_intervals)
{
    vector<uint64_t> intervals;

    if (num_intervals == 0) return intervals;
    if (num_intervals == 1)
    {
        intervals.push_back(total_items);
        return intervals;
    }

    // Geometric series: sum = total_items = a * (2^n - 1)
    // where a is the last (smallest) interval
    // So: a = total_items / (2^n - 1)

    uint64_t denominator = (1ULL << num_intervals) - 1;
    double smallest_interval_float = (double) total_items / (double) denominator;

    for (uint32_t i = 0; i < num_intervals; i++)
    {
        uint64_t power = num_intervals - 1 - i;
        double interval_float = smallest_interval_float * (1ULL << power);
        intervals.push_back((uint64_t) round(interval_float));
    }

    uint64_t sum = 0;
    for (uint32_t i = 0; i < num_intervals - 1; i++) { sum += intervals[i]; }
    intervals[num_intervals - 1] = total_items - sum;

    return intervals;
}

void run_expansion_shrinking_experiment(
    const ExpansionShrinkingConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config, const GeometricSketchConfig &gs_config,
    const DynamicSketchConfig &ds_config)
{
    cout << config << endl;
    cout << cm_config << endl;
    cout << rs_config << endl;
    cout << gs_config << endl;
    cout << ds_config << endl;

    map<string, vector<vector<Checkpoint>>> all_results;
    all_results["CountMin"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["ReSketch"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["ReSketch_ShrinkNoData"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["ReSketch_ShrinkWithData"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["StaticReSketch"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["DynamicSketch"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["GeometricSketch"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["GeometricSketch_ShrinkNoData"] = vector<vector<Checkpoint>>(config.repetitions);
    all_results["GeometricSketch_ShrinkWithData"] = vector<vector<Checkpoint>>(config.repetitions);

    for (uint32_t rep = 0; rep < config.repetitions; ++rep)
    {
        cout << "\n=== Repetition " << (rep + 1) << "/" << config.repetitions << " ===" << endl;

        // Generate or load base dataset
        vector<uint64_t> base_data;
        if (config.dataset_type == "zipf")
        {
            cout << "Generating Zipf data..." << endl;
            base_data = generate_zipf_data(config.stream_size, config.stream_diversity, config.zipf_param);
        }
        else if (config.dataset_type == "caida")
        {
            cout << "Reading CAIDA data..." << endl;
            base_data = read_caida_data(config.caida_path, config.stream_size);
            if (base_data.empty())
            {
                cerr << "Error: Failed to read CAIDA data. Skipping repetition." << endl;
                continue;
            }
        }
        else
        {
            cerr << "Error: Unknown dataset type: " << config.dataset_type << endl;
            continue;
        }

        cout << "Base dataset size: " << base_data.size() << endl;

        uint64_t m0_bytes = (uint64_t) config.m0_kb * 1024;
        uint64_t m2_bytes = (uint64_t) config.m2_kb * 1024;
        uint64_t memory_increment_bytes = (uint64_t) config.memory_increment_kb * 1024;

        // Calculate estimated M1 (final expansion memory)
        uint64_t num_expansion_steps = (config.expansion_items + config.expansion_interval - 1) / config.expansion_interval;
        uint64_t estimated_m1_bytes = m0_bytes + (num_expansion_steps * memory_increment_bytes);

        cout << "\n=== MEMORY TARGET ESTIMATES ===" << endl;
        cout << "M0 (start): " << config.m0_kb << " KB" << endl;
        cout << "M1 (estimated end of expansion): ~" << estimated_m1_bytes / 1024 << " KB" << endl;
        cout << "M2 (final ReSketch target): " << config.m2_kb << " KB" << endl;
        cout << "================================\n" << endl;

        // PHASE 1: EXPANSION (M0 -> auto-calculated M1)
        cout << "\n--- Phase 1: Expansion (" << config.m0_kb << " KB -> M1 auto-calculated) ---" << endl;

        // Initialize all sketches at M0
        uint32_t cm_width = calculate_width_from_memory_cm(m0_bytes, cm_config.depth);
        uint32_t rs_width = calculate_width_from_memory_resketch(m0_bytes, rs_config.depth, rs_config.kll_k);
        uint32_t gs_width = calculate_width_from_memory_geometric(m0_bytes, gs_config.depth);
        uint32_t ds_width = calculate_width_from_memory_dynamic(m0_bytes, ds_config.depth);

        cout << "Initial widths (M0): CM=" << cm_width << ", RS=" << rs_width << ", GS=" << gs_width << ", DS=" << ds_width << endl;

        CountMinConfig cm_conf = cm_config;
        cm_conf.width = cm_width;
        CountMinSketch cm_sketch(cm_conf);

        ReSketchConfig rs_conf = rs_config;
        rs_conf.width = rs_width;
        ReSketchV2 rs_sketch(rs_conf);

        ReSketchConfig static_rs_conf = rs_config;
        static_rs_conf.width = rs_width;
        ReSketchV2 static_rs_sketch(static_rs_conf);

        GeometricSketchConfig gs_conf = gs_config;
        gs_conf.width = gs_width;
        GeometricSketchWrapper gs_sketch(gs_conf);

        DynamicSketchConfig ds_conf = ds_config;
        ds_conf.width = ds_width;
        DynamicSketchWrapper ds_sketch(ds_conf);

        // Prepare shrinking sketches -> they will be updated during expansion but won't have metrics computed until shrinking phase
        ReSketchConfig rs_shrink_no_data_conf = rs_config;
        rs_shrink_no_data_conf.width = rs_width;
        ReSketchV2 rs_shrink_no_data(rs_shrink_no_data_conf);

        ReSketchConfig rs_shrink_with_data_conf = rs_config;
        rs_shrink_with_data_conf.width = rs_width;
        ReSketchV2 rs_shrink_with_data(rs_shrink_with_data_conf);

        GeometricSketchConfig gs_shrink_no_data_conf = gs_config;
        gs_shrink_no_data_conf.width = gs_width;
        GeometricSketchWrapper gs_shrink_no_data(gs_shrink_no_data_conf);

        GeometricSketchConfig gs_shrink_with_data_conf = gs_config;
        gs_shrink_with_data_conf.width = gs_width;
        GeometricSketchWrapper gs_shrink_with_data(gs_shrink_with_data_conf);

        // Track DynamicSketch expansion: accumulate budget until we can double
        uint64_t ds_accumulated_budget = 0;
        uint64_t ds_last_expansion_size = m0_bytes;

        Timer timer;
        uint64_t items_processed = 0;
        uint64_t current_target_memory = m0_bytes;

        while (items_processed < config.expansion_items)
        {
            uint64_t chunk_size = min((uint64_t) config.expansion_interval, config.expansion_items - items_processed);
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

            // Process chunk for StaticReSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { static_rs_sketch.update(base_data[i % base_data.size()]); }
            double static_rs_duration = timer.stop_s();

            // Process chunk for GeometricSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { gs_sketch.update(base_data[i % base_data.size()]); }
            double gs_duration = timer.stop_s();

            // Process chunk for DynamicSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { ds_sketch.update(base_data[i % base_data.size()]); }
            double ds_duration = timer.stop_s();

            // Update shrinking sketches (no metrics computed)
            for (uint64_t i = chunk_start; i < chunk_end; ++i)
            {
                rs_shrink_no_data.update(base_data[i % base_data.size()]);
                rs_shrink_with_data.update(base_data[i % base_data.size()]);
                gs_shrink_no_data.update(base_data[i % base_data.size()]);
                gs_shrink_with_data.update(base_data[i % base_data.size()]);
            }

            items_processed += chunk_size;

            // Calculate true frequencies
            map<uint64_t, uint64_t> true_freqs;
            for (uint64_t i = 0; i < items_processed; ++i) { true_freqs[base_data[i % base_data.size()]]++; }

            // Record checkpoint
            Checkpoint cm_cp, rs_cp, static_rs_cp, gs_cp, ds_cp;
            cm_cp.phase = rs_cp.phase = static_rs_cp.phase = gs_cp.phase = ds_cp.phase = "expansion";
            cm_cp.items_processed = rs_cp.items_processed = static_rs_cp.items_processed = gs_cp.items_processed = ds_cp.items_processed = items_processed;
            cm_cp.items_in_phase = rs_cp.items_in_phase = static_rs_cp.items_in_phase = gs_cp.items_in_phase = ds_cp.items_in_phase = items_processed;
            cm_cp.geometric_cannot_shrink = rs_cp.geometric_cannot_shrink = static_rs_cp.geometric_cannot_shrink = gs_cp.geometric_cannot_shrink = ds_cp.geometric_cannot_shrink =
                false;

            cm_cp.throughput_mops = (cm_duration > 0) ? (chunk_size / cm_duration / 1e6) : 0;
            rs_cp.throughput_mops = (rs_duration > 0) ? (chunk_size / rs_duration / 1e6) : 0;
            static_rs_cp.throughput_mops = (static_rs_duration > 0) ? (chunk_size / static_rs_duration / 1e6) : 0;
            gs_cp.throughput_mops = (gs_duration > 0) ? (chunk_size / gs_duration / 1e6) : 0;
            ds_cp.throughput_mops = (ds_duration > 0) ? (chunk_size / ds_duration / 1e6) : 0;

            cm_cp.memory_kb = cm_sketch.get_max_memory_usage() / 1024;
            rs_cp.memory_kb = rs_sketch.get_max_memory_usage() / 1024;
            static_rs_cp.memory_kb = static_rs_sketch.get_max_memory_usage() / 1024;
            gs_cp.memory_kb = gs_sketch.get_max_memory_usage() / 1024;
            ds_cp.memory_kb = ds_sketch.get_max_memory_usage() / 1024;

            cm_cp.are = calculate_are_all_items(cm_sketch, true_freqs);
            rs_cp.are = calculate_are_all_items(rs_sketch, true_freqs);
            static_rs_cp.are = calculate_are_all_items(static_rs_sketch, true_freqs);
            gs_cp.are = calculate_are_all_items(gs_sketch, true_freqs);
            ds_cp.are = calculate_are_all_items(ds_sketch, true_freqs);

            cm_cp.aae = calculate_aae_all_items(cm_sketch, true_freqs);
            rs_cp.aae = calculate_aae_all_items(rs_sketch, true_freqs);
            static_rs_cp.aae = calculate_aae_all_items(static_rs_sketch, true_freqs);
            gs_cp.aae = calculate_aae_all_items(gs_sketch, true_freqs);
            ds_cp.aae = calculate_aae_all_items(ds_sketch, true_freqs);

            cm_cp.are_variance = calculate_are_variance(cm_sketch, true_freqs, cm_cp.are);
            rs_cp.are_variance = calculate_are_variance(rs_sketch, true_freqs, rs_cp.are);
            static_rs_cp.are_variance = calculate_are_variance(static_rs_sketch, true_freqs, static_rs_cp.are);
            gs_cp.are_variance = calculate_are_variance(gs_sketch, true_freqs, gs_cp.are);
            ds_cp.are_variance = calculate_are_variance(ds_sketch, true_freqs, ds_cp.are);

            cm_cp.aae_variance = calculate_aae_variance(cm_sketch, true_freqs, cm_cp.aae);
            rs_cp.aae_variance = calculate_aae_variance(rs_sketch, true_freqs, rs_cp.aae);
            static_rs_cp.aae_variance = calculate_aae_variance(static_rs_sketch, true_freqs, static_rs_cp.aae);
            gs_cp.aae_variance = calculate_aae_variance(gs_sketch, true_freqs, gs_cp.aae);
            ds_cp.aae_variance = calculate_aae_variance(ds_sketch, true_freqs, ds_cp.aae);

            // Query throughput
            timer.start();
            for (const auto &[item, freq] : true_freqs) { volatile double q = cm_sketch.estimate(item); }
            cm_cp.query_throughput_mops = timer.stop_s() > 0 ? (true_freqs.size() / timer.stop_s() / 1e6) : 0;

            timer.start();
            for (const auto &[item, freq] : true_freqs) { volatile double q = rs_sketch.estimate(item); }
            rs_cp.query_throughput_mops = timer.stop_s() > 0 ? (true_freqs.size() / timer.stop_s() / 1e6) : 0;

            timer.start();
            for (const auto &[item, freq] : true_freqs) { volatile double q = static_rs_sketch.estimate(item); }
            static_rs_cp.query_throughput_mops = timer.stop_s() > 0 ? (true_freqs.size() / timer.stop_s() / 1e6) : 0;

            timer.start();
            for (const auto &[item, freq] : true_freqs) { volatile double q = gs_sketch.estimate(item); }
            gs_cp.query_throughput_mops = timer.stop_s() > 0 ? (true_freqs.size() / timer.stop_s() / 1e6) : 0;

            timer.start();
            for (const auto &[item, freq] : true_freqs) { volatile double q = ds_sketch.estimate(item); }
            ds_cp.query_throughput_mops = timer.stop_s() > 0 ? (true_freqs.size() / timer.stop_s() / 1e6) : 0;

            all_results["CountMin"][rep].push_back(cm_cp);
            all_results["ReSketch"][rep].push_back(rs_cp);
            all_results["StaticReSketch"][rep].push_back(static_rs_cp);
            all_results["GeometricSketch"][rep].push_back(gs_cp);
            all_results["DynamicSketch"][rep].push_back(ds_cp);

            // Print checkpoint info
            cout << "Expansion checkpoint at " << items_processed << " items:" << endl;
            cout << "  CM: " << cm_cp.throughput_mops << " Mops, Query: " << cm_cp.query_throughput_mops << " Mops, " << cm_cp.memory_kb << " KB, ARE=" << cm_cp.are
                 << ", AAE=" << cm_cp.aae << endl;
            cout << "  RS: " << rs_cp.throughput_mops << " Mops, Query: " << rs_cp.query_throughput_mops << " Mops, " << rs_cp.memory_kb << " KB, ARE=" << rs_cp.are
                 << ", AAE=" << rs_cp.aae << endl;
            cout << "  Static RS: " << static_rs_cp.throughput_mops << " Mops, Query: " << static_rs_cp.query_throughput_mops << " Mops, " << static_rs_cp.memory_kb
                 << " KB, ARE=" << static_rs_cp.are << ", AAE=" << static_rs_cp.aae << endl;
            cout << "  GS: " << gs_cp.throughput_mops << " Mops, Query: " << gs_cp.query_throughput_mops << " Mops, " << gs_cp.memory_kb << " KB, ARE=" << gs_cp.are
                 << ", AAE=" << gs_cp.aae << endl;
            cout << "  DS: " << ds_cp.throughput_mops << " Mops, Query: " << ds_cp.query_throughput_mops << " Mops, " << ds_cp.memory_kb << " KB, ARE=" << ds_cp.are
                 << ", AAE=" << ds_cp.aae << endl;

            // Expand sketches based on interval (only if more items to process)
            if (items_processed < config.expansion_items)
            {
                current_target_memory += memory_increment_bytes;

                uint32_t new_rs_width = calculate_width_from_memory_resketch(current_target_memory, rs_config.depth, rs_config.kll_k);
                uint32_t new_gs_width = calculate_width_from_memory_geometric(current_target_memory, gs_config.depth);

                // CountMin: cannot expand (do nothing)

                // ReSketch: expand by memory_increment_kb
                rs_sketch.expand(new_rs_width);
                rs_conf.width = new_rs_width;

                // GeometricSketch: expand by memory_increment_kb
                gs_sketch.expand(new_gs_width);
                gs_conf.width = new_gs_width;

                // Expand shrinking sketches too
                rs_shrink_no_data.expand(new_rs_width);
                rs_shrink_no_data_conf.width = new_rs_width;

                rs_shrink_with_data.expand(new_rs_width);
                rs_shrink_with_data_conf.width = new_rs_width;

                gs_shrink_no_data.expand(new_gs_width);
                gs_shrink_no_data_conf.width = new_gs_width;

                gs_shrink_with_data.expand(new_gs_width);
                gs_shrink_with_data_conf.width = new_gs_width;

                // DynamicSketch doubles when budget allows
                ds_accumulated_budget += memory_increment_bytes;
                if (ds_accumulated_budget >= ds_last_expansion_size)
                {
                    // DynamicSketch internally tracks width and doubles it
                    ds_sketch.expand(ds_last_expansion_size * 2 / (ds_config.depth * sizeof(uint32_t)));
                    ds_accumulated_budget = 0;
                    ds_last_expansion_size *= 2;
                }
            }
        }

        cout << "Expansion phase complete. Items processed: " << items_processed << endl;
        cout << "Final memories: CM=" << cm_sketch.get_max_memory_usage() / 1024 << " KB, RS=" << rs_sketch.get_max_memory_usage() / 1024
             << " KB, GS=" << gs_sketch.get_max_memory_usage() / 1024 << " KB, DS=" << ds_sketch.get_max_memory_usage() / 1024 << " KB" << endl;

        // Auto-calculate M1 from actual expansion results (use shrinking sketches to get final expanded memory)
        uint64_t actual_m1_rs = rs_shrink_no_data.get_max_memory_usage();
        uint64_t actual_m1_gs = gs_shrink_no_data.get_max_memory_usage();
        cout << "\n=== ACTUAL MEMORY TARGETS ===" << endl;
        cout << "M1 (actual after expansion): RS=" << actual_m1_rs / 1024 << " KB, GS=" << actual_m1_gs / 1024 << " KB" << endl;
        cout << "M2 (final ReSketch target): " << config.m2_kb << " KB" << endl;
        cout << "==============================" << endl;

        // PHASE 2: SHRINKING WITHOUT DATA
        cout << "\n--- Phase 2: Shrinking Without Data (RS: M1=" << actual_m1_rs / 1024 << " KB -> M2=" << m2_bytes / 1024 << " KB, GS: M1=" << actual_m1_gs / 1024
             << " KB -> M0=" << config.m0_kb << " KB) ---" << endl;

        // Calculate true frequencies from expansion phase (for accuracy measurement)
        map<uint64_t, uint64_t> expansion_true_freqs;
        for (uint64_t i = 0; i < items_processed; ++i) { expansion_true_freqs[base_data[i % base_data.size()]]++; }

        // Calculate memory checkpoints for shrinking (power-of-2 multiples of M2)
        vector<uint64_t> rs_memory_checkpoints = calculate_shrinking_memory_checkpoints(actual_m1_rs, m2_bytes);
        vector<uint64_t> gs_memory_checkpoints = calculate_shrinking_memory_checkpoints(actual_m1_gs, m0_bytes);

        cout << "ReSketch checkpoints: ";
        for (auto cp : rs_memory_checkpoints) cout << cp / 1024 << " KB ";
        cout << endl;
        cout << "GeometricSketch checkpoints: ";
        for (auto cp : gs_memory_checkpoints) cout << cp / 1024 << " KB ";
        cout << endl;

        // Loop through checkpoints and shrink (M1 already recorded in expansion phase)
        size_t max_checkpoints = max(rs_memory_checkpoints.size(), gs_memory_checkpoints.size());
        for (size_t i = 0; i < max_checkpoints; ++i)
        {
            bool gs_cannot_shrink = false;

            // Shrink ReSketch if checkpoint exists
            if (i < rs_memory_checkpoints.size())
            {
                uint64_t target_memory = rs_memory_checkpoints[i];
                uint32_t new_rs_width = calculate_width_from_memory_resketch(target_memory, rs_config.depth, rs_config.kll_k);
                if (new_rs_width < rs_shrink_no_data_conf.width)
                {
                    rs_shrink_no_data.shrink(new_rs_width);
                    rs_shrink_no_data_conf.width = new_rs_width;
                }
            }

            // Shrink GeometricSketch if checkpoint exists
            if (i < gs_memory_checkpoints.size())
            {
                uint64_t target_memory = gs_memory_checkpoints[i];
                uint32_t new_gs_width = calculate_width_from_memory_geometric(target_memory, gs_config.depth);

                if (new_gs_width >= gs_shrink_no_data_conf.width) { gs_cannot_shrink = true; }
                else
                {
                    gs_shrink_no_data.shrink(new_gs_width);
                    gs_shrink_no_data_conf.width = new_gs_width;
                }
            }

            // Record checkpoint
            Checkpoint rs_no_data_cp, gs_no_data_cp;
            rs_no_data_cp.phase = gs_no_data_cp.phase = "shrinking_no_data";
            rs_no_data_cp.items_processed = gs_no_data_cp.items_processed = items_processed;
            rs_no_data_cp.items_in_phase = gs_no_data_cp.items_in_phase = 0;
            rs_no_data_cp.throughput_mops = gs_no_data_cp.throughput_mops = 0;
            rs_no_data_cp.geometric_cannot_shrink = gs_no_data_cp.geometric_cannot_shrink = gs_cannot_shrink;

            rs_no_data_cp.memory_kb = rs_shrink_no_data.get_max_memory_usage() / 1024;
            gs_no_data_cp.memory_kb = gs_shrink_no_data.get_max_memory_usage() / 1024;

            rs_no_data_cp.are = calculate_are_all_items(rs_shrink_no_data, expansion_true_freqs);
            gs_no_data_cp.are = calculate_are_all_items(gs_shrink_no_data, expansion_true_freqs);

            rs_no_data_cp.aae = calculate_aae_all_items(rs_shrink_no_data, expansion_true_freqs);
            gs_no_data_cp.aae = calculate_aae_all_items(gs_shrink_no_data, expansion_true_freqs);

            rs_no_data_cp.are_variance = calculate_are_variance(rs_shrink_no_data, expansion_true_freqs, rs_no_data_cp.are);
            gs_no_data_cp.are_variance = calculate_are_variance(gs_shrink_no_data, expansion_true_freqs, gs_no_data_cp.are);

            rs_no_data_cp.aae_variance = calculate_aae_variance(rs_shrink_no_data, expansion_true_freqs, rs_no_data_cp.aae);
            gs_no_data_cp.aae_variance = calculate_aae_variance(gs_shrink_no_data, expansion_true_freqs, gs_no_data_cp.aae);

            // Query throughput
            timer.start();
            for (const auto &[item, freq] : expansion_true_freqs) { volatile double q = rs_shrink_no_data.estimate(item); }
            rs_no_data_cp.query_throughput_mops = timer.stop_s() > 0 ? (expansion_true_freqs.size() / timer.stop_s() / 1e6) : 0;

            timer.start();
            for (const auto &[item, freq] : expansion_true_freqs) { volatile double q = gs_shrink_no_data.estimate(item); }
            gs_no_data_cp.query_throughput_mops = timer.stop_s() > 0 ? (expansion_true_freqs.size() / timer.stop_s() / 1e6) : 0;

            all_results["ReSketch_ShrinkNoData"][rep].push_back(rs_no_data_cp);
            all_results["GeometricSketch_ShrinkNoData"][rep].push_back(gs_no_data_cp);

            // Print checkpoint info
            cout << "Shrinking NoData checkpoint " << i << " -> " << (rs_memory_checkpoints[i] / 1024) << " KB:" << endl;
            cout << "  RS: Query: " << rs_no_data_cp.query_throughput_mops << " Mops, " << rs_no_data_cp.memory_kb << " KB, ARE=" << rs_no_data_cp.are
                 << ", AAE=" << rs_no_data_cp.aae << endl;
            cout << "  GS: Query: " << gs_no_data_cp.query_throughput_mops << " Mops, " << gs_no_data_cp.memory_kb << " KB, ARE=" << gs_no_data_cp.are
                 << ", AAE=" << gs_no_data_cp.aae << (gs_cannot_shrink ? " [Cannot shrink further]" : "") << endl;
        }

        cout << "Shrinking without data complete. Checkpoints: " << max_checkpoints << endl;
        cout << "Final memories: RS_NoData=" << rs_shrink_no_data.get_max_memory_usage() / 1024 << " KB, GS_NoData=" << gs_shrink_no_data.get_max_memory_usage() / 1024 << " KB"
             << endl;

        // PHASE 3: SHRINKING WITH DATA
        cout << "\n--- Phase 3: Shrinking With Data (RS: M1=" << actual_m1_rs / 1024 << " KB -> M2=" << m2_bytes / 1024 << " KB, GS: M1=" << actual_m1_gs / 1024
             << " KB -> M0=" << config.m0_kb << " KB) ---" << endl;

        // Use ReSketch intervals as standard for all sketches
        // Number of intervals = number of shrinking checkpoints
        size_t num_shrinking_checkpoints = rs_memory_checkpoints.size();
        vector<uint64_t> standard_item_intervals = calculate_geometric_item_intervals(config.shrinking_items, num_shrinking_checkpoints);

        cout << "Standard item intervals (based on ReSketch shrinking checkpoints): ";
        for (auto items : standard_item_intervals) cout << items << " ";
        cout << endl;

        cout << "ReSketch will process through all " << num_shrinking_checkpoints << " shrinking intervals" << endl;
        cout << "GeometricSketch will process through first " << min(num_shrinking_checkpoints, gs_memory_checkpoints.size()) << " shrinking intervals" << endl;

        uint64_t shrink_items_processed = 0;

        // Process intervals and shrink at checkpoints (using standard intervals)
        size_t num_intervals = standard_item_intervals.size();
        for (size_t interval_idx = 0; interval_idx < num_intervals; ++interval_idx)
        {
            uint64_t items_this_interval = standard_item_intervals[interval_idx];

            if (shrink_items_processed + items_this_interval > config.shrinking_items) { items_this_interval = config.shrinking_items - shrink_items_processed; }

            if (items_this_interval == 0) break;

            uint64_t chunk_start = items_processed + shrink_items_processed;
            uint64_t chunk_end = chunk_start + items_this_interval;

            // Process chunk for ReSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { rs_shrink_with_data.update(base_data[i % base_data.size()]); }
            double rs_duration = timer.stop_s();

            // Process chunk for GeometricSketch
            timer.start();
            for (uint64_t i = chunk_start; i < chunk_end; ++i) { gs_shrink_with_data.update(base_data[i % base_data.size()]); }
            double gs_duration = timer.stop_s();

            shrink_items_processed += items_this_interval;

            // Calculate true frequencies including shrinking phase
            map<uint64_t, uint64_t> combined_true_freqs = expansion_true_freqs;
            for (uint64_t i = items_processed; i < items_processed + shrink_items_processed; ++i) { combined_true_freqs[base_data[i % base_data.size()]]++; }

            bool gs_cannot_shrink = false;

            // Shrink ReSketch to checkpoint at interval_idx
            size_t checkpoint_idx = interval_idx;
            if (checkpoint_idx < rs_memory_checkpoints.size())
            {
                uint64_t target_memory = rs_memory_checkpoints[checkpoint_idx];
                uint32_t new_rs_width = calculate_width_from_memory_resketch(target_memory, rs_config.depth, rs_config.kll_k);
                if (new_rs_width < rs_shrink_with_data_conf.width)
                {
                    rs_shrink_with_data.shrink(new_rs_width);
                    rs_shrink_with_data_conf.width = new_rs_width;
                }
            }

            // Shrink GeometricSketch to checkpoint at interval_idx
            if (checkpoint_idx < gs_memory_checkpoints.size())
            {
                uint64_t target_memory = gs_memory_checkpoints[checkpoint_idx];
                uint32_t new_gs_width = calculate_width_from_memory_geometric(target_memory, gs_config.depth);

                if (new_gs_width >= gs_shrink_with_data_conf.width)
                {
                    gs_cannot_shrink = true;
                    cout << "GeometricSketch cannot shrink to width " << new_gs_width << " (current width: " << gs_shrink_with_data_conf.width << ")" << endl;
                }
                else
                {
                    gs_shrink_with_data.shrink(new_gs_width);
                    gs_shrink_with_data_conf.width = new_gs_width;
                }
            }

            // Record checkpoint
            Checkpoint rs_with_data_cp, gs_with_data_cp;
            rs_with_data_cp.phase = gs_with_data_cp.phase = "shrinking_with_data";
            rs_with_data_cp.items_processed = gs_with_data_cp.items_processed = items_processed + shrink_items_processed;
            rs_with_data_cp.items_in_phase = gs_with_data_cp.items_in_phase = shrink_items_processed;
            rs_with_data_cp.geometric_cannot_shrink = gs_with_data_cp.geometric_cannot_shrink = gs_cannot_shrink;

            rs_with_data_cp.throughput_mops = (rs_duration > 0) ? (items_this_interval / rs_duration / 1e6) : 0;
            gs_with_data_cp.throughput_mops = (gs_duration > 0) ? (items_this_interval / gs_duration / 1e6) : 0;

            rs_with_data_cp.memory_kb = rs_shrink_with_data.get_max_memory_usage() / 1024;
            gs_with_data_cp.memory_kb = gs_shrink_with_data.get_max_memory_usage() / 1024;

            rs_with_data_cp.are = calculate_are_all_items(rs_shrink_with_data, combined_true_freqs);
            gs_with_data_cp.are = calculate_are_all_items(gs_shrink_with_data, combined_true_freqs);

            rs_with_data_cp.aae = calculate_aae_all_items(rs_shrink_with_data, combined_true_freqs);
            gs_with_data_cp.aae = calculate_aae_all_items(gs_shrink_with_data, combined_true_freqs);

            rs_with_data_cp.are_variance = calculate_are_variance(rs_shrink_with_data, combined_true_freqs, rs_with_data_cp.are);
            gs_with_data_cp.are_variance = calculate_are_variance(gs_shrink_with_data, combined_true_freqs, gs_with_data_cp.are);

            rs_with_data_cp.aae_variance = calculate_aae_variance(rs_shrink_with_data, combined_true_freqs, rs_with_data_cp.aae);
            gs_with_data_cp.aae_variance = calculate_aae_variance(gs_shrink_with_data, combined_true_freqs, gs_with_data_cp.aae);

            // Query throughput
            timer.start();
            for (const auto &[item, freq] : combined_true_freqs) { volatile double q = rs_shrink_with_data.estimate(item); }
            rs_with_data_cp.query_throughput_mops = timer.stop_s() > 0 ? (combined_true_freqs.size() / timer.stop_s() / 1e6) : 0;

            timer.start();
            for (const auto &[item, freq] : combined_true_freqs) { volatile double q = gs_shrink_with_data.estimate(item); }
            gs_with_data_cp.query_throughput_mops = timer.stop_s() > 0 ? (combined_true_freqs.size() / timer.stop_s() / 1e6) : 0;

            all_results["ReSketch_ShrinkWithData"][rep].push_back(rs_with_data_cp);
            all_results["GeometricSketch_ShrinkWithData"][rep].push_back(gs_with_data_cp);

            // Print checkpoint info
            cout << "Shrinking WithData checkpoint at " << shrink_items_processed << " items (" << rs_with_data_cp.items_processed << " total):" << endl;
            cout << "  RS: " << rs_with_data_cp.throughput_mops << " Mops, Query: " << rs_with_data_cp.query_throughput_mops << " Mops, " << rs_with_data_cp.memory_kb
                 << " KB, ARE=" << rs_with_data_cp.are << ", AAE=" << rs_with_data_cp.aae << endl;
            cout << "  GS: " << gs_with_data_cp.throughput_mops << " Mops, Query: " << gs_with_data_cp.query_throughput_mops << " Mops, " << gs_with_data_cp.memory_kb
                 << " KB, ARE=" << gs_with_data_cp.are << ", AAE=" << gs_with_data_cp.aae << (gs_cannot_shrink ? " [Cannot shrink further]" : "") << endl;
        }

        cout << "Shrinking with data complete. Shrinking items processed: " << shrink_items_processed << endl;
        cout << "Final memories: RS_WithData=" << rs_shrink_with_data.get_max_memory_usage() / 1024 << " KB, GS_WithData=" << gs_shrink_with_data.get_max_memory_usage() / 1024
             << " KB" << endl;
    }

    // Add timestamp to output filename
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    string timestamp = timestamp_stream.str();

    string output_file = config.output_file;
    uint32_t ext_pos = output_file.find_last_of('.');
    if (ext_pos != string::npos) { output_file = output_file.substr(0, ext_pos) + "_" + timestamp + output_file.substr(ext_pos); }
    else
    {
        output_file += "_" + timestamp;
    }

    export_to_json(output_file, config, cm_config, rs_config, gs_config, ds_config, all_results);
}

int main(int argc, char **argv)
{
    ConfigParser parser;
    ExpansionShrinkingConfig exp_shrink_config;
    CountMinConfig cm_config;
    ReSketchConfig rs_config;
    GeometricSketchConfig gs_config;
    DynamicSketchConfig ds_config;

    ExpansionShrinkingConfig::add_params_to_config_parser(exp_shrink_config, parser);
    CountMinConfig::add_params_to_config_parser(cm_config, parser);
    ReSketchConfig::add_params_to_config_parser(rs_config, parser);
    GeometricSketchConfig::add_params_to_config_parser(gs_config, parser);
    DynamicSketchConfig::add_params_to_config_parser(ds_config, parser);

    if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h"))
    {
        parser.PrintUsage();
        return 0;
    }

    Status s = parser.ParseCommandLine(argc, argv);
    if (!s.IsOK())
    {
        fprintf(stderr, "%s\n", s.ToString().c_str());
        return -1;
    }

    run_expansion_shrinking_experiment(exp_shrink_config, cm_config, rs_config, gs_config, ds_config);

    return 0;
}
