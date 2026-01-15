#include "frequency_summary/frequency_summary_config.hpp"

#include "frequency_summary/count_min_sketch.hpp"
#include "frequency_summary/resketchv2.hpp"
#include "common.hpp"

#include "utils/ConfigParser.hpp"

#include <json/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace std;
using json = nlohmann::json;

// Sensitivity Experiment Config
struct SensitivityConfig
{
    uint32_t repetitions = 5;
    string dataset_type = "zipf";   // "zipf" or "caida"
    string caida_path = "data/CAIDA/only_ip";
    uint64_t total_items = 10000000;
    uint64_t stream_size = 10000000;
    uint64_t stream_diversity = 10000;
    float zipf_param = 1.1;
    string output_file = "output/sensitivity_results.json";

    // Memory budgets to test for all sketches
    vector<uint32_t> memory_budgets_kb = {32, 64, 256, 1024};
    // k values to test for ReSketch
    vector<uint32_t> k_values = {10, 30, 50, 70, 90};
    // depth values to test for ReSketch
    vector<uint32_t> depth_values = {1, 2, 3, 4, 5, 6, 7, 8};

    static void add_params_to_config_parser(SensitivityConfig &config, ConfigParser &parser)
    {
        parser.AddParameter(new UnsignedInt32Parameter("app.repetitions", to_string(config.repetitions), &config.repetitions, false, "Number of experiment repetitions"));
        parser.AddParameter(new StringParameter("app.dataset_type", config.dataset_type, &config.dataset_type, false, "Dataset type: zipf or caida"));
        parser.AddParameter(new StringParameter("app.caida_path", config.caida_path, &config.caida_path, false, "Path to CAIDA data file"));
        parser.AddParameter(new UnsignedInt64Parameter("app.total_items", to_string(config.total_items), &config.total_items, false, "Total items to process"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", to_string(config.stream_size), &config.stream_size, false, "Dataset size for zipf generation"));
        parser.AddParameter(
            new UnsignedInt64Parameter("app.stream_diversity", to_string(config.stream_diversity), &config.stream_diversity, false, "Unique items in stream (zipf)"));
        parser.AddParameter(new FloatParameter("app.zipf", to_string(config.zipf_param), &config.zipf_param, false, "Zipfian param 'a'"));
        parser.AddParameter(new StringParameter("app.output_file", config.output_file, &config.output_file, false, "Output JSON file path"));
    }

    friend std::ostream &operator<<(std::ostream &os, const SensitivityConfig &config)
    {
        os << "\n=== Sensitivity Experiment Configuration ===\n";
        os << "Memory budgets (KiB): ";
        for (uint32_t i = 0; i < config.memory_budgets_kb.size(); ++i)
        {
            os << config.memory_budgets_kb[i];
            if (i < config.memory_budgets_kb.size() - 1) os << ", ";
        }
        os << "\n";
        os << format("Repetitions: {}\n", config.repetitions);
        os << format("Dataset: {}\n", config.dataset_type);
        if (config.dataset_type == "caida") { os << format("CAIDA Path: {}\n", config.caida_path); }
        os << format("Total Items: {}\n", config.total_items);
        os << format("Dataset Size: {}\n", config.stream_size);
        if (config.dataset_type == "zipf")
        {
            os << format("Stream Diversity: {}\n", config.stream_diversity);
            os << format("Zipf Parameter: {}\n", config.zipf_param);
        }
        os << "ReSketch K values: ";
        for (uint32_t i = 0; i < config.k_values.size(); ++i)
        {
            os << config.k_values[i];
            if (i < config.k_values.size() - 1) os << ", ";
        }
        os << "\n";
        os << "ReSketch Depth values: ";
        for (uint32_t i = 0; i < config.depth_values.size(); ++i)
        {
            os << config.depth_values[i];
            if (i < config.depth_values.size() - 1) os << ", ";
        }
        os << "\n";
        os << format("Output File: {}\n", config.output_file);
        return os;
    }
};

// Result for a single configuration
struct SensitivityResult
{
    string algorithm;
    uint32_t k_value;   // Only relevant for ReSketch
    uint32_t width;
    uint32_t depth;
    uint64_t memory_budget_bytes;
    uint64_t memory_used_bytes;
    double throughput_mops;
    double query_throughput_mops;
    double are;
    double aae;
    double are_within_var;
    double aae_within_var;
};

void export_to_json(
    const string &filename, const SensitivityConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config,
    const map<string, vector<vector<SensitivityResult>>> &all_results)
{
    create_directory(filename);

    // Build JSON object
    json j;

    // Metadata section
    auto now = chrono::system_clock::now();
    string timestamp = format("{:%FT%TZ}", chrono::round<chrono::seconds>(now));

    j["metadata"] = {{"experiment_type", "sensitivity"}, {"timestamp", timestamp}};

    // Config section
    j["config"]["experiment"] = {{"repetitions", config.repetitions}, {"dataset_type", config.dataset_type},         {"total_items", config.total_items},
                                 {"stream_size", config.stream_size}, {"stream_diversity", config.stream_diversity}, {"zipf_param", config.zipf_param}};

    j["config"]["base_sketch_config"]["countmin"] = {{"depth", cm_config.depth}};
    j["config"]["base_sketch_config"]["resketch"] = {{"depth", rs_config.depth}};

    j["config"]["sensitivity_params"] = {{"memory_budgets_kb", config.memory_budgets_kb}, {"k_values", config.k_values}, {"depth_values", config.depth_values}};

    // Results section
    json results_json;
    for (const auto &[config_name, repetitions] : all_results)
    {
        json config_reps = json::array();

        for (uint32_t rep = 0; rep < repetitions.size(); ++rep)
        {
            json rep_json;
            rep_json["repetition_id"] = rep;

            json results_array = json::array();
            for (const auto &result : repetitions[rep])
            {
                json result_json = {
                    {"algorithm", result.algorithm},
                    {"k_value", result.k_value},
                    {"width", result.width},
                    {"depth", result.depth},
                    {"memory_budget_bytes", result.memory_budget_bytes},
                    {"memory_used_bytes", result.memory_used_bytes},
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
    if (!out.is_open())
    {
        cerr << format("Error: Cannot open output file: {}\n", filename);
        return;
    }

    out << j.dump(2);
    out.close();

    cout << format("\nResults exported to: {}\n", filename);
}

void run_sensitivity_experiment(const SensitivityConfig &config, const CountMinConfig &cm_config, const ReSketchConfig &rs_config)
{
    cout << config << endl;
    cout << cm_config << endl;
    cout << rs_config << endl;

    // Results storage: config_name -> repetitions -> results
    map<string, vector<vector<SensitivityResult>>> all_results;

    // Configuration names
    all_results["CountMin"] = vector<vector<SensitivityResult>>(config.repetitions);
    for (auto mem : config.memory_budgets_kb)
    {
        for (auto depth : config.depth_values)
        {
            for (auto k : config.k_values)
            {
                string config_name = format("ReSketch_M{}_d{}_k{}", mem, depth, k);
                all_results[config_name] = vector<vector<SensitivityResult>>(config.repetitions);
            }
        }
    }

    for (uint32_t rep = 0; rep < config.repetitions; ++rep)
    {
        cout << format("\n=== Repetition {}/{} ===\n", rep + 1, config.repetitions);

        // Generate or load dataset
        vector<uint64_t> data;
        if (config.dataset_type == "zipf")
        {
            cout << "Generating Zipf data..." << endl;
            data = generate_zipf_data(config.stream_size, config.stream_diversity, config.zipf_param);
        }
        else if (config.dataset_type == "caida")
        {
            cout << "Reading CAIDA data..." << endl;
            data = read_caida_data(config.caida_path, config.stream_size);
            if (data.empty())
            {
                cerr << "Error: Failed to read CAIDA data. Skipping repetition." << endl;
                continue;
            }
        }
        else
        {
            cerr << format("Error: Unknown dataset type: {}. Skipping repetition.\n", config.dataset_type);
            continue;
        }

        // Limit to total_items
        if (data.size() > config.total_items) { data.resize(config.total_items); }
        else if (data.size() < config.total_items)
        {
            // Repeat data to reach total_items
            vector<uint64_t> extended_data;
            extended_data.reserve(config.total_items);
            for (uint64_t i = 0; i < config.total_items; ++i) { extended_data.push_back(data[i % data.size()]); }
            data = std::move(extended_data);
        }

        cout << format("Processing {} items\n", data.size());

        // Calculate true frequencies
        map<uint64_t, uint64_t> true_freqs;
        for (const auto &item : data) { true_freqs[item]++; }

        // Prepare query set (all unique items)
        vector<uint64_t> query_items;
        query_items.reserve(true_freqs.size());
        for (const auto &[item, freq] : true_freqs) { query_items.push_back(item); }

        for (auto memory_budget_kb : config.memory_budgets_kb)
        {
            auto memory_budget_bytes = memory_budget_kb * 1024UL;
            // Test Count-Min Sketch
            {
                uint32_t cm_width = calculate_width_from_memory_cm(memory_budget_bytes, cm_config.depth);
                cout << format("\nCount-Min: depth={}, width={}\n", cm_config.depth, cm_width);

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
                double are_var_within = calculate_are_variance(cm_sketch, true_freqs, are);
                double aae_var_within = calculate_aae_variance(cm_sketch, true_freqs, aae);

                SensitivityResult result;
                result.algorithm = "CountMin";
                result.k_value = 0;   // Not applicable
                result.width = cm_width;
                result.depth = cm_config.depth;
                result.memory_budget_bytes = memory_budget_bytes;
                result.memory_used_bytes = cm_sketch.get_max_memory_usage();
                result.throughput_mops = throughput;
                result.query_throughput_mops = query_throughput;
                result.are = are;
                result.aae = aae;
                result.are_within_var = are_var_within;
                result.aae_within_var = aae_var_within;

                all_results["CountMin"][rep].push_back(result);

                cout << format("  Throughput: {} Mops/s\n", throughput);
                cout << format("  Query Throughput: {} Mops/s\n", query_throughput);
                cout << format("  Memory used: {} KiB\n", result.memory_used_bytes / 1024);
                cout << format("  ARE: {}, AAE: {}\n", are, aae);
            }

            // Test ReSketch with different depth and k values
            for (auto depth : config.depth_values)
            {
                for (auto k : config.k_values)
                {
                    uint32_t rs_width = calculate_width_from_memory_resketch(memory_budget_bytes, depth, k);
                    cout << format("ReSketch: M={}KiB, depth={}, k={}, width={}\n", memory_budget_kb, depth, k, rs_width);

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
                    double are_var_within_rs = calculate_are_variance(rs_sketch, true_freqs, are);
                    double aae_var_within_rs = calculate_aae_variance(rs_sketch, true_freqs, aae);

                    SensitivityResult result;
                    result.algorithm = "ReSketch";
                    result.k_value = k;
                    result.width = rs_width;
                    result.depth = depth;
                    result.memory_budget_bytes = memory_budget_bytes;
                    result.memory_used_bytes = rs_sketch.get_max_memory_usage();
                    result.throughput_mops = throughput;
                    result.query_throughput_mops = query_throughput;
                    result.are = are;
                    result.aae = aae;
                    result.are_within_var = are_var_within_rs;
                    result.aae_within_var = aae_var_within_rs;

                    string config_name = format("ReSketch_M{}_d{}_k{}", memory_budget_kb, depth, k);
                    all_results[config_name][rep].push_back(result);

                    cout << format("  Throughput: {} Mops/s\n", throughput);
                    cout << format("  Query Throughput: {} Mops/s\n", query_throughput);
                    cout << format("  Memory used: {} KiB\n", result.memory_used_bytes / 1024);
                    cout << format("  ARE: {}, AAE: {}\n", are, aae);
                }
            }
        }
    }

    // Add timestamp to output filename

    auto now = chrono::system_clock::now();
    string timestamp = format("{:%Y%m%d_%H%M%S}", chrono::round<chrono::seconds>(now));

    // Insert timestamp before file extension
    string output_file = config.output_file;
    uint32_t ext_pos = output_file.find_last_of('.');
    if (ext_pos != string::npos) { output_file = output_file.substr(0, ext_pos) + "_" + timestamp + output_file.substr(ext_pos); }
    else
    {
        output_file += "_" + timestamp;
    }

    export_to_json(output_file, config, cm_config, rs_config, all_results);
}

int main(int argc, char **argv)
{
    ConfigParser parser;
    SensitivityConfig sens_config;
    CountMinConfig cm_config;
    ReSketchConfig rs_config;

    SensitivityConfig::add_params_to_config_parser(sens_config, parser);
    CountMinConfig::add_params_to_config_parser(cm_config, parser);
    ReSketchConfig::add_params_to_config_parser(rs_config, parser);

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

    run_sensitivity_experiment(sens_config, cm_config, rs_config);

    return 0;
}
