#define DOCTEST_CONFIG_IMPLEMENT
#include "frequency_summary/frequency_summary_config.hpp"

#include "frequency_summary/resketchv2.hpp"
#include "common.hpp"

#include "utils/ConfigParser.hpp"

#include <json/json.hpp>

#include "doctest.h"
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

// Split Experiment Config
struct SplitConfig
{
    uint32_t memory_budget_kb = 32;
    uint32_t repetitions = 10;
    string dataset_type = "caida";
    string caida_path = "data/CAIDA/only_ip";
    uint64_t stream_size = 10'000'000;
    uint64_t stream_diversity = 1'000'000;
    float zipf_param = 1.1;
    string output_file = "output/split_results.json";

    static void add_params_to_config_parser(SplitConfig &config, ConfigParser &parser)
    {
        parser.AddParameter(
            new UnsignedInt32Parameter("app.memory_budget_kb", to_string(config.memory_budget_kb), &config.memory_budget_kb, false, "Memory budget in KB per sketch"));
        parser.AddParameter(new UnsignedInt32Parameter("app.repetitions", to_string(config.repetitions), &config.repetitions, false, "Number of experiment repetitions"));
        parser.AddParameter(new StringParameter("app.dataset_type", config.dataset_type, &config.dataset_type, false, "Dataset type: zipf or caida"));
        parser.AddParameter(new StringParameter("app.caida_path", config.caida_path, &config.caida_path, false, "Path to CAIDA data file"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", to_string(config.stream_size), &config.stream_size, false, "Stream size"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", to_string(config.stream_diversity), &config.stream_diversity, false, "Unique items in stream"));
        parser.AddParameter(new FloatParameter("app.zipf", to_string(config.zipf_param), &config.zipf_param, false, "Zipfian param 'a'"));
        parser.AddParameter(new StringParameter("app.output_file", config.output_file, &config.output_file, false, "Output JSON file path"));
    }

    friend ostream &operator<<(ostream &os, const SplitConfig &config)
    {
        os << "\n=== Split Experiment Configuration ===" << endl;
        os << format("Memory Budget (per sketch): {} KiB\n", config.memory_budget_kb);
        os << format("Repetitions: {}\n", config.repetitions);
        os << format("Dataset: {}\n", config.dataset_type);
        if (config.dataset_type == "caida") { os << format("CAIDA Path: {}\n", config.caida_path); }
        os << format("Total Stream Size: {}\n", config.stream_size);
        os << format("Stream Diversity: {}\n", config.stream_diversity);
        if (config.dataset_type == "zipf") { os << format("Zipf Parameter: {}\n", config.zipf_param); }
        os << format("Output File: {}\n", config.output_file);
        return os;
    }
};

// Result structure
struct AccuracyMetrics
{
    double are = 0.0;
    double aae = 0.0;
    double are_variance = 0.0;
    double aae_variance = 0.0;
};

struct SketchMetrics
{
    double process_time_s = 0.0;
    uint32_t memory_bytes = 0;
};

struct SplitResult
{
    // Sketch metrics
    SketchMetrics sketch_c_full;
    SketchMetrics sketch_a_direct;
    SketchMetrics sketch_b_direct;
    double split_time_s = 0.0;

    // Accuracy comparisons
    AccuracyMetrics a_prime_vs_true_on_da;   // A' (from split) vs true frequencies on DA
    AccuracyMetrics b_prime_vs_true_on_db;   // B' (from split) vs true frequencies on DB
    AccuracyMetrics a_vs_true_on_da;         // A (direct) vs true frequencies on DA
    AccuracyMetrics b_vs_true_on_db;         // B (direct) vs true frequencies on DB
    AccuracyMetrics c_vs_true_on_all;        // C (full) vs true frequencies on All
};

void export_to_json(const string &filename, const SplitConfig &app_config, const ReSketchConfig &rs_config, const vector<SplitResult> &results)
{
    create_directory(filename);

    // Build JSON object
    json j;

    // Metadata section
    auto now = chrono::system_clock::now();
    string timestamp = format("{:%FT%TZ}", chrono::round<chrono::seconds>(now));

    j["metadata"] = {{"experiment_type", "split"}, {"timestamp", timestamp}};

    // Config section
    j["config"]["experiment"] = {{"memory_budget_kb", app_config.memory_budget_kb}, {"repetitions", results.size()},
                                 {"dataset_type", app_config.dataset_type},         {"stream_size", app_config.stream_size},
                                 {"stream_diversity", app_config.stream_diversity}, {"zipf_param", app_config.zipf_param}};

    j["config"]["base_sketch_config"]["resketch"] = {{"depth", rs_config.depth}, {"kll_k", rs_config.kll_k}, {"width", rs_config.width}};

    // Results section
    j["results"] = json::array();
    for (uint32_t rep = 0; rep < results.size(); ++rep)
    {
        const auto &result = results[rep];
        json rep_json = {
            {"repetition_id", rep},
            {"sketch_c_full", {{"memory_bytes", result.sketch_c_full.memory_bytes}, {"process_time_s", result.sketch_c_full.process_time_s}}},
            {"sketch_a_direct", {{"memory_bytes", result.sketch_a_direct.memory_bytes}, {"process_time_s", result.sketch_a_direct.process_time_s}}},
            {"sketch_b_direct", {{"memory_bytes", result.sketch_b_direct.memory_bytes}, {"process_time_s", result.sketch_b_direct.process_time_s}}},
            {"split_time_s", result.split_time_s},
            {"a_prime_vs_true_on_da",
             {{"are", result.a_prime_vs_true_on_da.are},
              {"aae", result.a_prime_vs_true_on_da.aae},
              {"are_variance", result.a_prime_vs_true_on_da.are_variance},
              {"aae_variance", result.a_prime_vs_true_on_da.aae_variance}}},
            {"b_prime_vs_true_on_db",
             {{"are", result.b_prime_vs_true_on_db.are},
              {"aae", result.b_prime_vs_true_on_db.aae},
              {"are_variance", result.b_prime_vs_true_on_db.are_variance},
              {"aae_variance", result.b_prime_vs_true_on_db.aae_variance}}},
            {"a_vs_true_on_da",
             {{"are", result.a_vs_true_on_da.are},
              {"aae", result.a_vs_true_on_da.aae},
              {"are_variance", result.a_vs_true_on_da.are_variance},
              {"aae_variance", result.a_vs_true_on_da.aae_variance}}},
            {"b_vs_true_on_db",
             {{"are", result.b_vs_true_on_db.are},
              {"aae", result.b_vs_true_on_db.aae},
              {"are_variance", result.b_vs_true_on_db.are_variance},
              {"aae_variance", result.b_vs_true_on_db.aae_variance}}},
            {"c_vs_true_on_all",
             {{"are", result.c_vs_true_on_all.are},
              {"aae", result.c_vs_true_on_all.aae},
              {"are_variance", result.c_vs_true_on_all.are_variance},
              {"aae_variance", result.c_vs_true_on_all.aae_variance}}}};
        j["results"].push_back(rep_json);
    }

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

void run_split_experiment(const SplitConfig &config, const ReSketchConfig &rs_config)
{
    cout << config << endl;
    cout << rs_config << endl;

    vector<SplitResult> all_results;
    all_results.reserve(config.repetitions);

    uint32_t memory_bytes = config.memory_budget_kb * 1024;
    uint32_t width = ReSketchV2::calculate_max_width(memory_bytes, rs_config.depth, rs_config.kll_k);

    cout << "\n=== Calculated Width ===\n";
    cout << format("Width per sketch: {}\n", width);

    for (uint32_t rep = 0; rep < config.repetitions; ++rep)
    {
        cout << "\n========================================" << endl;
        cout << format("\n=== Repetition {}/{} ===\n", rep + 1, config.repetitions);
        cout << "========================================" << endl;

        SplitResult result;

        // Generate shared seeds for all sketches to ensure consistent hashing
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint32_t> dist;

        uint32_t shared_partition_seed = dist(rng);
        std::vector<uint32_t> shared_seeds;
        shared_seeds.reserve(rs_config.depth);
        for (uint32_t i = 0; i < rs_config.depth; ++i) { shared_seeds.push_back(dist(rng)); }

        // Calculate split point for hash-based partitioning
        uint64_t split_point = static_cast<uint64_t>((static_cast<long double>(width / 2) / width) * std::numeric_limits<uint64_t>::max());

        // Generate full dataset first
        vector<uint64_t> full_data;
        if (config.dataset_type == "zipf")
        {
            cout << "Generating Zipf data..." << endl;
            full_data = generate_zipf_data(config.stream_size, config.stream_diversity, config.zipf_param);
        }
        else if (config.dataset_type == "caida")
        {
            cout << "Reading CAIDA data..." << endl;
            full_data = read_caida_data(config.caida_path, config.stream_size);
            if (full_data.empty())
            {
                cerr << "Error: Failed to read CAIDA data. Skipping repetition." << endl;
                continue;
            }
        }
        else
        {
            cerr << format("Error: Unknown dataset type: {}. Skipping repetition.", config.dataset_type);
            continue;
        }

        // Split data based on hash domain (same as split operation)
        vector<uint64_t> data_A, data_B;
        data_A.reserve(full_data.size() / 2);
        data_B.reserve(full_data.size() / 2);

        for (const auto &item : full_data)
        {
            uint64_t partition_hash = ReSketchV2::compute_partition_hash(item, shared_partition_seed);
            if (partition_hash < split_point) { data_A.push_back(item); }
            else
            {
                data_B.push_back(item);
            }
        }

        cout << format("  Full dataset: {} items\n", full_data.size());
        cout << format("  DA (hash < split_point): {} items\n", data_A.size());
        cout << format("  DB (hash >= split_point): {} items\n", data_B.size());

        // Calculate true frequencies for each dataset
        map<uint64_t, uint64_t> true_freqs_A, true_freqs_B, true_freqs_all;
        for (const auto &item : data_A)
        {
            true_freqs_A[item]++;
            true_freqs_all[item]++;
        }
        for (const auto &item : data_B)
        {
            true_freqs_B[item]++;
            true_freqs_all[item]++;
        }

        cout << format("  Unique items: {} (A), {} (B), {}, (All)\n", true_freqs_A.size(), true_freqs_B.size(), true_freqs_all.size());

        // Process Sketch C (full width, processes both A and B)
        cout << "\nProcessing Sketch C (full, A+B)..." << endl;
        ReSketchV2 sketch_C(rs_config.depth, width, shared_seeds, rs_config.kll_k, shared_partition_seed);
        Timer timer;
        timer.start();
        for (const auto &item : data_A) { sketch_C.update(item); }
        for (const auto &item : data_B) { sketch_C.update(item); }
        result.sketch_c_full.process_time_s = timer.stop_s();
        result.sketch_c_full.memory_bytes = sketch_C.get_max_memory_usage();
        cout << format("  Time: {} s, Memory: {} KiB\n", result.sketch_c_full.process_time_s, result.sketch_c_full.memory_bytes / 1024);

        // Split C into A' and B'
        cout << "\nSplitting Sketch C into A' and B'..." << endl;
        timer.start();
        auto [sketch_A_prime, sketch_B_prime] = ReSketchV2::split(sketch_C, width / 2, width / 2);
        result.split_time_s = timer.stop_s();
        cout << format("  Split time: {} s", result.split_time_s);

        // Print partition ranges to verify split
        cout << "  A' partition ranges: ";
        for (const auto &[start, end] : sketch_A_prime.get_partition_ranges()) { cout << format("[{}, {}) ", start, end); }
        cout << endl;
        cout << "  B' partition ranges: ";
        for (const auto &[start, end] : sketch_B_prime.get_partition_ranges()) { cout << format("[{}, {}) ", start, end); }
        cout << endl;

        // Process Sketch A (direct, half width)
        cout << "\nProcessing Sketch A (direct, only A)..." << endl;
        ReSketchV2 sketch_A(rs_config.depth, width / 2, shared_seeds, rs_config.kll_k, shared_partition_seed);
        timer.start();
        for (const auto &item : data_A) { sketch_A.update(item); }
        result.sketch_a_direct.process_time_s = timer.stop_s();
        result.sketch_a_direct.memory_bytes = sketch_A.get_max_memory_usage();
        cout << format("  Time: {} s, Memory: {} KiB", result.sketch_a_direct.process_time_s, result.sketch_a_direct.memory_bytes / 1024);

        // Process Sketch B (direct, half width)
        cout << "\nProcessing Sketch B (direct, only B)..." << endl;
        ReSketchV2 sketch_B(rs_config.depth, width / 2, shared_seeds, rs_config.kll_k, shared_partition_seed);
        timer.start();
        for (const auto &item : data_B) { sketch_B.update(item); }
        result.sketch_b_direct.process_time_s = timer.stop_s();
        result.sketch_b_direct.memory_bytes = sketch_B.get_max_memory_usage();
        cout << format("  Time: {} s, Memory: {} KiB", result.sketch_b_direct.process_time_s, result.sketch_b_direct.memory_bytes / 1024);

        // Calculate accuracy comparisons
        cout << "\nCalculating accuracy metrics..." << endl;

        // Note: For A' and B', test on ALL items but route to correct partition
        double total_rel_error_a_prime = 0.0, total_abs_error_a_prime = 0.0;
        double total_rel_error_b_prime = 0.0, total_abs_error_b_prime = 0.0;
        int count_a_prime = 0, count_b_prime = 0;
        vector<double> a_prime_rel_errors, a_prime_abs_errors;
        vector<double> b_prime_rel_errors, b_prime_abs_errors;

        for (const auto &[item, true_freq] : true_freqs_all)
        {
            double est_freq;
            if (sketch_A_prime.is_responsible_for(item))
            {
                // Item belongs to A' partition
                est_freq = sketch_A_prime.estimate(item);
                double rel_error = (true_freq > 0) ? (std::abs(est_freq - true_freq) / true_freq) : 0.0;
                double abs_error = std::abs(est_freq - true_freq);
                total_rel_error_a_prime += rel_error;
                total_abs_error_a_prime += abs_error;
                a_prime_rel_errors.push_back(rel_error);
                a_prime_abs_errors.push_back(abs_error);
                count_a_prime++;
            }
            else
            {
                // Item belongs to B' partition
                est_freq = sketch_B_prime.estimate(item);
                double rel_error = (true_freq > 0) ? (std::abs(est_freq - true_freq) / true_freq) : 0.0;
                double abs_error = std::abs(est_freq - true_freq);
                total_rel_error_b_prime += rel_error;
                total_abs_error_b_prime += abs_error;
                b_prime_rel_errors.push_back(rel_error);
                b_prime_abs_errors.push_back(abs_error);
                count_b_prime++;
            }
        }

        result.a_prime_vs_true_on_da.are = count_a_prime > 0 ? total_rel_error_a_prime / count_a_prime : 0.0;
        result.a_prime_vs_true_on_da.aae = count_a_prime > 0 ? total_abs_error_a_prime / count_a_prime : 0.0;

        // Calculate variance for A'
        if (!a_prime_rel_errors.empty())
        {
            double sum_sq = 0.0;
            for (double v : a_prime_rel_errors) sum_sq += (v - result.a_prime_vs_true_on_da.are) * (v - result.a_prime_vs_true_on_da.are);
            result.a_prime_vs_true_on_da.are_variance = sum_sq / a_prime_rel_errors.size();
        }
        if (!a_prime_abs_errors.empty())
        {
            double sum_sq = 0.0;
            for (double v : a_prime_abs_errors) sum_sq += (v - result.a_prime_vs_true_on_da.aae) * (v - result.a_prime_vs_true_on_da.aae);
            result.a_prime_vs_true_on_da.aae_variance = sum_sq / a_prime_abs_errors.size();
        }

        cout << format("  A' (split) on its partition ({} items): ARE={}, AAE={}\n", count_a_prime, result.a_prime_vs_true_on_da.are, result.a_prime_vs_true_on_da.aae);

        result.b_prime_vs_true_on_db.are = count_b_prime > 0 ? total_rel_error_b_prime / count_b_prime : 0.0;
        result.b_prime_vs_true_on_db.aae = count_b_prime > 0 ? total_abs_error_b_prime / count_b_prime : 0.0;

        // Calculate variance for B'
        if (!b_prime_rel_errors.empty())
        {
            double sum_sq = 0.0;
            for (double v : b_prime_rel_errors) sum_sq += (v - result.b_prime_vs_true_on_db.are) * (v - result.b_prime_vs_true_on_db.are);
            result.b_prime_vs_true_on_db.are_variance = sum_sq / b_prime_rel_errors.size();
        }
        if (!b_prime_abs_errors.empty())
        {
            double sum_sq = 0.0;
            for (double v : b_prime_abs_errors) sum_sq += (v - result.b_prime_vs_true_on_db.aae) * (v - result.b_prime_vs_true_on_db.aae);
            result.b_prime_vs_true_on_db.aae_variance = sum_sq / b_prime_abs_errors.size();
        }

        cout << format("  B' (split) on its partition ({} items): ARE={}, AAE={}\n", count_b_prime, result.b_prime_vs_true_on_db.are, result.b_prime_vs_true_on_db.aae);

        // A (direct) vs true on DA items: baseline
        result.a_vs_true_on_da.are = calculate_are_all_items(sketch_A, true_freqs_A);
        result.a_vs_true_on_da.aae = calculate_aae_all_items(sketch_A, true_freqs_A);
        result.a_vs_true_on_da.are_variance = calculate_are_variance(sketch_A, true_freqs_A, result.a_vs_true_on_da.are);
        result.a_vs_true_on_da.aae_variance = calculate_aae_variance(sketch_A, true_freqs_A, result.a_vs_true_on_da.aae);
        cout << format("  A (direct) vs True on DA: ARE={}, AAE={}\n", result.a_vs_true_on_da.are, result.a_vs_true_on_da.aae);

        // B (direct) vs true on DB items: baseline
        result.b_vs_true_on_db.are = calculate_are_all_items(sketch_B, true_freqs_B);
        result.b_vs_true_on_db.aae = calculate_aae_all_items(sketch_B, true_freqs_B);
        result.b_vs_true_on_db.are_variance = calculate_are_variance(sketch_B, true_freqs_B, result.b_vs_true_on_db.are);
        result.b_vs_true_on_db.aae_variance = calculate_aae_variance(sketch_B, true_freqs_B, result.b_vs_true_on_db.aae);
        cout << format("  B (direct) vs True on DB: ARE={}, AAE={}\n", result.b_vs_true_on_db.are, result.b_vs_true_on_db.aae);

        // C (full) vs true on all items
        result.c_vs_true_on_all.are = calculate_are_all_items(sketch_C, true_freqs_all);
        result.c_vs_true_on_all.aae = calculate_aae_all_items(sketch_C, true_freqs_all);
        result.c_vs_true_on_all.are_variance = calculate_are_variance(sketch_C, true_freqs_all, result.c_vs_true_on_all.are);
        result.c_vs_true_on_all.aae_variance = calculate_aae_variance(sketch_C, true_freqs_all, result.c_vs_true_on_all.aae);
        cout << format("  C (full) vs True on All: ARE={}, AAE={}\n", result.c_vs_true_on_all.are, result.c_vs_true_on_all.aae);

        all_results.push_back(result);
    }

    // Add timestamp to output filename
    auto now = chrono::system_clock::now();
    string timestamp = format("{:%FT%TZ}", chrono::round<chrono::seconds>(now));

    // Insert timestamp before file extension
    string output_file = config.output_file;
    uint32_t ext_pos = output_file.find_last_of('.');
    if (ext_pos != string::npos) { output_file = output_file.substr(0, ext_pos) + "_" + timestamp + output_file.substr(ext_pos); }
    else
    {
        output_file += "_" + timestamp;
    }

    export_to_json(output_file, config, rs_config, all_results);
}

int main(int argc, char **argv)
{
    ConfigParser parser;
    SplitConfig app_config;
    ReSketchConfig rs_config;

    SplitConfig::add_params_to_config_parser(app_config, parser);
    ReSketchConfig::add_params_to_config_parser(rs_config, parser);

    if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h"))
    {
        parser.PrintUsage();
        return 0;
    }

    Status s = parser.ParseCommandLine(argc, argv);
    if (!s.IsOK())
    {
        cerr << s.ToString();
        return -1;
    }

    run_split_experiment(app_config, rs_config);

    return 0;
}
