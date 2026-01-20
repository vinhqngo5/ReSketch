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

// Merge Experiment Config
struct MergeConfig
{
    uint32_t memory_budget_kb = 64;
    uint32_t repetitions = 3;
    string dataset_type = "caida";
    string caida_path = "data/CAIDA/only_ip";
    uint64_t stream_size = 10'000'000;
    uint64_t stream_diversity = 1'000'000;
    float zipf_param = 1.1;
    string output_file = "output/merge_results.json";

    static void add_params_to_config_parser(MergeConfig &config, ConfigParser &parser)
    {
        parser.AddParameter(
            new UnsignedInt32Parameter("app.memory_budget_kb", to_string(config.memory_budget_kb), &config.memory_budget_kb, false, "Memory budget in KB per sketch"));
        parser.AddParameter(new UnsignedInt32Parameter("app.repetitions", to_string(config.repetitions), &config.repetitions, false, "Number of experiment repetitions"));
        parser.AddParameter(new StringParameter("app.dataset_type", config.dataset_type, &config.dataset_type, false, "Dataset type: zipf or caida"));
        parser.AddParameter(new StringParameter("app.caida_path", config.caida_path, &config.caida_path, false, "Path to CAIDA data file"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", to_string(config.stream_size), &config.stream_size, false, "Total stream size (will be split 50-50)"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", to_string(config.stream_diversity), &config.stream_diversity, false, "Unique items in stream"));
        parser.AddParameter(new FloatParameter("app.zipf", to_string(config.zipf_param), &config.zipf_param, false, "Zipfian param 'a'"));
        parser.AddParameter(new StringParameter("app.output_file", config.output_file, &config.output_file, false, "Output JSON file path"));
    }

    friend std::ostream &operator<<(std::ostream &os, const MergeConfig &config)
    {
        os << "\n=== Merge Experiment Configuration ===\n";
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

// Result for a single repetition
struct MergeResult
{
    // Sketch metadata
    struct SketchInfo
    {
        uint64_t memory_bytes;
        double process_time_s;
    };

    SketchInfo sketch_a;
    SketchInfo sketch_b;
    SketchInfo sketch_c_merged;
    SketchInfo sketch_d_ground_truth;
    double merge_time_s;

    // Accuracy comparisons
    struct AccuracyComparison
    {
        double are;
        double aae;
        double are_variance;
        double aae_variance;
    };

    AccuracyComparison a_vs_true_on_da;    // Sketch A's accuracy on DA items
    AccuracyComparison b_vs_true_on_db;    // Sketch B's accuracy on DB items
    AccuracyComparison c_vs_true_on_all;   // Merged sketch C's accuracy on all items
    AccuracyComparison d_vs_true_on_all;   // Ground truth sketch D's accuracy on all items

    struct ItemFrequency
    {
        uint64_t key;
        uint64_t frequency;
        double estimated_frequency;
    };

    vector<ItemFrequency> c_item_frequencies;
    vector<ItemFrequency> d_item_frequencies;
};

void export_to_json(const string &filename, const MergeConfig &config, const ReSketchConfig &rs_config, const vector<MergeResult> &results)
{
    create_directory(filename);

    json j;

    // Metadata section
    auto now = chrono::system_clock::now();
    string timestamp = format("{:%FT%TZ}", chrono::round<chrono::seconds>(now));

    j["metadata"] = {{"experiment_type", "merge"}, {"timestamp", timestamp}};

    // Config section
    j["config"]["experiment"] = {{"memory_budget_kb", config.memory_budget_kb}, {"repetitions", config.repetitions},           {"dataset_type", config.dataset_type},
                                 {"stream_size", config.stream_size},           {"stream_diversity", config.stream_diversity}, {"zipf_param", config.zipf_param}};

    j["config"]["base_sketch_config"]["resketch"] = {{"depth", rs_config.depth}, {"kll_k", rs_config.kll_k}};

    // Results section
    j["results"] = json::array();
    for (uint32_t rep = 0; rep < results.size(); ++rep)
    {
        const auto &r = results[rep];
        json rep_json = {
            {"repetition_id", rep},
            {"sketch_a", {{"memory_bytes", r.sketch_a.memory_bytes}, {"process_time_s", r.sketch_a.process_time_s}}},
            {"sketch_b", {{"memory_bytes", r.sketch_b.memory_bytes}, {"process_time_s", r.sketch_b.process_time_s}}},
            {"sketch_c_merged", {{"memory_bytes", r.sketch_c_merged.memory_bytes}, {"merge_time_s", r.merge_time_s}}},
            {"sketch_d_ground_truth", {{"memory_bytes", r.sketch_d_ground_truth.memory_bytes}, {"process_time_s", r.sketch_d_ground_truth.process_time_s}}},
            {"accuracy",
             {{"a_vs_true_on_da",
               {{"are", r.a_vs_true_on_da.are},
                {"aae", r.a_vs_true_on_da.aae},
                {"are_variance", r.a_vs_true_on_da.are_variance},
                {"aae_variance", r.a_vs_true_on_da.aae_variance}}},
              {"b_vs_true_on_db",
               {{"are", r.b_vs_true_on_db.are},
                {"aae", r.b_vs_true_on_db.aae},
                {"are_variance", r.b_vs_true_on_db.are_variance},
                {"aae_variance", r.b_vs_true_on_db.aae_variance}}},
              {"c_vs_true_on_all",
               {{"are", r.c_vs_true_on_all.are},
                {"aae", r.c_vs_true_on_all.aae},
                {"are_variance", r.c_vs_true_on_all.are_variance},
                {"aae_variance", r.c_vs_true_on_all.aae_variance}}},
              {"d_vs_true_on_all",
               {{"are", r.d_vs_true_on_all.are},
                {"aae", r.d_vs_true_on_all.aae},
                {"are_variance", r.d_vs_true_on_all.are_variance},
                {"aae_variance", r.d_vs_true_on_all.aae_variance}}}}}};

        rep_json["c_frequencies"] = json::array();
        for (const auto &[key, frequency, estimated_frequency] : r.c_item_frequencies)
        {
            rep_json["c_frequencies"].push_back({
                {"key", key},
                {"freq", frequency},
                {"est", estimated_frequency},
            });
        }
        rep_json["d_frequencies"] = json::array();
        for (const auto &[key, frequency, estimated_frequency] : r.d_item_frequencies)
        {
            rep_json["d_frequencies"].push_back({
                {"key", key},
                {"freq", frequency},
                {"est", estimated_frequency},
            });
        }

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

void run_merge_experiment(const MergeConfig &config, const ReSketchConfig &rs_config)
{
    cout << config << endl;
    cout << rs_config << endl;

    vector<MergeResult> all_results;
    all_results.reserve(config.repetitions);

    uint64_t memory_budget_bytes = static_cast<uint64_t>(config.memory_budget_kb) * 1024;
    uint32_t width = calculate_width_from_memory_resketch(memory_budget_bytes, rs_config.depth, rs_config.kll_k);

    cout << format("\nReSketch Configuration: depth={}, k={}, width={}\n", rs_config.depth, rs_config.kll_k, width);

    for (uint32_t rep = 0; rep < config.repetitions; ++rep)
    {
        cout << format("\n=== Repetition {}/{} ===\n", rep + 1, config.repetitions);

        MergeResult result;

        // Generate datasets DA and DB with disjoint item ranges
        vector<uint64_t> data_A, data_B;

        if (config.dataset_type == "zipf")
        {
            cout << "Generating disjoint Zipf datasets..." << endl;

            // DA: items from [0, stream_diversity/2)
            uint64_t half_diversity = config.stream_diversity / 2;
            uint64_t half_stream = config.stream_size / 2;

            data_A = generate_zipf_data(half_stream, half_diversity, config.zipf_param);

            // DB: items from [stream_diversity/2, stream_diversity)
            vector<uint64_t> data_B_raw = generate_zipf_data(half_stream, half_diversity, config.zipf_param);
            data_B.reserve(data_B_raw.size());
            for (const auto &item : data_B_raw) { data_B.push_back(item + half_diversity); }

            cout << format("  DA: {} items from range [0, {}]\n", data_A.size(), half_diversity - 1);
            cout << format("  DB: {} items from range [{}, {}]\n", data_B.size(), half_diversity, config.stream_diversity - 1);
        }
        else if (config.dataset_type == "caida")
        {
            cout << "Reading CAIDA data..." << endl;
            vector<uint64_t> full_data = read_caida_data(config.caida_path, config.stream_size);
            if (full_data.empty())
            {
                cerr << "Error: Failed to read CAIDA data. Skipping repetition." << endl;
                continue;
            }

            // Split CAIDA data into disjoint sets based on item hash (odd/even)
            // This ensures data_A and data_B have completely disjoint item sets
            data_A.reserve(full_data.size() / 2);
            data_B.reserve(full_data.size() / 2);

            for (const auto &item : full_data)
            {
                if (item % 2 == 0) { data_A.push_back(item); }
                else
                {
                    data_B.push_back(item);
                }
            }

            cout << format("  DA: {} items (even IPs)\n", data_A.size());
            cout << format("  DB: {} items (odd IPs)\n", data_B.size());
        }
        else
        {
            cerr << format("Error: Unknown dataset type: {}. Skipping repetition.", config.dataset_type);
            continue;
        }

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

        cout << format("  Unique items: {} (A), {} (B), {} (All)\n", true_freqs_A.size(), true_freqs_B.size(), true_freqs_all.size());

        // Generate shared seeds for all sketches to ensure consistent hashing
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint32_t> dist;

        uint32_t shared_partition_seed = dist(rng);
        std::vector<uint32_t> shared_seeds;
        shared_seeds.reserve(rs_config.depth);
        for (uint32_t i = 0; i < rs_config.depth; ++i) { shared_seeds.push_back(dist(rng)); }

        // Process Sketch A
        cout << "\nProcessing Sketch A..." << endl;
        ReSketchV2 sketch_A(rs_config.depth, width, shared_seeds, rs_config.kll_k, shared_partition_seed);
        Timer timer;
        timer.start();
        for (const auto &item : data_A) { sketch_A.update(item); }
        result.sketch_a.process_time_s = timer.stop_s();
        result.sketch_a.memory_bytes = sketch_A.get_max_memory_usage();
        cout << format("  Time: {} s, Memory: {} KiB\n", result.sketch_a.process_time_s, result.sketch_a.memory_bytes / 1024);

        // Process Sketch B
        cout << "\nProcessing Sketch B..." << endl;
        ReSketchV2 sketch_B(rs_config.depth, width, shared_seeds, rs_config.kll_k, shared_partition_seed);
        timer.start();
        for (const auto &item : data_B) { sketch_B.update(item); }
        result.sketch_b.process_time_s = timer.stop_s();
        result.sketch_b.memory_bytes = sketch_B.get_max_memory_usage();
        cout << format("  Time: {} s, Memory: {} KiB\n", result.sketch_b.process_time_s, result.sketch_b.memory_bytes / 1024);

        // Merge A and B into C
        cout << "\nMerging Sketch A and B into C..." << endl;
        timer.start();
        ReSketchV2 sketch_C = ReSketchV2::merge(sketch_A, sketch_B);
        result.merge_time_s = timer.stop_s();
        result.sketch_c_merged.memory_bytes = sketch_C.get_max_memory_usage();
        cout << format("  Merge time: {} s, Memory: {} KiB\n", result.merge_time_s, result.sketch_c_merged.memory_bytes / 1024);

        // Process Ground Truth Sketch D
        cout << "\nProcessing Ground Truth Sketch D..." << endl;
        ReSketchV2 sketch_D(rs_config.depth, width * 2, shared_seeds, rs_config.kll_k, shared_partition_seed);
        timer.start();
        for (const auto &item : data_A) { sketch_D.update(item); }
        for (const auto &item : data_B) { sketch_D.update(item); }
        result.sketch_d_ground_truth.process_time_s = timer.stop_s();
        result.sketch_d_ground_truth.memory_bytes = sketch_D.get_max_memory_usage();
        cout << format("  Time: {} s, Memory: {} KiB\n", result.sketch_d_ground_truth.process_time_s, result.sketch_d_ground_truth.memory_bytes / 1024);

        // Calculate accuracy comparisons
        cout << "\nCalculating accuracy metrics..." << endl;

        // A vs true on DA items: How accurate is sketch A?
        result.a_vs_true_on_da.are = calculate_are_all_items(sketch_A, true_freqs_A);
        result.a_vs_true_on_da.aae = calculate_aae_all_items(sketch_A, true_freqs_A);
        result.a_vs_true_on_da.are_variance = calculate_are_variance(sketch_A, true_freqs_A, result.a_vs_true_on_da.are);
        result.a_vs_true_on_da.aae_variance = calculate_aae_variance(sketch_A, true_freqs_A, result.a_vs_true_on_da.aae);
        cout << format("  A vs True on DA: ARE={}, AAE={}\n", result.a_vs_true_on_da.are, result.a_vs_true_on_da.aae);

        // B vs true on DB items: How accurate is sketch B?
        result.b_vs_true_on_db.are = calculate_are_all_items(sketch_B, true_freqs_B);
        result.b_vs_true_on_db.aae = calculate_aae_all_items(sketch_B, true_freqs_B);
        result.b_vs_true_on_db.are_variance = calculate_are_variance(sketch_B, true_freqs_B, result.b_vs_true_on_db.are);
        result.b_vs_true_on_db.aae_variance = calculate_aae_variance(sketch_B, true_freqs_B, result.b_vs_true_on_db.aae);
        cout << format("  B vs True on DB: ARE={}, AAE={}\n", result.b_vs_true_on_db.are, result.b_vs_true_on_db.aae);

        // C (merged) vs true on all items: How accurate is the merged sketch?
        result.c_vs_true_on_all.are = calculate_are_all_items(sketch_C, true_freqs_all);
        result.c_vs_true_on_all.aae = calculate_aae_all_items(sketch_C, true_freqs_all);
        result.c_vs_true_on_all.are_variance = calculate_are_variance(sketch_C, true_freqs_all, result.c_vs_true_on_all.are);
        result.c_vs_true_on_all.aae_variance = calculate_aae_variance(sketch_C, true_freqs_all, result.c_vs_true_on_all.aae);
        cout << format("  C (merged) vs True on All: ARE={}, AAE={}\n", result.c_vs_true_on_all.are, result.c_vs_true_on_all.aae);

        // D (ground truth) vs true on all items: How accurate is the double-width sketch?
        result.d_vs_true_on_all.are = calculate_are_all_items(sketch_D, true_freqs_all);
        result.d_vs_true_on_all.aae = calculate_aae_all_items(sketch_D, true_freqs_all);
        result.d_vs_true_on_all.are_variance = calculate_are_variance(sketch_D, true_freqs_all, result.d_vs_true_on_all.are);
        result.d_vs_true_on_all.aae_variance = calculate_aae_variance(sketch_D, true_freqs_all, result.d_vs_true_on_all.aae);
        cout << format("  D (ground truth) vs True on All: ARE={}, AAE={}\n", result.d_vs_true_on_all.are, result.d_vs_true_on_all.aae);

        result.c_item_frequencies.reserve(true_freqs_all.size());
        result.d_item_frequencies.reserve(true_freqs_all.size());
        for (auto &[key, true_freq] : true_freqs_all)
        {
            MergeResult::ItemFrequency c_item_freq{key, true_freq, sketch_C.estimate(key)};
            result.c_item_frequencies.push_back(c_item_freq);
            MergeResult::ItemFrequency d_item_freq{key, true_freq, sketch_D.estimate(key)};
            result.d_item_frequencies.push_back(d_item_freq);
        }

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
    MergeConfig merge_config;
    ReSketchConfig rs_config;

    MergeConfig::add_params_to_config_parser(merge_config, parser);
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

    run_merge_experiment(merge_config, rs_config);

    return 0;
}
