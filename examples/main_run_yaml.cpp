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
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

// Library
#include "yaml-cpp/yaml.h"
#include "json/json.hpp"

// Utils
#include "utils/ConfigParser.hpp"

// Sketch Headers
#include "frequency_summary/frequency_summary_config.hpp"
#include "frequency_summary/resketchv2.hpp"
#include "quantile_summary/kll.hpp"

// Common utilities
#include "common.hpp"

using namespace std;
using json = nlohmann::json;

// Dataset configuration
struct DatasetConfig {
    string name;
    string dataset_type;
    string caida_path;
    uint64_t stream_size;
    uint64_t stream_diversity;
    double zipf_param;
};

struct DatasetReference {
    string dataset_name;
    uint64_t num_items;
    uint64_t start_offset;
};

// Sketch node in DAG
struct SketchNode {
    string name;
    string operation;
    uint32_t memory_budget_kb;
    vector<string> sources;
    vector<DatasetReference> datasets;
};

// DAG configuration
struct DAGConfig {
    string name;
    uint32_t repetitions;
    string output_file;
    uint32_t master_seed;

    map<string, DatasetConfig> datasets;

    uint32_t sketch_depth;
    uint32_t sketch_kll_k;

    vector<string> eval_metrics;
    uint64_t checkpoint_interval;

    map<string, SketchNode> sketches;
    vector<string> execution_order;
};

// Checkpoint data
struct Checkpoint {
    string sketch_name;
    uint64_t items_processed;
    double throughput_mops;
    double query_throughput_mops;
    uint64_t memory_kb;
    double are;
    double aae;
    double are_variance;
    double aae_variance;
};

// Result
struct StructuralOpResult {
    string sketch_name;
    string operation;
    double latency_s;
    uint64_t memory_kb;
    double are;
    double aae;
    double are_variance;
    double aae_variance;
};

struct RepetitionResult {
    uint32_t repetition_id;
    vector<Checkpoint> checkpoints;
    vector<StructuralOpResult> structural_ops;
};

DAGConfig parse_yaml(const string &yaml_file) {
    YAML::Node root = YAML::LoadFile(yaml_file);
    DAGConfig config;

    // Parse metadata
    auto metadata = root["metadata"];
    config.name = metadata["name"].as<string>();
    config.repetitions = metadata["repetitions"].as<uint32_t>();
    config.output_file = metadata["output_file"].as<string>();

    // Parse datasets
    auto datasets_node = root["datasets"];
    for (auto it = datasets_node.begin(); it != datasets_node.end(); ++it) {
        string dataset_name = it->first.as<string>();
        auto ds = it->second;

        DatasetConfig dataset;
        dataset.name = dataset_name;
        dataset.dataset_type = ds["dataset_type"].as<string>();
        dataset.stream_size = ds["stream_size"].as<uint64_t>();

        if (dataset.dataset_type == "caida") {
            dataset.caida_path = ds["caida_path"].as<string>();
        } else if (dataset.dataset_type == "zipf") {
            dataset.stream_diversity = ds["stream_diversity"].as<uint64_t>();
            dataset.zipf_param = ds["zipf_param"].as<double>();
        }

        config.datasets[dataset_name] = dataset;
    }

    // Parse sketch configuration
    auto sketch_config_node = root["sketch_config"];
    config.sketch_depth = sketch_config_node["depth"].as<uint32_t>();
    config.sketch_kll_k = sketch_config_node["kll_k"].as<uint32_t>();

    // Parse evaluation settings
    auto eval_node = root["evaluation"];
    for (const auto &metric : eval_node["metrics"]) { config.eval_metrics.push_back(metric.as<string>()); }
    config.checkpoint_interval = eval_node["checkpoint_intervals"].as<uint64_t>();

    // Parse sketch nodes
    auto sketches_node = root["sketches"];
    for (auto it = sketches_node.begin(); it != sketches_node.end(); ++it) {
        string sketch_name = it->first.as<string>();
        auto sk = it->second;

        SketchNode sketch;
        sketch.name = sketch_name;
        sketch.operation = sk["operation"].as<string>();
        sketch.memory_budget_kb = sk["memory_budget_kb"].as<uint32_t>();

        if (sk["source"]) { sketch.sources.push_back(sk["source"].as<string>()); }

        if (sk["sources"]) {
            for (const auto &src : sk["sources"]) { sketch.sources.push_back(src.as<string>()); }
        }

        if (sk["datasets"]) {
            for (const auto &ds_ref : sk["datasets"]) {
                DatasetReference ref;
                ref.dataset_name = ds_ref["dataset"].as<string>();
                ref.num_items = ds_ref["num_items"].as<uint64_t>();
                ref.start_offset = ds_ref["start_offset"] ? ds_ref["start_offset"].as<uint64_t>() : 0;
                sketch.datasets.push_back(ref);
            }
        }

        config.sketches[sketch_name] = sketch;
    }

    // Parse other options
    auto other_options = root["other_options"];
    config.master_seed = other_options["master_seed"].as<uint32_t>();

    return config;
}

// Topological sort to determine execution order
vector<string> topological_sort(const map<string, SketchNode> &sketches) {
    map<string, vector<string>> adjacency;
    map<string, int> in_degree;

    for (const auto &[name, sketch] : sketches) { in_degree[name] = 0; }

    for (const auto &[name, sketch] : sketches) {
        for (const auto &src : sketch.sources) {
            adjacency[src].push_back(name);
            in_degree[name]++;
        }
    }

    // Perform topological sort using Kahn's algorithm
    queue<string> q;
    for (const auto &[name, deg] : in_degree) {
        if (deg == 0) { q.push(name); }
    }

    vector<string> order;
    while (!q.empty()) {
        string current = q.front();
        q.pop();
        order.push_back(current);

        for (const auto &neighbor : adjacency[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) { q.push(neighbor); }
        }
    }

    if (order.size() != sketches.size()) {
        cerr << "Error: Cycle detected in DAG!" << endl;
        exit(1);
    }

    return order;
}

vector<uint64_t> load_or_generate_dataset(const DatasetConfig &dataset, uint64_t seed) {
    vector<uint64_t> data;

    if (dataset.dataset_type == "zipf") {
        data = generate_zipf_data(dataset.stream_size, dataset.stream_diversity, dataset.zipf_param);
    } else if (dataset.dataset_type == "caida") {
        data = read_caida_data(dataset.caida_path, dataset.stream_size);
        if (data.size() < dataset.stream_size) { cerr << "Warning: CAIDA dataset has fewer items than requested. Using full dataset." << endl; }
    }

    return data;
}

void process_data_with_checkpoints(ReSketchV2 &sketch, const vector<uint64_t> &data, uint64_t start_idx, uint64_t num_items, const string &sketch_name,
                                   uint64_t checkpoint_interval, const map<uint64_t, uint64_t> &ground_truth, vector<Checkpoint> &checkpoints_out) {
    uint64_t end_idx = min(start_idx + num_items, (uint64_t) data.size());
    uint64_t items_processed_in_phase = 0;

    Timer timer;
    timer.start();

    for (uint64_t i = start_idx; i < end_idx; ++i) {
        sketch.update(data[i]);
        items_processed_in_phase++;

        if (items_processed_in_phase % checkpoint_interval == 0 || i == end_idx - 1) {
            double elapsed = timer.stop_s();
            double throughput = (elapsed > 0) ? (items_processed_in_phase / elapsed / 1e6) : 0;

            // Prepare query items
            vector<uint64_t> query_items;
            query_items.reserve(ground_truth.size());
            for (const auto &[item, freq] : ground_truth) { query_items.push_back(item); }

            // Measure query throughput
            Timer query_timer;
            query_timer.start();
            volatile uint64_t sum = 0;
            for (const auto &item : query_items) { sum += sketch.estimate(item); }
            double query_elapsed = query_timer.stop_s();
            double query_throughput = (query_elapsed > 0) ? (query_items.size() / query_elapsed / 1e6) : 0;

            // Calculate error metrics
            double are = calculate_are_all_items(sketch, ground_truth);
            double aae = calculate_aae_all_items(sketch, ground_truth);
            double are_variance = calculate_are_variance(sketch, ground_truth, are);
            double aae_variance = calculate_aae_variance(sketch, ground_truth, aae);

            // Record checkpoint
            Checkpoint cp;
            cp.sketch_name = sketch_name;
            cp.items_processed = items_processed_in_phase;
            cp.throughput_mops = throughput;
            cp.query_throughput_mops = query_throughput;
            cp.memory_kb = sketch.get_max_memory_usage() / 1024;
            cp.are = are;
            cp.aae = aae;
            cp.are_variance = are_variance;
            cp.aae_variance = aae_variance;

            checkpoints_out.push_back(cp);

            items_processed_in_phase = 0;
            timer.start();
        }
    }
}

void export_to_json(const string &filename, const DAGConfig &config, const vector<RepetitionResult> &results) {
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

    j["metadata"] = {{"experiment_type", "dag"}, {"dag_name", config.name}, {"timestamp", timestamp.str()}};

    j["config"]["experiment"] = {{"repetitions", config.repetitions}, {"master_seed", config.master_seed}};

    j["config"]["sketch_config"] = {{"depth", config.sketch_depth}, {"kll_k", config.sketch_kll_k}};

    j["config"]["evaluation"] = {{"metrics", config.eval_metrics}, {"checkpoint_interval", config.checkpoint_interval}};

    // Datasets config section
    json datasets_json;
    for (const auto &[name, ds] : config.datasets) {
        datasets_json[name] = {{"dataset_type", ds.dataset_type}, {"stream_size", ds.stream_size}};
        if (ds.dataset_type == "zipf") {
            datasets_json[name]["stream_diversity"] = ds.stream_diversity;
            datasets_json[name]["zipf_param"] = ds.zipf_param;
        } else if (ds.dataset_type == "caida") {
            datasets_json[name]["caida_path"] = ds.caida_path;
        }
    }
    j["config"]["datasets"] = datasets_json;

    // Sketches config section
    json sketches_json;
    for (const auto &sketch_name : config.execution_order) {
        const auto &sketch = config.sketches.at(sketch_name);
        sketches_json[sketch_name] = {{"operation", sketch.operation}, {"memory_budget_kb", sketch.memory_budget_kb}};
        if (!sketch.sources.empty()) { sketches_json[sketch_name]["sources"] = sketch.sources; }
    }
    j["config"]["sketches"] = sketches_json;

    // Results section
    j["results"] = json::array();
    for (const auto &rep : results) {
        json rep_json;
        rep_json["repetition_id"] = rep.repetition_id;

        rep_json["checkpoints"] = json::array();
        for (const auto &cp : rep.checkpoints) {
            rep_json["checkpoints"].push_back({{{"sketch_name", cp.sketch_name},
                                                {"items_processed", cp.items_processed},
                                                {"throughput_mops", cp.throughput_mops},
                                                {"query_throughput_mops", cp.query_throughput_mops},
                                                {"memory_kb", cp.memory_kb},
                                                {"are", cp.are},
                                                {"aae", cp.aae},
                                                {"are_variance", cp.are_variance},
                                                {"aae_variance", cp.aae_variance}}});
        }

        rep_json["structural_operations"] = json::array();
        for (const auto &op : rep.structural_ops) {
            rep_json["structural_operations"].push_back({{{"sketch_name", op.sketch_name},
                                                          {"operation", op.operation},
                                                          {"latency_s", op.latency_s},
                                                          {"memory_kb", op.memory_kb},
                                                          {"are", op.are},
                                                          {"aae", op.aae},
                                                          {"are_variance", op.are_variance},
                                                          {"aae_variance", op.aae_variance}}});
        }

        j["results"].push_back(rep_json);
    }

    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: Cannot open output file: " << filename << endl;
        return;
    }

    out << j.dump(2);
    out.close();

    cout << "\nResults exported to: " << filename << endl;
}

void run_dag_experiment(const DAGConfig &config) {
    cout << "\n=== DAG Execution: " << config.name << " ===" << endl;
    cout << "Repetitions: " << config.repetitions << endl;
    cout << "Master Seed: " << config.master_seed << endl;
    cout << "Sketch Config: depth=" << config.sketch_depth << ", k=" << config.sketch_kll_k << endl;
    cout << "Execution Order: ";
    for (const auto &name : config.execution_order) { cout << name << " "; }
    cout << "\n" << endl;

    vector<RepetitionResult> all_results;

    for (uint32_t rep = 0; rep < config.repetitions; ++rep) {
        cout << "\n========================================" << endl;
        cout << "Repetition " << (rep + 1) << "/" << config.repetitions << endl;
        cout << "========================================" << endl;

        RepetitionResult rep_result;
        rep_result.repetition_id = rep;

        // Initialize RNG for this repetition
        std::mt19937_64 rng(config.master_seed + rep);
        std::uniform_int_distribution<uint32_t> dist;

        // Generate shared seeds for sketch construction
        uint32_t shared_partition_seed = dist(rng);
        vector<uint32_t> shared_seeds;
        shared_seeds.reserve(config.sketch_depth);
        for (uint32_t i = 0; i < config.sketch_depth; ++i) { shared_seeds.push_back(dist(rng)); }

        // Load all datasets
        map<string, vector<uint64_t>> loaded_datasets;
        for (const auto &[name, ds_config] : config.datasets) {
            uint32_t dataset_seed = dist(rng);
            loaded_datasets[name] = load_or_generate_dataset(ds_config, dataset_seed);
            cout << "Loaded dataset '" << name << "': " << loaded_datasets[name].size() << " items" << endl;
        }

        map<string, unique_ptr<ReSketchV2>> sketches;
        map<string, map<uint64_t, uint64_t>> sketch_ground_truths;
        set<string> skip_split_operation;

        // Process each sketch node in topological order
        for (const auto &sketch_name : config.execution_order) {
            const auto &sketch_node = config.sketches.at(sketch_name);

            // Skip structural operation for split sibling, but still process data
            bool skip_structural_op = skip_split_operation.count(sketch_name) > 0;

            if (skip_structural_op) {
                cout << "\n--- Processing Sketch " << sketch_name << " (split sibling - already created) ---" << endl;
            } else {
                cout << "\n--- Processing Sketch " << sketch_name << " (" << sketch_node.operation << ") ---" << endl;
            }

            uint64_t memory_bytes = (uint64_t) sketch_node.memory_budget_kb * 1024;
            uint32_t width = ReSketchV2::calculate_max_width(memory_bytes, config.sketch_depth, config.sketch_kll_k);

            // Perform structural operation
            if (!skip_structural_op) {
                if (sketch_node.operation == "create") {
                    sketches[sketch_name] = make_unique<ReSketchV2>(config.sketch_depth, width, shared_seeds, config.sketch_kll_k, shared_partition_seed);
                    sketch_ground_truths[sketch_name] = map<uint64_t, uint64_t>();
                    uint64_t actual_memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    cout << "Created sketch with width=" << width << " | budget=" << sketch_node.memory_budget_kb << " KB, actual=" << actual_memory_kb << " KB" << endl;

                } else if (sketch_node.operation == "expand") {
                    if (sketch_node.sources.empty() || sketches.find(sketch_node.sources[0]) == sketches.end()) {
                        cerr << "Error: Source sketch for expand not found!" << endl;
                        exit(1);
                    }

                    string source_name = sketch_node.sources[0];
                    Timer op_timer;
                    op_timer.start();

                    sketches[source_name]->expand(width);

                    double latency = op_timer.stop_s();

                    sketches[sketch_name] = std::move(sketches[source_name]);
                    sketches.erase(source_name);

                    sketch_ground_truths[sketch_name] = std::move(sketch_ground_truths[source_name]);
                    sketch_ground_truths.erase(source_name);

                    double are = calculate_are_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double aae = calculate_aae_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double are_variance = calculate_are_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], are);
                    double aae_variance = calculate_aae_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], aae);

                    StructuralOpResult op_result;
                    op_result.sketch_name = sketch_name;
                    op_result.operation = "expand";
                    op_result.latency_s = latency;
                    op_result.memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    op_result.are = are;
                    op_result.aae = aae;
                    op_result.are_variance = are_variance;
                    op_result.aae_variance = aae_variance;
                    rep_result.structural_ops.push_back(op_result);

                    uint64_t actual_memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    cout << "Expanded from " << source_name << " to width=" << width << " | budget=" << sketch_node.memory_budget_kb << " KB, actual=" << actual_memory_kb
                         << " KB, latency=" << latency << "s" << endl;

                } else if (sketch_node.operation == "shrink") {
                    if (sketch_node.sources.empty() || sketches.find(sketch_node.sources[0]) == sketches.end()) {
                        cerr << "Error: Source sketch for shrink not found!" << endl;
                        exit(1);
                    }

                    string source_name = sketch_node.sources[0];
                    Timer op_timer;
                    op_timer.start();

                    sketches[source_name]->shrink(width);

                    double latency = op_timer.stop_s();

                    sketches[sketch_name] = std::move(sketches[source_name]);
                    sketches.erase(source_name);

                    sketch_ground_truths[sketch_name] = std::move(sketch_ground_truths[source_name]);
                    sketch_ground_truths.erase(source_name);

                    double are = calculate_are_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double aae = calculate_aae_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double are_variance = calculate_are_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], are);
                    double aae_variance = calculate_aae_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], aae);

                    StructuralOpResult op_result;
                    op_result.sketch_name = sketch_name;
                    op_result.operation = "shrink";
                    op_result.latency_s = latency;
                    op_result.memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    op_result.are = are;
                    op_result.aae = aae;
                    op_result.are_variance = are_variance;
                    op_result.aae_variance = aae_variance;
                    rep_result.structural_ops.push_back(op_result);

                    uint64_t actual_memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    cout << "Shrunk from " << source_name << " to width=" << width << " | budget=" << sketch_node.memory_budget_kb << " KB, actual=" << actual_memory_kb
                         << " KB, latency=" << latency << "s" << endl;

                } else if (sketch_node.operation == "merge") {
                    Timer op_timer;
                    op_timer.start();

                    if (sketch_node.sources.size() < 2) {
                        cerr << "Error: Merge operation requires at least 2 sources!" << endl;
                        exit(1);
                    }

                    for (const auto &source_name : sketch_node.sources) {
                        if (sketches.find(source_name) == sketches.end()) {
                            cerr << "Error: Source sketch " << source_name << " not found!" << endl;
                            exit(1);
                        }
                    }

                    ReSketchV2 merged = ReSketchV2::merge(*sketches[sketch_node.sources[0]], *sketches[sketch_node.sources[1]]);

                    for (size_t i = 2; i < sketch_node.sources.size(); ++i) { merged = ReSketchV2::merge(merged, *sketches[sketch_node.sources[i]]); }

                    double latency = op_timer.stop_s();

                    sketches[sketch_name] = make_unique<ReSketchV2>(std::move(merged));

                    sketch_ground_truths[sketch_name] = map<uint64_t, uint64_t>();
                    for (const auto &source_name : sketch_node.sources) {
                        for (const auto &[item, freq] : sketch_ground_truths[source_name]) { sketch_ground_truths[sketch_name][item] += freq; }
                    }

                    double are = calculate_are_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double aae = calculate_aae_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double are_variance = calculate_are_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], are);
                    double aae_variance = calculate_aae_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], aae);

                    StructuralOpResult op_result;
                    op_result.sketch_name = sketch_name;
                    op_result.operation = "merge";
                    op_result.latency_s = latency;
                    op_result.memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    op_result.are = are;
                    op_result.aae = aae;
                    op_result.are_variance = are_variance;
                    op_result.aae_variance = aae_variance;
                    rep_result.structural_ops.push_back(op_result);

                    uint64_t actual_memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    cout << "Merged sources: ";
                    for (const auto &src : sketch_node.sources) { cout << src << " "; }
                    cout << "-> " << sketch_name << " | budget=" << sketch_node.memory_budget_kb << " KB, actual=" << actual_memory_kb
                         << " KB (sum of sources), latency=" << latency << "s" << endl;

                } else if (sketch_node.operation == "split") {
                    if (sketch_node.sources.empty() || sketches.find(sketch_node.sources[0]) == sketches.end()) {
                        cerr << "Error: Source sketch for split not found!" << endl;
                        exit(1);
                    }

                    string source_name = sketch_node.sources[0];

                    auto sibling_iter = std::find(config.execution_order.begin(), config.execution_order.end(), sketch_name);
                    if (sibling_iter == config.execution_order.end() || std::next(sibling_iter) == config.execution_order.end()) {
                        cerr << "Error: Split operation requires a sibling sketch in execution order!" << endl;
                        exit(1);
                    }

                    string sibling_name = *std::next(sibling_iter);
                    const auto &sibling_node = config.sketches.at(sibling_name);

                    if (sibling_node.operation != "split" || sibling_node.sources.empty() || sibling_node.sources[0] != source_name) {
                        cerr << "Error: Split sibling mismatch! Expected " << sibling_name << " to split from " << source_name << endl;
                        exit(1);
                    }

                    uint32_t source_width = sketches[source_name]->get_max_memory_usage() / (config.sketch_depth * (KLL({config.sketch_kll_k}).get_max_memory_usage()));

                    uint64_t total_memory = (uint64_t) sketch_node.memory_budget_kb * 1024 + (uint64_t) sibling_node.memory_budget_kb * 1024;
                    uint64_t memory_ratio_first = (uint64_t) sketch_node.memory_budget_kb * 1024;

                    uint32_t width_first = (uint32_t) ((double) source_width * memory_ratio_first / total_memory);
                    uint32_t width_second = source_width - width_first;

                    cout << "Source width: " << source_width << ", splitting into " << width_first << " + " << width_second << endl;

                    Timer op_timer;
                    op_timer.start();

                    auto split_result = ReSketchV2::split(*sketches[source_name], width_first, width_second);

                    double latency = op_timer.stop_s();

                    sketches[sketch_name] = make_unique<ReSketchV2>(std::move(split_result.first));
                    sketches[sibling_name] = make_unique<ReSketchV2>(std::move(split_result.second));

                    // Filter ground truth for each split sketch based on partition responsibility
                    map<uint64_t, uint64_t> ground_truth_first, ground_truth_second;
                    for (const auto &[item, freq] : sketch_ground_truths[source_name]) {
                        if (sketches[sketch_name]->is_responsible_for(item)) {
                            ground_truth_first[item] = freq;
                        } else {
                            ground_truth_second[item] = freq;
                        }
                    }
                    sketch_ground_truths[sketch_name] = ground_truth_first;
                    sketch_ground_truths[sibling_name] = ground_truth_second;

                    double are = calculate_are_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double aae = calculate_aae_all_items(*sketches[sketch_name], sketch_ground_truths[sketch_name]);
                    double are_variance = calculate_are_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], are);
                    double aae_variance = calculate_aae_variance(*sketches[sketch_name], sketch_ground_truths[sketch_name], aae);

                    StructuralOpResult op_result;
                    op_result.sketch_name = sketch_name;
                    op_result.operation = "split";
                    op_result.latency_s = latency;
                    op_result.memory_kb = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    op_result.are = are;
                    op_result.aae = aae;
                    op_result.are_variance = are_variance;
                    op_result.aae_variance = aae_variance;
                    rep_result.structural_ops.push_back(op_result);

                    double are2 = calculate_are_all_items(*sketches[sibling_name], sketch_ground_truths[sibling_name]);
                    double aae2 = calculate_aae_all_items(*sketches[sibling_name], sketch_ground_truths[sibling_name]);

                    StructuralOpResult op_result2;
                    op_result2.sketch_name = sibling_name;
                    op_result2.operation = "split";
                    op_result2.latency_s = latency;
                    op_result2.memory_kb = sketches[sibling_name]->get_max_memory_usage() / 1024;
                    op_result2.are = are2;
                    op_result2.aae = aae2;
                    rep_result.structural_ops.push_back(op_result2);

                    uint64_t actual_memory_kb_first = sketches[sketch_name]->get_max_memory_usage() / 1024;
                    uint64_t actual_memory_kb_second = sketches[sibling_name]->get_max_memory_usage() / 1024;
                    cout << "Split from " << source_name << " -> " << sketch_name << " + " << sibling_name << " | " << sketch_name << " (budget=" << sketch_node.memory_budget_kb
                         << " KB, actual=" << actual_memory_kb_first << " KB), " << sibling_name << " (budget=" << sibling_node.memory_budget_kb
                         << " KB, actual=" << actual_memory_kb_second << " KB), latency=" << latency << "s" << endl;

                    sketches.erase(source_name);
                    sketch_ground_truths.erase(source_name);
                    skip_split_operation.insert(sibling_name);
                }
            }

            // Process datasets for this sketch
            if (!sketch_node.datasets.empty()) {
                cout << "Processing datasets for " << sketch_name << "..." << endl;

                for (const auto &ds_ref : sketch_node.datasets) {
                    const auto &data = loaded_datasets[ds_ref.dataset_name];

                    cout << "  Dataset: " << ds_ref.dataset_name << ", items: " << ds_ref.num_items << ", offset: " << ds_ref.start_offset << endl;

                    // Check if sketch has full partition coverage (no filtering needed)
                    auto partition_ranges = sketches[sketch_name]->get_partition_ranges();
                    bool has_full_coverage = (partition_ranges.size() == 1 && partition_ranges[0].first == 0 && partition_ranges[0].second == UINT64_MAX);

                    if (has_full_coverage) {
                        // Sketch covers all partitions -> process all data
                        cout << "  Sketch has full partition coverage -> processing all items" << endl;
                        for (uint64_t i = ds_ref.start_offset; i < min(ds_ref.start_offset + ds_ref.num_items, (uint64_t) data.size()); ++i) {
                            sketch_ground_truths[sketch_name][data[i]]++;
                        }

                        process_data_with_checkpoints(*sketches[sketch_name], data, ds_ref.start_offset, ds_ref.num_items, sketch_name, config.checkpoint_interval,
                                                      sketch_ground_truths[sketch_name], rep_result.checkpoints);
                    } else {
                        // Sketch has partial partition -> filter data based on responsibility
                        cout << "  Sketch has partial partition coverage -> filtering items" << endl;
                        cout << "  Partition ranges: ";
                        for (const auto &[start, end] : partition_ranges) { cout << "[" << start << ", " << end << ") "; }
                        cout << endl;

                        // Scan and collect exactly num_items that pass the filter
                        vector<uint64_t> filtered_data;
                        filtered_data.reserve(ds_ref.num_items);

                        uint64_t items_collected = 0;
                        uint64_t items_scanned = 0;
                        uint64_t scan_idx = ds_ref.start_offset;

                        while (items_collected < ds_ref.num_items && scan_idx < data.size()) {
                            if (sketches[sketch_name]->is_responsible_for(data[scan_idx])) {
                                filtered_data.push_back(data[scan_idx]);
                                sketch_ground_truths[sketch_name][data[scan_idx]]++;
                                items_collected++;
                            }
                            items_scanned++;
                            scan_idx++;
                        }

                        cout << "  Filtered: " << items_collected << " items collected (scanned " << items_scanned << " items)" << endl;

                        // Process filtered data
                        if (!filtered_data.empty()) {
                            process_data_with_checkpoints(*sketches[sketch_name], filtered_data, 0, filtered_data.size(), sketch_name, config.checkpoint_interval,
                                                          sketch_ground_truths[sketch_name], rep_result.checkpoints);
                        }
                    }
                }
            }
        }

        all_results.push_back(rep_result);
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
    size_t ext_pos = output_file.find_last_of('.');
    if (ext_pos != string::npos) {
        output_file = output_file.substr(0, ext_pos) + "_" + timestamp + output_file.substr(ext_pos);
    } else {
        output_file += "_" + timestamp;
    }

    export_to_json(output_file, config, all_results);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <yaml_file>" << endl;
        return 1;
    }

    string yaml_file = argv[1];

    try {
        DAGConfig config = parse_yaml(yaml_file);
        config.execution_order = topological_sort(config.sketches);

        run_dag_experiment(config);

    } catch (const YAML::Exception &e) {
        cerr << "YAML parsing error: " << e.what() << endl;
        return 1;
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
