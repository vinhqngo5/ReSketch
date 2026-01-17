#define DOCTEST_CONFIG_IMPLEMENT
#include "frequency_summary/frequency_summary_config.hpp"
#include "quantile_summary/quantile_summary_config.hpp"

#include "frequency_summary/count_min_sketch.hpp"
#include "frequency_summary/dynamic_sketch_wrapper.hpp"
#include "frequency_summary/geometric_sketch_wrapper.hpp"
#include "frequency_summary/resketch.hpp"
#include "frequency_summary/resketchv2.hpp"
#include "common.hpp"

#include "utils/ConfigParser.hpp"

#include "doctest.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace std;

// App Config
struct AppConfig
{
    uint64_t stream_size = 1000000;
    uint64_t stream_diversity = 10000;
    float zipf_param = 1.1;

    static void add_params_to_config_parser(AppConfig &config, ConfigParser &parser)
    {
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", "1000000", &config.stream_size, false, "Total items in stream"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", "10000", &config.stream_diversity, false, "Unique items in stream"));
        parser.AddParameter(new FloatParameter("app.zipf", "1.1", &config.zipf_param, false, "Zipfian param 'a'"));
    }
    friend std::ostream &operator<<(std::ostream &os, const AppConfig &config)
    {
        ConfigPrinter<AppConfig>::print(os, config);
        return os;
    }
    auto to_tuple() const { return std::make_tuple("stream_size", stream_size, "stream_diversity", stream_diversity, "zipf_param", zipf_param); }
};

// Evaluation result structure
struct EvaluationResult
{
    string name;
    double aae_top100 = 0.0, are_top100 = 0.0;
    double aae_top1k = 0.0, are_top1k = 0.0;
    double aae_all = 0.0, are_all = 0.0;
    double throughput = 0.0;
    uint32_t memory_kb = 0;

    template <typename SketchType>
    void calculate_error_for(const SketchType &sketch, const map<uint64_t, uint64_t> &true_freqs, const vector<uint64_t> &items, double &out_aae, double &out_are)
    {
        if (items.empty()) return;
        double total_abs_error = 0;
        double total_rel_error = 0;
        for (const auto &item : items)
        {
            double est_freq = sketch.estimate(item);
            double true_freq = true_freqs.at(item);
            double abs_error = abs(est_freq - true_freq);
            total_abs_error += abs_error;
            if (true_freq > 0) total_rel_error += abs_error / true_freq;
        }
        out_aae = total_abs_error / items.size();
        out_are = total_rel_error / items.size();
    }
};

void print_results(const string &title, const vector<EvaluationResult> &results)
{
    cout << "\n--- " << title << " ---\n\n";
    cout << "+--------------------------+----------+------------+------------+-----------+-----------+-----------+------------+------------+" << endl;
    cout << "| Sketch Name              | Mem (KB) | Tput(Mops) | AAE Top100 | ARE Top100| AAE Top1K | ARE Top1K |    AAE All |    ARE All |" << endl;
    cout << "+--------------------------+----------+------------+------------+-----------+-----------+-----------+------------+------------+" << endl;
    for (const auto &res : results)
    {
        cout << "| " << left << setw(24) << res.name << "| " << right << setw(8) << res.memory_kb << " | " << setw(10) << fixed << setprecision(2) << res.throughput << " | "
             << setw(10) << fixed << setprecision(2) << res.aae_top100 << " | " << setw(8) << fixed << setprecision(2) << res.are_top100 * 100.0 << "%"
             << " | " << setw(9) << fixed << setprecision(2) << res.aae_top1k << " | " << setw(8) << fixed << setprecision(2) << res.are_top1k * 100.0 << "%"
             << " | " << setw(10) << fixed << setprecision(2) << res.aae_all << " | " << setw(9) << fixed << setprecision(2) << res.are_all * 100.0 << "% |" << endl;
    }
    cout << "+--------------------------+----------+------------+------------+-----------+-----------+-----------+------------+------------+" << endl;
}

template <typename SketchType>
EvaluationResult evaluate(
    const string &name, const SketchType &sketch, const map<uint64_t, uint64_t> &true_freqs, const vector<uint64_t> &top100, const vector<uint64_t> &top1k,
    const vector<uint64_t> &all_unique, double duration_s, uint32_t stream_size)
{
    EvaluationResult res;
    res.name = name;
    res.memory_kb = sketch.get_max_memory_usage() / 1024;
    res.throughput = (duration_s > 0) ? ((static_cast<double>(stream_size) / duration_s) / 1000000.0) : 0;

    res.calculate_error_for(sketch, true_freqs, top100, res.aae_top100, res.are_top100);
    res.calculate_error_for(sketch, true_freqs, top1k, res.aae_top1k, res.are_top1k);
    res.calculate_error_for(sketch, true_freqs, all_unique, res.aae_all, res.are_all);
    return res;
}

void scenario_2_resize(const AppConfig &conf, CountMinConfig cm_conf, KLLConfig kll_conf, ReSketchConfig rs_conf, GeometricSketchConfig gs_conf, DynamicSketchConfig ds_conf)
{
    vector<EvaluationResult> results;
    Timer timer;

    cout << "Generating data for resize scenario..." << endl;
    auto data = generate_zipf_data(conf.stream_size, conf.stream_diversity, conf.zipf_param);
    auto true_freqs = get_true_freqs(data);
    cout << "Number of distinct items in stream: " << true_freqs.size() << " out of " << data.size() << " total items" << endl;
    auto top100 = get_top_k_items(true_freqs, 100);
    auto top1k = get_top_k_items(true_freqs, 1000);
    auto all_unique = get_top_k_items(true_freqs, true_freqs.size());

    // --- Create 2x size configs for static comparisons ---
    CountMinConfig cm_conf_x2 = cm_conf;
    cm_conf_x2.width *= 2;
    KLLConfig kll_conf_x2 = kll_conf;
    kll_conf_x2.k *= 2;
    ReSketchConfig rs_conf_x2 = rs_conf;
    rs_conf_x2.width *= 2;
    GeometricSketchConfig gs_conf_x2 = gs_conf;
    gs_conf_x2.width *= 2;
    DynamicSketchConfig ds_conf_x2 = ds_conf;
    ds_conf_x2.width *= 2;

    // --- Static Baselines ---
    {
        CountMinSketch s(cm_conf);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("CM (1x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        CountMinSketch s(cm_conf_x2);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("CM (2x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        KLL s(kll_conf);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("KLL (1x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        KLL s(kll_conf_x2);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("KLL (2x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        ReSketch s(rs_conf);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("ReSketch (1x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        ReSketch s(rs_conf_x2);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("ReSketch (2x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    // --- ReSketchV2 static baselines ---
    {
        ReSketchV2 s(rs_conf);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("ReSketchV2 (1x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        ReSketchV2 s(rs_conf_x2);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("ReSketchV2 (2x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        GeometricSketchWrapper s(gs_conf);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("GS (1x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        GeometricSketchWrapper s(gs_conf_x2);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("GS (2x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        DynamicSketchWrapper s(ds_conf);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("DS (1x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }
    {
        DynamicSketchWrapper s(ds_conf_x2);
        timer.start();
        for (const auto &i : data) s.update(i);
        double duration = timer.stop_s();
        results.push_back(evaluate("DS (2x)", s, true_freqs, top100, top1k, all_unique, duration, data.size()));
    }

    // --- Dynamic ReSketch: Expand mid-stream ---
    {
        ReSketch sketch(rs_conf);
        double total_duration = 0;
        uint32_t halfway = data.size() / 2;

        timer.start();
        for (uint32_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.expand(rs_conf.width * 2);

        timer.start();
        for (uint32_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("ReSketch (Expand)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }
    // --- Dynamic ReSketch: Shrink mid-stream ---
    {
        ReSketch sketch(rs_conf_x2);
        double total_duration = 0;
        uint32_t halfway = data.size() / 2;

        timer.start();
        for (uint32_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.shrink(rs_conf.width);

        timer.start();
        for (uint32_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("ReSketch (Shrink)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }
    // --- Dynamic ReSketchV2: Expand mid-stream ---
    {
        ReSketchV2 sketch(rs_conf);
        double total_duration = 0;
        uint32_t halfway = data.size() / 2;

        timer.start();
        for (uint32_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.expand(rs_conf.width * 2);

        timer.start();
        for (uint32_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("ReSketchV2 (Expand)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }
    // --- Dynamic ReSketchV2: Shrink mid-stream ---
    {
        ReSketchV2 sketch(rs_conf_x2);
        double total_duration = 0;
        uint32_t halfway = data.size() / 2;

        timer.start();
        for (uint32_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.shrink(rs_conf.width);

        timer.start();
        for (uint32_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("ReSketchV2 (Shrink)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }

    // --- Dynamic GeometricSketch: Expand mid-stream ---
    {
        GeometricSketchWrapper sketch(gs_conf);
        double total_duration = 0;
        uint32_t halfway = data.size() / 2;

        timer.start();
        for (uint32_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.expand(gs_conf.width * 2);

        timer.start();
        for (uint32_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("GS (Expand)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }
    // --- Dynamic DynamicSketch: Expand mid-stream ---
    {
        DynamicSketchWrapper sketch(ds_conf);
        double total_duration = 0;
        uint32_t halfway = data.size() / 2;

        timer.start();
        for (uint32_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.expand(ds_conf.width * 2);

        timer.start();
        for (uint32_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("DS (Expand)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }

    print_results("SCENARIO 2: DYNAMIC RESIZING", results);
}

void scenario_frequency_comparison(
    const AppConfig &conf, CountMinConfig cm_conf, KLLConfig kll_conf, ReSketchConfig rs_conf, GeometricSketchConfig gs_conf, DynamicSketchConfig ds_conf)
{
    Timer timer;

    cout << "\nGenerating data for frequency comparison..." << endl;
    auto data = generate_zipf_data(conf.stream_size, conf.stream_diversity, conf.zipf_param);
    auto true_freqs = get_true_freqs(data);
    cout << "Number of distinct items in stream: " << true_freqs.size() << " out of " << data.size() << " total items" << endl;
    auto top50 = get_top_k_items(true_freqs, 50);
    auto random100 = get_random_items(true_freqs, 100);

    // Create sketches with the same configuration
    CountMinSketch cm(cm_conf);
    KLL kll(kll_conf);
    ReSketch rs(rs_conf);
    ReSketchV2 rs_v2(rs_conf);
    GeometricSketchWrapper gs(gs_conf);
    DynamicSketchWrapper ds(ds_conf);

    // Update all sketches with the data
    cout << "Updating sketches..." << endl;
    for (const auto &item : data)
    {
        cm.update(item);
        kll.update(item);
        rs.update(item);
        rs_v2.update(item);
        gs.update(item);
        ds.update(item);
    }

    // Print frequency comparisons
    vector<string> sketch_names = {"CM", "KLL", "RS", "RSv2", "GS", "DS"};

    cout << "\n=== FREQUENCY COMPARISON ===" << endl;
    print_frequency_comparison("Top-50 Items", top50, true_freqs, sketch_names, cm, kll, rs, rs_v2, gs, ds);
    print_frequency_comparison("Random 100 Items", random100, true_freqs, sketch_names, cm, kll, rs, rs_v2, gs, ds);
}

int main(int argc, char **argv)
{
    ConfigParser parser;
    AppConfig app_configs;
    CountMinConfig count_min_configs;
    KLLConfig kll_configs;
    ReSketchConfig resketch_configs;
    GeometricSketchConfig geometric_sketch_configs;
    DynamicSketchConfig dynamic_sketch_configs;

    AppConfig::add_params_to_config_parser(app_configs, parser);
    CountMinConfig::add_params_to_config_parser(count_min_configs, parser);
    KLLConfig::add_params_to_config_parser(kll_configs, parser);
    ReSketchConfig::add_params_to_config_parser(resketch_configs, parser);
    GeometricSketchConfig::add_params_to_config_parser(geometric_sketch_configs, parser);
    DynamicSketchConfig::add_params_to_config_parser(dynamic_sketch_configs, parser);

    if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h"))
    {
        parser.PrintUsage();
        return 0;
    }
    if (argc > 1 && (string(argv[1]) == "--generate-doc"))
    {
        parser.PrintMarkdown();
        return 0;
    }

    Status s = parser.ParseCommandLine(argc, argv);
    if (!s.IsOK())
    {
        fprintf(stderr, "%s\n", s.ToString().c_str());
        return -1;
    }

    cout << app_configs;
    cout << count_min_configs;
    cout << kll_configs;
    cout << resketch_configs;
    cout << geometric_sketch_configs;
    cout << dynamic_sketch_configs;

    scenario_2_resize(app_configs, count_min_configs, kll_configs, resketch_configs, geometric_sketch_configs, dynamic_sketch_configs);

    scenario_frequency_comparison(app_configs, count_min_configs, kll_configs, resketch_configs, geometric_sketch_configs, dynamic_sketch_configs);

    return 0;
}
