#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Utils
#include "utils/ConfigParser.hpp"

// Sketch Headers (now templated)
#include "frequency_summary/count_min_sketch.hpp"
#include "frequency_summary/resketch.hpp"
#include "quantile_summary/kll.hpp"

// Config Headers
#include "frequency_summary/frequency_summary_config.hpp"
#include "quantile_summary/quantile_summary_config.hpp"

using namespace std;

// App Config
struct AppConfig {
    uint64_t stream_size = 1000000;
    uint64_t stream_diversity = 10000;
    double zipf_param = 1.1;

    static void add_params_to_config_parser(AppConfig &config, ConfigParser &parser) {
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_size", "1000000", &config.stream_size, false, "Total items in stream"));
        parser.AddParameter(new UnsignedInt64Parameter("app.stream_diversity", "10000", &config.stream_diversity, false, "Unique items in stream"));
        parser.AddParameter(new FloatParameter("app.zipf", "1.1", reinterpret_cast<float *>(&config.zipf_param), false, "Zipfian param 'a'"));
    }
    friend std::ostream &operator<<(std::ostream &os, const AppConfig &config) {
        ConfigPrinter<AppConfig>::print(os, config);
        return os;
    }
    auto to_tuple() const { return std::make_tuple("stream_size", stream_size, "stream_diversity", stream_diversity, "zipf_param", zipf_param); }
};

// Timer class to measure execution time
class Timer {
   public:
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    double stop_s() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - m_start).count();
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

// Evaluation result structure
struct EvaluationResult {
    string name;
    double aae_top100 = 0.0, are_top100 = 0.0;
    double aae_top1k = 0.0, are_top1k = 0.0;
    double aae_all = 0.0, are_all = 0.0;
    double throughput = 0.0;
    size_t memory_kb = 0;

    template <typename SketchType>
    void calculate_error_for(const SketchType &sketch, const map<uint64_t, uint64_t> &true_freqs, const vector<uint64_t> &items, double &out_aae, double &out_are) {
        if (items.empty()) return;
        double total_abs_error = 0;
        double total_rel_error = 0;
        for (const auto &item : items) {
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

// --- Core Helper Functions ---

vector<uint64_t> generate_zipf_data(uint64_t size, uint64_t diversity, double a) {
    vector<double> pdf(diversity);
    double sum = 0.0;
    for (uint64_t i = 1; i <= diversity; ++i) {
        pdf[i - 1] = 1.0 / pow(static_cast<double>(i), a);
        sum += pdf[i - 1];
    }
    for (uint64_t i = 0; i < diversity; ++i) {
        pdf[i] /= sum;
    }
    std::discrete_distribution<uint64_t> dist(pdf.begin(), pdf.end());
    std::mt19937_64 rng(std::random_device{}());
    vector<uint64_t> data;
    data.reserve(size);
    for (uint64_t i = 0; i < size; ++i) {
        data.push_back(dist(rng));
    }
    return data;
}

map<uint64_t, uint64_t> get_true_freqs(const vector<uint64_t> &data) {
    map<uint64_t, uint64_t> freqs;
    for (const auto &item : data) {
        freqs[item]++;
    }
    return freqs;
}

vector<uint64_t> get_top_k_items(const map<uint64_t, uint64_t> &freqs, int k) {
    vector<pair<uint64_t, uint64_t>> sorted_freqs(freqs.begin(), freqs.end());
    sort(sorted_freqs.begin(), sorted_freqs.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
    vector<uint64_t> top_items;
    top_items.reserve(min((size_t)k, sorted_freqs.size()));
    for (int i = 0; i < k && i < sorted_freqs.size(); ++i) {
        top_items.push_back(sorted_freqs[i].first);
    }
    return top_items;
}

void print_results(const string &title, const vector<EvaluationResult> &results) {
    cout << "\n--- " << title << " ---\n\n";
    cout << "+--------------------------+----------+------------+------------+-----------+-----------+-----------+------------+------------+" << endl;
    cout << "| Sketch Name              | Mem (KB) | Tput(Mops) | AAE Top100 | ARE Top100| AAE Top1K | ARE Top1K |    AAE All |    ARE All |" << endl;
    cout << "+--------------------------+----------+------------+------------+-----------+-----------+-----------+------------+------------+" << endl;
    for (const auto &res : results) {
        cout << "| " << left << setw(24) << res.name << "| " << right << setw(8) << res.memory_kb << " | " << setw(10) << fixed << setprecision(2) << res.throughput << " | " << setw(10) << fixed << setprecision(2) << res.aae_top100 << " | " << setw(8) << fixed << setprecision(2) << res.are_top100 * 100.0 << "%"
             << " | " << setw(9) << fixed << setprecision(2) << res.aae_top1k << " | " << setw(8) << fixed << setprecision(2) << res.are_top1k * 100.0 << "%"
             << " | " << setw(10) << fixed << setprecision(2) << res.aae_all << " | " << setw(9) << fixed << setprecision(2) << res.are_all * 100.0 << "% |" << endl;
    }
    cout << "+--------------------------+----------+------------+------------+-----------+-----------+-----------+------------+------------+" << endl;
}

template <typename SketchType>
EvaluationResult evaluate(const string &name, const SketchType &sketch, const map<uint64_t, uint64_t> &true_freqs, const vector<uint64_t> &top100, const vector<uint64_t> &top1k, const vector<uint64_t> &all_unique, double duration_s, size_t stream_size) {
    EvaluationResult res;
    res.name = name;
    res.memory_kb = sketch.get_max_memory_usage() / 1024;
    res.throughput = (duration_s > 0) ? ((static_cast<double>(stream_size) / duration_s) / 1000000.0) : 0;

    res.calculate_error_for(sketch, true_freqs, top100, res.aae_top100, res.are_top100);
    res.calculate_error_for(sketch, true_freqs, top1k, res.aae_top1k, res.are_top1k);
    res.calculate_error_for(sketch, true_freqs, all_unique, res.aae_all, res.are_all);
    return res;
}

void scenario_2_resize(const AppConfig &conf, CountMinConfig cm_conf, KLLConfig kll_conf, ReSketchConfig rs_conf) {
    vector<EvaluationResult> results;
    Timer timer;

    cout << "Generating data for resize scenario..." << endl;
    auto data = generate_zipf_data(conf.stream_size, conf.stream_diversity, conf.zipf_param);
    auto true_freqs = get_true_freqs(data);
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

    // --- Dynamic ReSketch: Expand mid-stream ---
    {
        ReSketch sketch(rs_conf);
        double total_duration = 0;
        size_t halfway = data.size() / 2;

        timer.start();
        for (size_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.expand(rs_conf.width * 2);

        timer.start();
        for (size_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("ReSketch (Expand)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }
    // --- Dynamic ReSketch: Shrink mid-stream ---
    {
        ReSketch sketch(rs_conf_x2);
        double total_duration = 0;
        size_t halfway = data.size() / 2;

        timer.start();
        for (size_t i = 0; i < halfway; ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        sketch.shrink(rs_conf.width);

        timer.start();
        for (size_t i = halfway; i < data.size(); ++i) sketch.update(data[i]);
        total_duration += timer.stop_s();

        results.push_back(evaluate("ReSketch (Shrink)", sketch, true_freqs, top100, top1k, all_unique, total_duration, data.size()));
    }

    print_results("SCENARIO 2: DYNAMIC RESIZING", results);
}

int main(int argc, char **argv) {
    ConfigParser parser;
    AppConfig app_configs;
    CountMinConfig count_min_configs;
    KLLConfig kll_configs;
    ReSketchConfig resketch_configs;

    AppConfig::add_params_to_config_parser(app_configs, parser);
    CountMinConfig::add_params_to_config_parser(count_min_configs, parser);
    KLLConfig::add_params_to_config_parser(kll_configs, parser);
    ReSketchConfig::add_params_to_config_parser(resketch_configs, parser);

    if (argc > 1 && (string(argv[1]) == "--help" || string(argv[1]) == "-h")) {
        parser.PrintUsage();
        return 0;
    }
    if (argc > 1 && (string(argv[1]) == "--generate-doc")) {
        parser.PrintMarkdown();
        return 0;
    }

    Status s = parser.ParseCommandLine(argc, argv);
    if (!s.IsOK()) {
        fprintf(stderr, "%s\n", s.ToString().c_str());
        return -1;
    }

    cout << app_configs;
    cout << count_min_configs;
    cout << kll_configs;
    cout << resketch_configs;

    scenario_2_resize(app_configs, count_min_configs, kll_configs, resketch_configs);

    return 0;
}