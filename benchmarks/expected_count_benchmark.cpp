/**
 * Expected Count per Bucket Benchmark for Consistent Hashing
 * Measures E[count in bucket where query lands] ≈ 2N/w (size-biased sampling)
 * Test:  ./build/release/bin/release/expected_count_benchmark --trials 30 --items 1000000 --queries 100000 --width 1000
 */

#include "frequency_summary/resketchv2.hpp"

#include "json/json.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using json = nlohmann::json;

class ConsistentHashingRing
{
public:
    using Ring = std::vector<std::pair<uint64_t, uint32_t>>;

    ConsistentHashingRing(uint32_t width, uint64_t seed = 0) : m_width(width)
    {
        std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
        std::uniform_int_distribution<uint64_t> dist;
        std::uniform_int_distribution<uint32_t> seed_dist;

        m_partition_seed = seed_dist(rng);

        // Initialize pairwise independent hash parameters
        std::mt19937_64 param_rng(m_partition_seed);
        m_a = (param_rng() | 1);
        m_b = param_rng();

        // Initialize ring with random hash points
        m_ring.reserve(width);
        for (uint32_t j = 0; j < width; ++j) { m_ring.push_back({dist(rng), j}); }
        std::sort(m_ring.begin(), m_ring.end());
    }

    uint64_t get_bucket(uint64_t item) const
    {
        uint64_t h = _placement_hash(item);
        return _find_bucket_id(h, m_ring);
    }

private:
    uint64_t _partition_hash(uint64_t item) const { return XXHash64::hash(&item, sizeof(uint64_t), m_partition_seed); }

    uint64_t _placement_hash(uint64_t item) const
    {
        uint64_t partition_h = _partition_hash(item);
        return m_a * partition_h + m_b;
    }

    static uint32_t _find_bucket_id(uint64_t item_hash, const Ring &ring)
    {
        auto it = std::lower_bound(ring.begin(), ring.end(), std::make_pair(item_hash, std::numeric_limits<uint32_t>::max()));
        if (it == ring.end()) return ring.front().second;
        return it->second;
    }

    uint32_t m_width;
    uint32_t m_partition_seed;
    uint64_t m_a;   // Pairwise hash parameter
    uint64_t m_b;   // Pairwise hash parameter
    Ring m_ring;
};

struct BucketCountResult
{
    double avg_bucket_count;
    double ratio_to_n_over_w;
};

BucketCountResult measure_expected_bucket_count(uint32_t width, uint64_t num_items, uint64_t num_queries)
{
    ConsistentHashingRing ring(width);
    std::vector<uint64_t> bucket_counts(width, 0);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;

    for (uint64_t i = 0; i < num_items; ++i)
    {
        uint64_t item = dist(rng);
        bucket_counts[ring.get_bucket(item)]++;
    }

    double total = 0.0;
    for (uint64_t i = 0; i < num_queries; ++i)
    {
        uint64_t query = dist(rng);
        total += bucket_counts[ring.get_bucket(query)];
    }

    double avg_count = total / num_queries;
    double n_over_w = static_cast<double>(num_items) / width;
    return {avg_count, avg_count / n_over_w};
}

int main(int argc, char *argv[])
{
    std::cout << "Expected Count per Bucket Benchmark\n" << std::string(80, '=') << std::endl;

    uint32_t width = 100;
    uint64_t num_items = 100000;
    uint64_t num_queries = 100000;
    uint32_t num_trials = 100;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) { width = std::stoul(argv[++i]); }
        else if (arg == "--items" && i + 1 < argc) { num_items = std::stoull(argv[++i]); }
        else if (arg == "--queries" && i + 1 < argc) { num_queries = std::stoull(argv[++i]); }
        else if (arg == "--trials" && i + 1 < argc) { num_trials = std::stoul(argv[++i]); }
        else if (arg == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --width N     Number of buckets (default: 100)\n"
                      << "  --items N     Number of items to insert (default: 100000)\n"
                      << "  --queries N   Number of queries (default: 100000)\n"
                      << "  --trials N    Number of trials (default: 100)\n";
            return 0;
        }
    }

    double n_over_w = static_cast<double>(num_items) / width;
    std::cout << "Config: width=" << width << ", items=" << num_items << ", queries=" << num_queries << ", trials=" << num_trials << ", N/w=" << std::fixed << std::setprecision(2)
              << n_over_w << "\n"
              << std::endl;

    std::vector<double> ratios;
    std::vector<double> bucket_counts;

    for (uint32_t trial = 0; trial < num_trials; ++trial)
    {
        auto result = measure_expected_bucket_count(width, num_items, num_queries);
        ratios.push_back(result.ratio_to_n_over_w);
        bucket_counts.push_back(result.avg_bucket_count);
    }

    std::sort(ratios.begin(), ratios.end());
    double avg_ratio = std::accumulate(ratios.begin(), ratios.end(), 0.0) / num_trials;
    double avg_bucket_count = std::accumulate(bucket_counts.begin(), bucket_counts.end(), 0.0) / num_trials;
    double median_ratio = ratios[num_trials / 2];

    std::cout << "\nRESULTS" << std::endl;
    std::cout << "Avg. Items in Queried Bucket:   " << std::fixed << std::setprecision(4) << avg_bucket_count << std::endl;
    std::cout << "Avg. Bias vs. Uniform ratio (N/W):    " << avg_ratio << "x" << std::endl;
    std::cout << "Median Bias vs. Uniform ratio (N/W):  " << median_ratio << "x" << std::endl;

    json results;
    results["config"] = {{"width", width}, {"num_items", num_items}, {"num_queries", num_queries}, {"num_trials", num_trials}};
    results["results"] = {{"avg_count", avg_bucket_count}, {"avg_ratio", avg_ratio}, {"median_ratio", median_ratio}};
    results["all_ratios"] = ratios;

    std::ofstream out("output/expected_count_results.json");
    if (out)
    {
        out << results.dump(2);
        std::cout << "\nSaved: output/expected_count_results.json" << std::endl;
    }

    return 0;
}
