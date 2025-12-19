#pragma once

#include "frequency_summary_config.hpp"

#include "frequency_summary.hpp"
#include "hash/xxhash64.hpp"
#include "quantile_summary/kll.hpp"
#include "quantile_summary/kll_datasketches.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

class ReSketch : public FrequencySummary
{
private:
    struct Bucket
    {
        uint64_t count = 0;
        KLL q_sketch;

        Bucket() = default;
        explicit Bucket(const KLLConfig &kll_config) : q_sketch(kll_config) {}
    };

    // A ring is a sorted list of pairs: {hash_point, bucket_id}
    using Ring = std::vector<std::pair<uint64_t, uint32_t>>;

public:
    explicit ReSketch(const ReSketchConfig &config) : m_config(config), m_width(config.width), m_depth(config.depth), m_kll_config({config.kll_k})
    {
        _initialize_seeds();
        _initialize_buckets();
        _initialize_rings();
    }

    ReSketch(uint32_t depth, uint32_t width, const std::vector<uint32_t> &seeds, uint32_t kll_k) : m_width(width), m_depth(depth), m_seeds(seeds), m_kll_config({kll_k})
    {
        m_config = {m_width, m_depth, kll_k};
        _initialize_buckets();
        _initialize_rings();
    }

    void update(uint64_t item) override
    {
        for (uint32_t i = 0; i < m_depth; ++i)
        {
            uint64_t h = _hash(item, m_seeds[i]);
            uint32_t id = _find_bucket_id(h, m_rings[i]);
            m_buckets[i][id].count++;
            m_buckets[i][id].q_sketch.update(h);
        }
    }

    double estimate(uint64_t item) const override
    {
        double total_kll_est = 0.0;
        for (uint32_t i = 0; i < m_depth; ++i)
        {
            uint64_t h = _hash(item, m_seeds[i]);
            uint32_t id = _find_bucket_id(h, m_rings[i]);
            total_kll_est += m_buckets[i][id].q_sketch.estimate(h);
        }
        return total_kll_est / static_cast<double>(m_depth);
    }

    // --- Dynamic Operations ---

    void expand(uint32_t new_width)
    {
        if (new_width <= m_width) throw std::invalid_argument("New width must be larger than current width.");

        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> dist;

        for (uint32_t i = 0; i < m_depth; ++i)
        {
            Ring new_ring = m_rings[i];
            for (uint32_t j = 0; j < new_width - m_width; ++j) { new_ring.push_back({dist(rng), m_width + j}); }
            std::sort(new_ring.begin(), new_ring.end());

            std::vector<Bucket> new_buckets = _remap_row(m_rings[i], m_buckets[i], new_ring);
            m_rings[i] = new_ring;
            m_buckets[i] = std::move(new_buckets);
        }
        m_width = new_width;
    }

    void shrink(uint32_t new_width)
    {
        if (new_width >= m_width) throw std::invalid_argument("New width must be smaller than current width.");

        std::mt19937_64 rng(std::random_device{}());

        for (uint32_t i = 0; i < m_depth; ++i)
        {
            Ring new_ring = m_rings[i];
            std::shuffle(new_ring.begin(), new_ring.end(), rng);
            new_ring.resize(new_width);

            //  reindex the bucket_id of the new ring
            std::sort(
                new_ring.begin(), new_ring.end(),
                [](const auto &a, const auto &b)
                {
                    return a.second < b.second;
                });

            for (uint32_t j = 0; j < new_ring.size(); ++j)
            {
                new_ring[j].second = j;   // Reassign bucket IDs
            }

            std::sort(new_ring.begin(), new_ring.end());

            std::vector<Bucket> new_buckets = _remap_row(m_rings[i], m_buckets[i], new_ring);
            m_rings[i] = new_ring;
            m_buckets[i] = std::move(new_buckets);
        }
        m_width = new_width;
    }

    uint32_t get_max_memory_usage() const
    {
        // uint32_t buckets_grid_memory = m_depth * sizeof(std::vector<Bucket>);

        // uint32_t rings_memory = m_depth * sizeof(Ring);

        KLL sample_kll(m_kll_config);
        uint32_t single_kll_max_memory = sample_kll.get_max_memory_usage();

        return single_kll_max_memory * m_depth * m_width;
    }

    static ReSketch merge(const ReSketch &s1, const ReSketch &s2)
    {
        if (s1.m_depth != s2.m_depth || s1.m_kll_config.k != s2.m_kll_config.k) { throw std::invalid_argument("Sketches must have same depth and kll_k to merge."); }

        uint32_t new_width = s1.m_width + s2.m_width;
        ReSketch merged_sketch(s1.m_depth, new_width, s1.m_seeds, s1.m_kll_config.k);

        for (uint32_t i = 0; i < s1.m_depth; ++i)
        {
            auto temp_buckets_1 = _remap_row(s1.m_rings[i], s1.m_buckets[i], merged_sketch.m_rings[i]);
            auto temp_buckets_2 = _remap_row(s2.m_rings[i], s2.m_buckets[i], merged_sketch.m_rings[i]);

            for (uint32_t j = 0; j < new_width; ++j)
            {
                merged_sketch.m_buckets[i][j].count = temp_buckets_1[j].count + temp_buckets_2[j].count;
                merged_sketch.m_buckets[i][j].q_sketch = std::move(temp_buckets_1[j].q_sketch);
                merged_sketch.m_buckets[i][j].q_sketch.merge(temp_buckets_2[j].q_sketch);
            }
        }
        return merged_sketch;
    }

    static std::pair<ReSketch, ReSketch> split(const ReSketch &sketch, uint32_t width_1, uint32_t width_2)
    {
        if (width_1 + width_2 != sketch.m_width) { throw std::invalid_argument("Split widths must sum to original width."); }

        ReSketch s1(sketch.m_depth, width_1, sketch.m_seeds, sketch.m_kll_config.k);
        ReSketch s2(sketch.m_depth, width_2, sketch.m_seeds, sketch.m_kll_config.k);

        for (uint32_t i = 0; i < sketch.m_depth; ++i)
        {
            s1.m_rings[i].assign(sketch.m_rings[i].begin(), sketch.m_rings[i].begin() + width_1);
            s1.m_buckets[i].assign(sketch.m_buckets[i].begin(), sketch.m_buckets[i].begin() + width_1);

            s2.m_rings[i].assign(sketch.m_rings[i].begin() + width_1, sketch.m_rings[i].end());
            s2.m_buckets[i].assign(sketch.m_buckets[i].begin() + width_1, sketch.m_buckets[i].end());
        }
        return {std::move(s1), std::move(s2)};
    }

private:
    void _initialize_seeds()
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint32_t> dist;
        m_seeds.reserve(m_depth);
        for (uint32_t i = 0; i < m_depth; ++i) { m_seeds.push_back(dist(rng)); }
    }

    void _initialize_buckets()
    {
        m_buckets.resize(m_depth);
        for (uint32_t i = 0; i < m_depth; ++i)
        {
            m_buckets[i].reserve(m_width);
            for (uint32_t j = 0; j < m_width; ++j) { m_buckets[i].emplace_back(m_kll_config); }
        }
    }

    void _initialize_rings()
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> dist;
        m_rings.resize(m_depth);
        for (uint32_t i = 0; i < m_depth; ++i)
        {
            m_rings[i].reserve(m_width);
            for (uint32_t j = 0; j < m_width; ++j) { m_rings[i].push_back({dist(rng), j}); }
            std::sort(m_rings[i].begin(), m_rings[i].end());
        }
    }

    // uint64_t _hash(uint64_t item, uint32_t seed) const { return std::hash<uint64_t>{}(item) ^ (static_cast<uint64_t>(seed) << 1); }

    uint64_t _full_domain_hash(uint64_t x) const
    {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    // uint64_t _hash(uint64_t item, uint32_t seed) const { return _full_domain_hash(item) ^ _full_domain_hash(seed); }

    // Using XXHash for better performance
    uint64_t _hash(uint64_t item, uint32_t seed) const { return XXHash64::hash(&item, sizeof(uint64_t), seed); }

    static uint32_t _find_bucket_id(uint64_t item_hash, const Ring &ring)
    {
        auto it = std::lower_bound(ring.begin(), ring.end(), std::make_pair(item_hash, std::numeric_limits<uint32_t>::max()));
        if (it == ring.end() || ring.empty())
        {
            return ring.empty() ? 0 : ring.front().second;   // Wrap around
        }
        return it->second;
    }

    static std::vector<Bucket> _remap_row(const Ring &in_ring, const std::vector<Bucket> &in_buckets, const Ring &out_ring)
    {
        std::vector<Bucket> out_buckets;
        if (in_buckets.empty())
        {
            out_buckets.resize(out_ring.size());
            return out_buckets;
        }

        const auto &kll_config = in_buckets[0].q_sketch.get_config();
        for (uint32_t i = 0; i < out_ring.size(); ++i) { out_buckets.emplace_back(kll_config); }

        std::set<uint64_t> point_set;
        for (const auto &p : in_ring) point_set.insert(p.first);
        for (const auto &p : out_ring) point_set.insert(p.first);

        std::vector<uint64_t> all_points(point_set.begin(), point_set.end());
        if (all_points.empty()) return out_buckets;

        uint64_t prev_p = all_points.back();
        for (const auto &current_p : all_points)
        {
            uint64_t start_p = prev_p;
            uint64_t end_p = current_p;

            uint32_t in_id = _find_bucket_id(start_p, in_ring);
            uint32_t out_id = _find_bucket_id(start_p, out_ring);

            // cout in_id, in_ring, out_id, out_ring, start_p, end_p << endl;
            // std::cout << "Processing range [" << start_p << ", " << end_p << "] for in_id: " << in_id << ", out_id: " << out_id << std::endl;
            // std::cout << "In Ring: ";
            // for (const auto &p : in_ring) std::cout << "{" << p.first << ", " << p.second << "} ";
            // std::cout << "\nOut Ring: ";
            // for (const auto &p : out_ring) std::cout << "{" << p.first << ", " << p.second << "} ";
            // std::cout << std::endl;

            const auto &in_bucket = in_buckets[in_id];
            double count = in_bucket.q_sketch.get_count_in_range(start_p, end_p);

            // cout k of in bucket and out bucket
            // std::cout << "In Bucket k: " << in_bucket.q_sketch.get_config().k << ", Out Bucket k: " << out_buckets[out_id].q_sketch.get_config().k << std::endl;

            if (count > 0)
            {
                out_buckets[out_id].count += static_cast<uint64_t>(std::round(count));
                auto sub_sketch = in_bucket.q_sketch.rebuild(start_p, end_p);
                out_buckets[out_id].q_sketch.merge(sub_sketch);
            }
            prev_p = current_p;
        }
        return out_buckets;
    }

    ReSketchConfig m_config;
    uint32_t m_width;
    uint32_t m_depth;
    std::vector<uint32_t> m_seeds;
    KLLConfig m_kll_config;

    std::vector<Ring> m_rings;
    std::vector<std::vector<Bucket>> m_buckets;
};
