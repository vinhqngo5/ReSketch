#pragma once

#include "utils/ConfigParser.hpp"
#include "utils/ConfigPrinter.hpp"

#include <string>
#include <tuple>

struct CountMinConfig
{
    uint32_t width;
    uint32_t depth;
    float epsilon;
    float delta;
    std::string calculate_from;
    static void add_params_to_config_parser(CountMinConfig &c, ConfigParser &p)
    {
        p.AddParameter(new UnsignedInt32Parameter("countmin.width", "1024", &c.width, false, "Width of CM sketch"));
        p.AddParameter(new UnsignedInt32Parameter("countmin.depth", "8", &c.depth, false, "Depth of CM sketch"));
        p.AddParameter(new FloatParameter("countmin.epsilon", "0.01", &c.epsilon, false, "Epsilon for CM sketch"));
        p.AddParameter(new FloatParameter("countmin.delta", "0.01", &c.delta, false, "Delta for CM sketch"));
        p.AddParameter(new StringParameter("countmin.calculate_from", "WIDTH_DEPTH", &c.calculate_from, false, "Calculate from WIDTH_DEPTH or EPSILON_DELTA"));
    }
    auto to_tuple() const { return std::make_tuple("width", width, "depth", depth, "epsilon", epsilon, "delta", delta, "calculate_from", calculate_from); }
    friend std::ostream &operator<<(std::ostream &os, const CountMinConfig &c)
    {
        ConfigPrinter<CountMinConfig>::print(os, c);
        return os;
    }
};

struct ReSketchConfig
{
    uint32_t width;
    uint32_t depth;
    uint32_t kll_k;
    static void add_params_to_config_parser(ReSketchConfig &c, ConfigParser &p)
    {
        p.AddParameter(new UnsignedInt32Parameter("resketch.width", "64", &c.width, false, "Initial width of ReSketch"));
        p.AddParameter(new UnsignedInt32Parameter("resketch.depth", "4", &c.depth, false, "Depth of ReSketch"));
        p.AddParameter(new UnsignedInt32Parameter("resketch.kll_k", "10", &c.kll_k, false, "K for inner KLL sketches"));
    }
    auto to_tuple() const { return std::make_tuple("width", width, "depth", depth, "kll_k", kll_k); }
    friend std::ostream &operator<<(std::ostream &os, const ReSketchConfig &c)
    {
        ConfigPrinter<ReSketchConfig>::print(os, c);
        return os;
    }
};

struct GeometricSketchConfig
{
    uint32_t width;
    uint32_t depth;
    uint32_t branching_factor;
    static void add_params_to_config_parser(GeometricSketchConfig &c, ConfigParser &p)
    {
        p.AddParameter(new UnsignedInt32Parameter("geometric.width", "1024", &c.width, false, "Width of Geometric Sketch"));
        p.AddParameter(new UnsignedInt32Parameter("geometric.depth", "8", &c.depth, false, "Depth of Geometric Sketch"));
        p.AddParameter(new UnsignedInt32Parameter("geometric.branching_factor", "2", &c.branching_factor, false, "Branching factor of Geometric Sketch"));
    }
    auto to_tuple() const { return std::make_tuple("width", width, "depth", depth, "branching_factor", branching_factor); }
    friend std::ostream &operator<<(std::ostream &os, const GeometricSketchConfig &c)
    {
        ConfigPrinter<GeometricSketchConfig>::print(os, c);
        return os;
    }
};

struct DynamicSketchConfig
{
    uint32_t width;
    uint32_t depth;
    bool is_same_seed;
    static void add_params_to_config_parser(DynamicSketchConfig &c, ConfigParser &p)
    {
        p.AddParameter(new UnsignedInt32Parameter("dynamic.width", "1024", &c.width, false, "Width of Dynamic Sketch"));
        p.AddParameter(new UnsignedInt32Parameter("dynamic.depth", "8", &c.depth, false, "Depth of Dynamic Sketch"));
        p.AddParameter(new BooleanParameter("dynamic.is_same_seed", "false", &c.is_same_seed, false, "Use same seed for all layers in Dynamic Sketch"));
    }
    auto to_tuple() const { return std::make_tuple("width", width, "depth", depth, "is_same_seed", is_same_seed); }
    friend std::ostream &operator<<(std::ostream &os, const DynamicSketchConfig &c)
    {
        ConfigPrinter<DynamicSketchConfig>::print(os, c);
        return os;
    }
};
