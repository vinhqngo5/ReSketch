#pragma once

#include "utils/ConfigParser.hpp"
#include "utils/ConfigPrinter.hpp"

#include <string>
#include <tuple>

struct KLLConfig
{
    uint32_t k;

    static void add_params_to_config_parser(KLLConfig &kll_config, ConfigParser &parser)
    { parser.AddParameter(new UnsignedInt32Parameter("kll.k", "2730", &kll_config.k, false, "K parameter for KLL sketch, controlling size and accuracy")); }

    auto to_tuple() const { return std::make_tuple("k", k); }

    friend std::ostream &operator<<(std::ostream &os, const KLLConfig &config)
    {
        ConfigPrinter<KLLConfig>::print(os, config);
        return os;
    }
};
