#pragma once
#include "model_container.hpp"
#include "accelerator_wrapper.hpp"

#include "yaml-cpp/yaml.h"
#include <vector>
#include <memory>


namespace GraphRuntime {
    class GraphFactory {
    private:
        std::vector<std::shared_ptr<Layer>> vecLayers;
        bool flagInputScatter;
    public:
       GraphFactory() = default;
       GraphFactory(std::string _traceFileName, std::string _parameterFileName, bool _inputScatter=false);
       ~GraphFactory() = default;

       std::unique_ptr<t_execution_graph> generateGraph();
    };
}
