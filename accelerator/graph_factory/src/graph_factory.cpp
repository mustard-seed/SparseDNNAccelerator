#include "graph_factory.hpp"

namespace GraphRuntime {
    GraphFactory::GraphFactory(std::string _traceFileName, std::string _parameterFileName)
    {
        //TODO implement this
    }

    std::unique_ptr<t_execution_graph> GraphFactory::generateGraph()
    {
        auto pGraph = std::unique_ptr<t_execution_graph>(new GraphRuntime::t_execution_graph);

        return pGraph;
    }
}

