#pragma once
#include "model_container.hpp"
#include "accelerator_wrapper.hpp"
#include "tile.hpp"

#include "yaml-cpp/yaml.h"
#include <vector>
#include <memory>

namespace GraphRuntime {
    typedef struct {
        t_graph_output_tile_info tileInfo;
        t_latency_info latencyInfo;
        unsigned int ops;
    } t_tile_pair;

    class GraphFactory {
    private:
        std::vector<std::shared_ptr<Layer>> vecLayers;
        std::vector<bool> vecFlagManualTile;
        std::vector<t_graph_output_tile_info> vecTileInfo;
        bool flagInputScatter;
    public:
       GraphFactory() = default;
       GraphFactory(std::string _traceFileName, std::string _parameterFileName, bool _inputScatter=false);
       ~GraphFactory() = default;

       /*!
        * \brief addLayer
        * \param _pLayer
        * \param _pTileInfo
        * \return 0 if successful
        */
       void setInputScatter(bool _flag);
       int addLayer(std::shared_ptr<Layer> _pLayer, t_graph_output_tile_info* _pTileInfo = NULL);
       std::unique_ptr<t_execution_graph> generateGraph();

    };
}
