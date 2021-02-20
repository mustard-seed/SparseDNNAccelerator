/*!
  Header for the model container class
*/
#pragma once
#include <vector>
#include <cassert>

#include "tile.hpp"

#include "yaml-cpp/yaml.h"
#include "cnpy.hpp"

namespace GraphRuntime {
    enum LayerType {CONVOLUTION, ELTADD, MAXPOOL, AVGPOOL, QUANT, DEQUANT};
    typedef std::vector<int> IntVec;
    typedef std::vector<bool> BoolVec;
    typedef std::vector<float> FloatVec;
    class Layer {
        protected:
           YAML::Node node;

        public:
            Layer() = default;
            Layer(const YAML::Node& _node);

            //Declare one method in the base class as virtue to achieve
            //polymorphic behaviour
            virtual ~Layer() = default;

            /*
             * Generic input information
            */
            int getLayerID();
            virtual LayerType getLayerType() = 0;
            void setLayerID(int _id);

            /*
             * Input information getters
             *
            */
            IntVec getInputHeights();
            IntVec getInputWidths();
            IntVec getInputChannels();
            IntVec getInputFracBits();
            IntVec getInputMemoryLocations();
            IntVec getInputGroupsSeenBySource();
            bool   getInputSparseFlag();

            /*!
              Input information setters
            */
            void setInputHeights(IntVec _inputHeights);
            void setInputWidths(IntVec _inputWidths);
            void setInputChannels(IntVec _inputChannels);
            void setInputFracBits(IntVec _inputFracBits);
            void setInputMemoryLocations(IntVec _inputMemLocations);
            void setInputGroupsSeenBySource(IntVec _inputGroupsSeenBySource);
            void setInputSparseFlag(bool _inputSparseFlag);

            /*
             * Output information getters
            */
            int getOutputFracBits();
            int getOutputHeight();
            int getOutputWidth();
            int getOutputChannel();
            int getOutputMemoryLocation();
            bool getOutputReluFlag();
            bool getOutputSparseFlag();

            /*!
              Output information setters
             */
            void setOutputFracBits(int);
            void setOutputHeight(int);
            void setOutputWidth(int);
            void setOutputChannel(int);
            void setOutputMemoryLocation(int);
            void setOutputReluFlag(bool);
            void setOutputSparseFlag(bool);

            /*
             * Group information getters and setters
            */
            int getCurrentNumberGroups();
            int getNextNumberGroups();

            void setCurrentNumberGroups(int);
            void setNextNumberGroups(int);

            /*!
              Tile helper functions
             */
            virtual bool cacheBoundaryCheck(t_graph_output_tile_info _tileCandidate);
            virtual t_latency_info deriveLatency(t_graph_output_tile_info _tileCandidate);
            virtual int deriveOps();
    };

    class ConvLayer: public Layer {
      private:
        FloatVec    vecWeights;
        FloatVec    vecBiases;
      public:
        ConvLayer() = default;
        ConvLayer(const YAML::Node& _node, const cnpy::NpyArray& _weightNode, const cnpy::NpyArray& _biasNode);

        LayerType getLayerType() override;

        /*
         * Conv kernel related information
        */
        int getKernelStride();
        int getKernelSize();
        int getInputBorderPadding();
        int getTransConvPadding();
        bool getBiasFlag();
        int getWeightFracBits();
        float getWeightSparsity();
        int getWeightPruneClusterSize();
        int getWeightPruneRangeSizeInCluster();

        void setKernelStride(int);
        void setKernelSize(int);
        void setInputBorderPadding(int);
        void setTransConvPadding(int);
        void setBiasFlag(bool);
        void setWeightFracBits(int);
        void setWeightSparsity(float);
        void setWeightPruneClusterSize(int);
        void setWeightPruneRangeSizeInCluster(int);

        /*
         * Parameter Related Flag
        */
        void    loadWeights(const cnpy::NpyArray& _weightNode);
        void    loadBiases(const cnpy::NpyArray&  _biasNode);
        void    loadWeights(const float* _pWeights);
        void    loadBiases (const float* _pBiases);
        FloatVec getWeights();
        FloatVec getBiases();

        /*
         * Special input processing
         */
        bool getIsAfterInput();
        void setIsAfterInput (bool);

        /*!
            Class specific implementations
        */
        bool cacheBoundaryCheck(t_graph_output_tile_info _tileCandidate) override;
        t_latency_info deriveLatency(t_graph_output_tile_info _tileCandidate) override;
        int deriveOps() override;

    };

    class MaxPoolLayer: public Layer {
       public:
        MaxPoolLayer() = default;
        MaxPoolLayer(const YAML::Node& _node);

        LayerType getLayerType() override;

        /*
         * Kernel related information
        */
        int getKernelStride();
        int getKernelSize();
        int getInputBorderPadding();

        void setKernelStride(int);
        void setKernelSize(int);
        void setInputBorderPadding(int);

        t_latency_info deriveLatency(t_graph_output_tile_info _tileCandidate) override;
        int deriveOps() override;
    };

    class AveragePoolLayer: public Layer {
       public:
        AveragePoolLayer();
        AveragePoolLayer(const YAML::Node& _node);

        LayerType getLayerType() override;

        /*
         * Kernel related information
        */
        int getKernelStride();
        int getKernelSize();
        int getInputBorderPadding();
        float getDivisor();

        void setKernelStride(int);
        void setKernelSize(int);
        void setInputBorderPadding(int);
        void setDivisor(float);

        t_latency_info deriveLatency(t_graph_output_tile_info _tileCandidate) override;
        int deriveOps() override;
    };

    class EltAddLayer: public Layer {
       public:
        EltAddLayer();
        EltAddLayer(const YAML::Node& _node);

        LayerType getLayerType() override;
        t_latency_info deriveLatency(t_graph_output_tile_info _tileCandidate) override;
        int deriveOps() override;
    };

    class QuantLayer: public Layer {
        public:
           QuantLayer() = default;
           QuantLayer(const YAML::Node& _node);

           LayerType getLayerType() override;
    };

    class DeQuantLayer: public Layer {
        public:
           DeQuantLayer() = default;
           DeQuantLayer(const YAML::Node& _node);

           LayerType getLayerType() override;
    };

    LayerType hashLayerTypeString(std::string stringValue);
}
