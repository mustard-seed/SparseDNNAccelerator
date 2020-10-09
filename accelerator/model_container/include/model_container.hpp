/*!
  Header for the model container class
*/
#pragma once
#include <vector>
#include <cassert>

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

            /*
             * Input information getters
             * TODO: Optional. In the tracer, regsiter input shape and frac bits information as vectors
             *
            */
            IntVec getInputHeights();
            IntVec getInputWidths();
            IntVec getInputChannels();
            IntVec getInputFracBits();
            IntVec getInputMemoryLocations();
            bool   getInputSparseFlag();

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

            /*
             * Group information
            */
            int getCurrentNumberGroups();
            int getNextNumberGroups();
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

        /*
         * Parameter Related Flag
        */
        void    loadWeights(const cnpy::NpyArray& _weightNode);
        void    loadBiases(const cnpy::NpyArray&  _biasNode);
        FloatVec getWeights();
        FloatVec getBiases();
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
    };

    class EltAddLayer: public Layer {
       public:
        EltAddLayer();
        EltAddLayer(const YAML::Node& _node);

        LayerType getLayerType() override;
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
