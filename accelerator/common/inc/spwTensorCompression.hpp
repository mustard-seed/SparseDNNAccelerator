#ifndef SPWTENSORCOMPRESSION_HPP
#define SPWTENSORCOMPRESSION_HPP
#include <vector>

#include "floatFixedPointConversion.hpp"
#include "vectorType.hpp"

class DeviceActivationTensor {
    /*!
      Protected values are seen by derived classes.
    */
    protected:
    //Vector holding the value
    //Address of the vector on the host is DMA aligned
    t_aligned_activation_vector valueVector;

    //Dimensions of the plain tensor (not padded)
    int channel;
    int width;
    int height;

    int stripStrideInExternalMemory;

    bool inputScatter;

public:
    DeviceActivationTensor () = delete;

    //Constructor used to allocate a device activation tensor
    /*!
     * \brief DeviceActivationTensor
     * \details Allocates a device activation tensor
     * \param _channel Number of channels
     * \param _width Tensor width
     * \param _height Tensor height
     * \param _stripStrideInExternalMemory
     */
    DeviceActivationTensor (
            int _channel,
            int _width,
            int _height,
            int _stripStrideInExternalMemory
            );

    //Constructor used to initialize a device activation tensor with known values
    /*!
     * \brief DeviceActivationTensor
     * \details Initializes a device activation tensor using the known values
     * from a fixed point tensor
     * \param _vecFixedPoint
     * \param _channel
     * \param _width
     * \param _height
     * \param _stripStrideInExternalMemory
     */
    DeviceActivationTensor (
            std::vector<fixedPointNumber> & _vecFixedPoint,
            int _channel,
            int _width,
            int _height,
            int _stripStrideInExternalMemory
            );

    //Accessors
    t_aligned_activation_vector & getValueVector();
    int getChannel() const;
    int getWidth() const;
    int getHeight() const;

    /*!
     * \brief getStripStride
     * \return Address stride in terms of activation values
     * when we move from one strip location to another
     */
    int getStripStride() const;

    /*!
     * \brief decodeTensor
     * \details Transform the device activation tensor into a host activation tensor
     * \param _fracWidth
     * \param _intWidth
     * \return
     */
    std::vector<fixedPointNumber> decodeTensor(
                signed char _fracWidth,
                signed char _intWidth
            ) const;
}; //DeviceActivationTensor


class DeviceWeightTensor {
protected:
    //Vector holding the value
    //Address of the vector on the host is DMA aligned
    t_aligned_weight_vector valueVector;

    //Dimensions of weights
    int outputChannel;
    int inputChannel;
    int width;
    int height;

    //Input channel size after padding is applied;
    int paddedInputChannel;
    //Number of weights in a padded filter
    int numWeightsInPaddedFilter;
    //Number of transfer block per strip
    int numTBPerStrip;
    //Number of dram blocks in a filter
    int numDramBlocksPerFilter;

public:
    DeviceWeightTensor() = delete;

    //Allocator
    DeviceWeightTensor(
            int _outputChannel,
            int _inputChannel,
            int _width,
            int _height,
            int _peSimdSize,
            int _clusterSize
            );

    //Initializer
    DeviceWeightTensor(
            std::vector<fixedPointNumber> & _vecFixedPoint,
            int _outputChannel,
            int _inputChannel,
            int _width,
            int _height,
            int _peSimdSize,
            int _clusterSize
            );


    t_aligned_weight_vector & getValueVector();

    virtual std::vector<fixedPointNumber> decodeTensor(
                signed char _fracWidth,
                signed char _intWidth
            );

    int getOutputChannel() const;
    int getInputChannel() const;
    int getWidth() const;
    int getHeight() const;
    int getPaddedInputChannel() const;
    int getNumWeightsInPaddedFilter() const;
    int getTBPerStrip() const;
    int getDramBlocksInFilter() const;

    /*!
     * \brief getStripStride
     * \return Strip stride in weight word (e.g. signed char)
     */
    int getStripStride() const;
    /*!
     * \brief getFilterStride
     * \return Filter stride in weight word (e.g. signed char)
     */
    int getFilterStride() const;
}; //DeviceWeightTensor

#if defined(SPW_SYSTEM)
class DeviceSpWTensor : public DeviceWeightTensor {
  protected:
    int numClustersInPruningRange;
    int numNZClustersInPruningRange;
    int peSimdSize;
    int clusterSize;

   public:
    DeviceSpWTensor() = delete;
    //Allocator
    DeviceSpWTensor(
            int _outputChannel,
            int _inputChannel,
            int _width,
            int _height,
            int _peSimdSize,
            int _clusterSize,
            int _numClustersInPruningRange,
            int _numNZClustersPerPruningRange
            );
    //Initializer
    DeviceSpWTensor(
            std::vector<fixedPointNumber> & _vecFixedPoint,
            int _outputChannel,
            int _inputChannel,
            int _width,
            int _height,
            int _peSimdSize,
            int _clusterSize,
            int _numClustersInPruningRange,
            int _numNZClustersPerPruningRange
            );

    std::vector<fixedPointNumber> decodeTensor(
                signed char _fracWidth,
                signed char _intWidth
            ) override;
}; //DeviceSpWTensor
#endif //SPW_TEST

#endif // SPWTENSORCOMPRESSION_HPP
