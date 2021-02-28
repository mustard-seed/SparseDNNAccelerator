#include "floatFixedPointConversion.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cfenv> //For rounding modes

fixedPointNumber::fixedPointNumber (float _realNumber
                                    ,signed char _fracWidth
                                    ,signed char _intWidth){
    //Make sure the number of bits for magnitude and the sign can fit within 32
    assert (_fracWidth + _intWidth < 8);

    //Find the precision
    if (_fracWidth > 0)
    {
        resolution = 1.0f / (float) (1 << _fracWidth);
    }
    else
    {
        resolution = 1 << ((-1) * (_fracWidth));
    }
    //int fullBits = (int) round(_realNumber / resolution);
    std::fesetround(FE_TONEAREST); //round to even
    int fullBits = (int) std::nearbyint(_realNumber / resolution);
//    int fullBits = (int) std::round(_realNumber / resolution);
    int totalWidth = _fracWidth + _intWidth;
    int minimum = -1 * (1 << totalWidth);
    int maximum = (1 << totalWidth) - 1;

    fractionWidth = _fracWidth;
    integerWidth = _intWidth;

    //Clip
    bits = std::max(
                std::min (maximum, fullBits),
                minimum
                );

    //Preserve the magnitude and the sign bit
    unsigned char bitMask = ~ (0xFF << (totalWidth + 1));
    bits = bitMask & bits;
}

fixedPointNumber::fixedPointNumber (signed char _bits,
                                    signed char _fracWidth,
                                    signed char _intWidth)
{
    assert (_fracWidth + _intWidth < 8);
    bits = _bits & (~ (0xFF << (_fracWidth + _intWidth + 1)) );
    fractionWidth = _fracWidth;
    integerWidth = _intWidth;
    if (_fracWidth > 0)
    {
        resolution = 1.0f / (float) (1 << _fracWidth);
    }
    else
    {
        resolution = 1 << ((-1) * (_fracWidth));
    }
}

signed char fixedPointNumber::getBits() {
    return bits;
}

unsigned char fixedPointNumber::getMask() {
    return ~(0xFF << (fractionWidth + integerWidth + 1));
}

signed char fixedPointNumber::getFracWidth() {
    return fractionWidth;
}


signed char fixedPointNumber::getIntWidth() {
    return integerWidth;
}

float fixedPointNumber::convert2Float () {
    //Need to perform sign extend
    //char signBit = 0x1 & (bits >> (fractionWidth + integerWidth));
    //char fullBits =
    //        signBit > 0 ?
    //        bits | 0xFFFF << (fractionWidth + integerWidth) :
    //        bits;
    return (float) ((signed char) (bits & 0xFF)) * resolution;
}
