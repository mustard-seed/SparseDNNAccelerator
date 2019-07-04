#ifndef FLOAT_FIXED_POINT_CONVERSION_HPP
#define FLOAT_FIXED_POINT_CONVERSION_HPP

class fixedPointNumber {
    private:
        short bits;
        char fractionWidth;
        char integerWidth;
        float resolution;
    public:
        fixedPointNumber () = delete;
        fixedPointNumber (float _realNumber
                          ,char _fracWidth
                          ,char _intWidth);
        fixedPointNumber (short _bits,
                          char _fracWidth,
                          char _intWidth);

        short getBits ();
        short getMask ();
        int getFracWidth ();
        int getIntWidth ();

        float convert2Float();
};

#endif
