#ifndef FLOAT_FIXED_POINT_CONVERSION_HPP
#define FLOAT_FIXED_POINT_CONVERSION_HPP

class fixedPointNumber {
    private:
        char bits;
        char fractionWidth;
        char integerWidth;
        float resolution;
    public:
        fixedPointNumber () = default;
        fixedPointNumber (float _realNumber
                          ,char _fracWidth
                          ,char _intWidth);
        fixedPointNumber (signed char _bits,
                          char _fracWidth,
                          char _intWidth);

        signed char getBits ();
        unsigned char getMask ();
        int getFracWidth ();
        int getIntWidth ();

        float convert2Float();
};

#endif
