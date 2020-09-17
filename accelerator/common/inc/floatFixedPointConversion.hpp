#ifndef FLOAT_FIXED_POINT_CONVERSION_HPP
#define FLOAT_FIXED_POINT_CONVERSION_HPP

class fixedPointNumber {
    private:
        signed char bits;
        signed char fractionWidth;
        signed char integerWidth;
        float resolution;
    public:
        fixedPointNumber () = default;
        fixedPointNumber (float _realNumber
                          ,signed char _fracWidth
                          ,signed char _intWidth);
        fixedPointNumber (signed char _bits,
                          signed char _fracWidth,
                          signed char _intWidth);

        signed char getBits ();
        unsigned char getMask ();
        signed char getFracWidth ();
        signed char getIntWidth ();

        float convert2Float();
};

#endif
