#ifndef FLOAT_FIXED_POINT_CONVERSION_HPP
#define FLOAT_FIXED_POINT_CONVERSION_HPP

class fixedPointNumber {
    private:
        int bits;
        char fractionWidth;
        char integerWidth;
        float resolution;
    public:
        fixedPointNumber () = delete;
        fixedPointNumber (float _realNumber
                          ,char _fracWidth
                          ,char _intWidth);
        fixedPointNumber (int _bits,
                          char _fracWidth,
                          char _intWidth);

        int getBits ();
        int getMask ();
        int getFracWidth ();
        int getIntWidth ();

        float convert2Float();
};

#endif
