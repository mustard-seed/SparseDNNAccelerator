#ifndef FLOAT_FIXED_POINT_CONVERSION_HPP
#define FLOAT_FIXED_POINT_CONVERSION_HPP

class fixedPointNumber {
    private:
        int bits;
        int fractionWidth;
        int integerWidth;
        float resolution;
    public:
        fixedPointNumber () = delete;
        fixedPointNumber (float _realNumber
                          ,int _fracWidth
                          ,int _intWidth);
        fixedPointNumber (int _bits,
                          int _fracWidth,
                          int _intWidth);

        int getBits ();
        int getMask ();
        int getFracWidth ();
        int getIntWidth ();

        float convert2Float();
};

#endif
