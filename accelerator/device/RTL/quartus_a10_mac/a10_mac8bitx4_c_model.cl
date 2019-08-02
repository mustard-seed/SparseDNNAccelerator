#include "rtl_lib.hpp"

int a10_mac_8bitx4 (char a0, char b0, char a1, char b1, char a2, char b2, char a3, char b3) {
    return (a0*b0 + a1*b1 + a2*b2 + a3*b3);
}
