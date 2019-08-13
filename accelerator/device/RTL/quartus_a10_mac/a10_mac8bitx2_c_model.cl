#include "rtl_lib.hpp"
int a10_mac_8bitx2 (char a0, char b0, char a1, char b1) {
    return ( ((signed int) a0)* ((signed int) b0) + ((signed int) a1)* ((signed int) b1));
}
