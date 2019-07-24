unsigned char leadingZeroCounter (unsigned char bitmask) {
    unsigned char count = 0;
    unsigned char countPlus1 = 0;
    while ( ( (bitmask & 0x01 )== 0) && count < 8) {
        count++;
        bitmask = bitmask >> 0x1;
    }
    countPlus1 = count + 1;
    unsigned char result = ( (countPlus1 & 0x0F) << 4 ) | (count & 0x0F);

    return result;
}
