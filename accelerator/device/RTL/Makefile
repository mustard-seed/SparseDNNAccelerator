# Remember to recompile the lib everytime you changed the rtl

lib:
	aoc -c quartus_a10_mac/a10_mac_8bitx4.xml -I . -I ../. -o a10_mac_8bitx4.aoco
	aoc -c quartus_a10_mac/a10_mac_8bitx2.xml -I . -I ../. -o a10_mac_8bitx2.aoco
	aoc -c quartus_c5_mac/c5_mac_8bitx4.xml -I . -I ../. -o c5_mac_8bitx4.aoco
	aoc -c quartus_c5_mac/c5_mac_8bitx2.xml -I . -I ../. -o c5_mac_8bitx2.aoco
	aoc -c quartus_a10_chain_madd/a10_chain_madd_8bit.xml -I . -I ../. -o a10_chain_madd_8bit.aoco
# 	aoc -c quartus_operandMatcher/operandMatcher.xml -I . -I ../. -o operandMatcher.aoco
# 	aoc -c quartus_leadingZeroCounter/leadingZeroCounter.xml -I . -I ../. -o leadingZeroCounter.aoco
# 	aoc -c quartus_smallBuffer_lib/smallBufferLib.xml -I . -I ../. -o smallBuffer.aoco
# 	aocl library create -name rtl_lib a10_mac_8bitx4.aoco a10_mac_8bitx2.aoco c5_mac_8bitx4.aoco c5_mac_8bitx2.aoco operandMatcher.aoco smallBuffer.aoco
	aocl library create -name rtl_lib a10_mac_8bitx4.aoco a10_mac_8bitx2.aoco c5_mac_8bitx4.aoco c5_mac_8bitx2.aoco a10_chain_madd_8bit.aoco

clean:
	rm -rf *.aoco *.aoclib
