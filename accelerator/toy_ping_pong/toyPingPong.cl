
#define CACHE_SIZE 128
__kernel void toyPingPongConv (
	__global float* restrict input,
	__global float* restrict output,
	int numInputs,
	float w0,
	float w1,
	float w2
	) {
	float cache[CACHE_SIZE][2] __attribute__ ((numbanks(2), bankwidth(4)));
	//float cache[CACHE_SIZE];

	//Computer some parameters
	int iterOutput = 0;
	bool isFirst = true;
	char loadFlag = 0x0;
	int prevNumInputToLoad;

	#pragma ivdep array(cache)
	for (int iterInput=0; iterInput < numInputs; iterInput += (CACHE_SIZE - 2) ){
		//Load the cache
		int numInputToLoad = ((iterInput + CACHE_SIZE) < numInputs) ? 
			CACHE_SIZE : (numInputs - iterInput);
		for (int iterCache=0; iterCache < numInputToLoad; iterCache++) {
			cache[iterCache][loadFlag & 0x01] = input[iterInput + iterCache];
			//cache[iterCache] = input[iterInput + iterCache];
		}

		//if (!isFirst) {
			for (int iterCache=0; iterCache < (prevNumInputToLoad - 2); iterCache++)
			{
				
				float result0 = w0 * cache[iterCache][(~loadFlag) & 0x01];
				float result1 = w1 * cache[iterCache+1][(~loadFlag) & 0x01]
					+ w2 * cache[iterCache+2][(~loadFlag) & 0x01];
				
				/*
				float result0 = w0 * cache[iterCache];
				float result1 = w1 * cache[iterCache+1]
					+ w2 * cache[iterCache+2];
				*/
				output[iterOutput++] = result0+result1;
			}
		//}
		//else {
		//	isFirst = false;
		//}

		prevNumInputToLoad = numInputToLoad;
		loadFlag = ~loadFlag;
	}

}