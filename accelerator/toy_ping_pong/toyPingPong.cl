
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
	char loadFlag = 0x0;
	unsigned char prevNumInputToLoad = 0;

	//Important to go one extra
	for (int iterInput=0; iterInput < numInputs + (CACHE_SIZE - 2); iterInput += (CACHE_SIZE - 2) ) {
		short numInputToLoad = ((iterInput + CACHE_SIZE) < numInputs) ? 
			CACHE_SIZE : (numInputs - iterInput);
		short iterCacheLoad=0;
		short iterCacheRead = 0;
		bool proceed = true;
		bool isFirst = (iterInput == 0) ? true : false;
		bool isLast = (iterInput >= numInputs) ? true : false;
		#pragma ivdep array(cache)
		while (proceed) {
			bool loadProceed = false;
			bool readProceed = false;
			if ( (iterCacheLoad < numInputToLoad) && (!isLast)) {
				cache[iterCacheLoad][loadFlag & 0x01] = input[iterInput + iterCacheLoad];
				iterCacheLoad++;
				loadProceed = true;
			}

			if ((iterCacheRead < (prevNumInputToLoad - 2)) && (!isFirst)) {
				float result0 = w0 * cache[iterCacheRead][(~loadFlag) & 0x01];
				float result1 = w1 * cache[iterCacheRead+1][(~loadFlag) & 0x01]
				+ w2 * cache[iterCacheRead+2][(~loadFlag) & 0x01];
			
				/*
				float result0 = w0 * cache[iterCache];
				float result1 = w1 * cache[iterCache+1]
					+ w2 * cache[iterCache+2];
				*/
				output[iterOutput++] = result0+result1;
				iterCacheRead++;
				readProceed = true;
			}
			//else {
			//	isFirst = false;
			//}

			proceed = loadProceed || readProceed;
		}
		loadFlag = ~loadFlag;
		isFirst = false;
		prevNumInputToLoad = numInputToLoad;
	}
	

}