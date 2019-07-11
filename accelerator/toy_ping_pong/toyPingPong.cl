
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
	unsigned char prevNumInputToLoad = 0;

	for (int iterInput=0; iterInput < numInputs; iterInput += (CACHE_SIZE - 2) ) {
		unsigned char numInputToLoad = ((iterInput + CACHE_SIZE) < numInputs) ? 
			CACHE_SIZE : (numInputs - iterInput);
		unsigned char iterCacheLoad=0;
		unsigned char iterCacheRead = 0;
		bool proceed = true;
		#pragma ivdep array(cache)
		while (proceed) {
			bool loadProceed = false;
			bool readProceed = false;
			if ( (unsigned char) iterCacheLoad < (unsigned char) numInputToLoad) {
				cache[iterCacheLoad][loadFlag & 0x01] = input[iterInput + iterCacheLoad];
				iterCacheLoad++;
				loadProceed = true;
			}

			if (!isFirst) {
				if (iterCacheRead < prevNumInputToLoad) {
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
			}
			else {
				isFirst = false;
			}

			proceed = loadProceed || readProceed;
			loadFlag = ~loadFlag;
		}
		prevNumInputToLoad = numInputToLoad;
	}

}