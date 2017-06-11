
#include <cuda_runtime_api.h>
#include <errno.h>
#include <stdint.h>

#include "host_loop.h"

#include "fs_initializer.cu.h"
#include "fs_calls.cu.h"
#include "fs_debug.cu.h"

#include "utils.h"
#include "loader.h"

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;

static const int MAX_CAND_LIST_SIZE = 256;
static const int BLOCK_SIZE = 32;
static const int NUM_TABLES = 32;
static const int ALL_ONES = -1;

static const int TLB_SIZE = 32;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

char* allocate_filename(const char* h_filename)
{
	int n = strlen(h_filename);
	assert(n > 0);
	if (n > FILENAME_SIZE)
	{
		fprintf(stderr, "Filname %s too long, should be only %d symbols including \\0", h_filename, FILENAME_SIZE);
		exit(-1);
	}
	char* d_filename;
	CUDA_SAFE_CALL(cudaMalloc(&d_filename, n + 1));
	CUDA_SAFE_CALL(cudaMemcpy(d_filename, h_filename, n + 1, cudaMemcpyHostToDevice));
	return d_filename;
}


void init_device_app()
{
	checkCudaErrors( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 25) );
}

struct Pixel
{
	uchar r;
	uchar g;
	uchar b;
	uchar a;
};

__global__ void mosaicKernel(const char* histFileName, uchar *inOut, float* coef, int** hashTables, int** chainTables, int** indexTables, const int width, const int D,const int K, const int L)
{
	__shared__ float  hist[3 * 256];

	Pixel* im = (Pixel*)inOut;

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int threadID = ty * 32 + tx;

	hist[0 * 256 + threadID] = 0.f;
	hist[1 * 256 + threadID] = 0.f;
	hist[2 * 256 + threadID] = 0.f;

	__syncthreads();

	int base = by * BLOCK_SIZE * width + bx * BLOCK_SIZE;

	Pixel myPix0 = im[base + (ty + 0 * 8) * width + tx];
	Pixel myPix1 = im[base + (ty + 1 * 8) * width + tx];
	Pixel myPix2 = im[base + (ty + 2 * 8) * width + tx];
	Pixel myPix3 = im[base + (ty + 3 * 8) * width + tx];

	atomicAdd( (float*)&hist[0 * 256 + myPix0.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix0.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix0.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix1.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix1.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix1.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix2.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix2.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix2.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix3.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix3.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix3.b ], 1.f );

	__syncthreads();

	float myBlockHist1 = hist[0 * 256 + threadID];
	float myBlockHist2 = hist[1 * 256 + threadID];
	float myBlockHist3 = hist[2 * 256 + threadID];

	__shared__ uint keys[32];

	for( int i = 0; i < 4; ++i )
	{
		uint bit = 0;

		if( tx < K )
		{
			float value = 0;

			for(int d = 0; d < D; ++d)
			{
				// We use 32 instead of K for alignment
				value += (hist[d * 4 + 0] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 1] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 2] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 3] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
			}

			bit = ( value > 0 ) ? 1 : 0;
		}

		keys[ty + i * 8] = __ballot( bit );
	}

	__shared__ volatile int candidates[256];
	__shared__ volatile int idx;
	__shared__ volatile int cand[32];
	__shared__ volatile bool found;

	candidates[threadID] = -1;
	idx = 0;

	__syncthreads();

	for( int l = 0; l < 32; ++l )
	{
		__shared__ int numCands;

		if( threadID == 0 )
		{
			bool foundBucket = false;
			numCands = 0;

			uint hIndex = keys[l] % 1000000;
			uint control = keys[l];

			int chainIndex = hashTables[l][hIndex];
			if( ALL_ONES != chainIndex )
			{
				while( true )
				{
					if( chainTables[l][chainIndex] == control )
					{
						foundBucket = true;
						break;
					}

					if( chainTables[l][chainIndex + 1] < 0 )
					{
						// If MSB is set, this means it's the last node in the chain
						break;
					}

					chainIndex += 2;
				}
			}

			if( foundBucket )
			{
				// Remove the MSB, we no longer need it
				int idIndex = chainTables[l][chainIndex + 1] & 0x7FFFFFFF;
				while( true )
				{
					int id = indexTables[l][idIndex];

					// Insert id without the MSB
					cand[numCands] = id & 0x7FFFFFFF;
					numCands++;

					if( id < 0 )
					{
						break;
					}

					idIndex++;
				}
			}
		}

		__syncthreads();

		for( int c = 0; c < numCands; ++c )
		{
			found = false;

			__syncthreads();

			if( candidates[threadID] == cand[c] )
			{
				found = true;
			}

			__syncthreads();

			if( threadID == 0 )
			{
				if( !found && idx < MAX_CAND_LIST_SIZE )
				{
					candidates[idx] = cand[c];
					idx++;
				}
			}

			__syncthreads();
		}

		__syncthreads();
	}

	size_t best = 0;
	float minDiff = __FLT_MAX__;

	volatile __shared__ float diff[8][32];

	int histFile = gopen(histFileName, O_GRDONLY);

	__syncthreads();

	for( int i = 0; i < MAX_CAND_LIST_SIZE; ++i )
	{
		float myDiff = 0;

		if( candidates[ i ] < 0 ) break;

		size_t candID = candidates[ i ];

		volatile float* cand = (volatile float*)gmmap(NULL, 3 * 256 * sizeof(float), 0, O_GRDONLY, histFile, candID * HIST_SIZE_ON_DISK);
		__syncthreads();

		cand += threadID;
		float candHist1 = *cand;

		cand += 256;
		float candHist2 = *cand;

		cand += 256;
		float candHist3 = *cand;

		myDiff += ( myBlockHist1 - candHist1 ) * ( myBlockHist1 - candHist1 );
		myDiff += ( myBlockHist2 - candHist2 ) * ( myBlockHist2 - candHist2 );
		myDiff += ( myBlockHist3 - candHist3 ) * ( myBlockHist3 - candHist3 );

		diff[ty][tx] = myDiff;

		if( tx < 16 )
		{
			diff[ty][tx] += diff[ty][tx + 16];
			diff[ty][tx] += diff[ty][tx + 8];
			diff[ty][tx] += diff[ty][tx + 4];
			diff[ty][tx] += diff[ty][tx + 2];
			diff[ty][tx] += diff[ty][tx + 1];
		}

		__syncthreads();

		if( (ty == 0) && (tx < 8) )
		{
			diff[0][tx] = diff[tx][0];
		}
		__syncthreads();

		if( (ty == 0) && (tx < 4) )
		{
			diff[0][tx] += diff[0][tx + 4];
			diff[0][tx] += diff[0][tx + 2];
			diff[0][tx] += diff[0][tx + 1];
		}

		__syncthreads();

		if( threadID == 0 )
		{
			if( diff[0][0] < minDiff )
			{
				minDiff = diff[0][0];
				best = candID;
			}
		}

		gmunmap(cand, NULL);
		__syncthreads();
	}

	gclose(histFile);

	if( 0 == threadID )
	{
		((int*)inOut)[base] = best;
	}
}

void mosaic_GPUfs( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS )
{
	checkCudaErrors( cudaSetDeviceFlags(cudaDeviceMapHost) );

	float msec = 0.f;

	const size_t IMAGE_SIZE = ROWS * COLS * 4 * sizeof( uchar );
	const size_t COEF_SIZE = data.D * /*data.K*/32 * data.L * sizeof(float);

	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals);
	init_device_app();

	cudaStream_t mainStream = gpuGlobals->streamMgr->kernelStream;

	cudaEvent_t start, stop;

	checkCudaErrors( cudaEventCreate(&start) );
	checkCudaErrors( cudaEventCreate(&stop) );

	float *h_coef = (float*)malloc( COEF_SIZE );
	data.getCoefficentsTransposed( h_coef );

	// Allocate the device input image
	uchar *d_inOut = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_inOut, IMAGE_SIZE) );

	// Allocate buffer for LSH functions coefficients
	float *d_coef = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_coef, COEF_SIZE) );

	// Copy the host to the device memory
	checkCudaErrors( cudaMemcpy(d_inOut, inOut, IMAGE_SIZE, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_coef, h_coef, COEF_SIZE, cudaMemcpyHostToDevice) );

	size_t tablesAlocationSize = 0;

    void **h_hashTables = (void**)malloc(sizeof(void*) * NUM_TABLES);
    void **h_chainTables = (void**)malloc(sizeof(void*) * NUM_TABLES);
    void **h_indexTables = (void**)malloc(sizeof(void*) * NUM_TABLES);

    for( int i = 0; i < NUM_TABLES; ++i )
    {
    	checkCudaErrors( cudaMalloc((void **)&h_hashTables[i], data.hashTableSizes[i] * sizeof(int)) );
    	checkCudaErrors( cudaMemcpy(h_hashTables[i], data.hashTable[i], data.hashTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

    	checkCudaErrors( cudaMalloc((void **)&h_chainTables[i], data.chainTableSizes[i] * sizeof(int)) );
		checkCudaErrors( cudaMemcpy(h_chainTables[i], data.chainTable[i], data.chainTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaMalloc((void **)&h_indexTables[i], data.indexTableSizes[i] * sizeof(int)) );
		checkCudaErrors( cudaMemcpy(h_indexTables[i], data.indexTable[i], data.indexTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

		tablesAlocationSize += data.hashTableSizes[i] * sizeof(int);
		tablesAlocationSize += data.chainTableSizes[i] * sizeof(int);
		tablesAlocationSize += data.indexTableSizes[i] * sizeof(int);
    }

    int **d_hashTables = NULL;
    int **d_chainTables = NULL;
    int **d_indexTables = NULL;

    checkCudaErrors( cudaMalloc((void **)&d_hashTables, sizeof(void*) * NUM_TABLES) );
    checkCudaErrors( cudaMemcpy(d_hashTables, h_hashTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMalloc((void **)&d_chainTables, sizeof(void*) * NUM_TABLES) );
	checkCudaErrors( cudaMemcpy(d_chainTables, h_chainTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMalloc((void **)&d_indexTables, sizeof(void*) * NUM_TABLES) );
	checkCudaErrors( cudaMemcpy(d_indexTables, h_indexTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

	tablesAlocationSize += sizeof(void*) * NUM_TABLES;
	tablesAlocationSize += sizeof(void*) * NUM_TABLES;
	tablesAlocationSize += sizeof(void*) * NUM_TABLES;

    cout << "Alloc sizes" << endl;
    cout << "Input image: " << (double)IMAGE_SIZE / 1024 / 1024 << "MB" << endl;
    cout << "Coefficients: " << (double)COEF_SIZE / 1024 / 1024 << "MB" << endl;
    cout << "Tables: " << (double)(tablesAlocationSize) / 1024 / 1024 << "MB" << endl;

	checkCudaErrors( cudaEventRecord(start, gpuGlobals->streamMgr->kernelStream) );

	char* d_histFileName = allocate_filename(data.histogramsFileName.c_str());

	// Launch the CUDA Kernel
	dim3 block(32, 8);
	dim3 grid(COLS / BLOCK_SIZE, ROWS / BLOCK_SIZE);
	mosaicKernel<<<grid, block, 0, gpuGlobals->streamMgr->kernelStream>>>( d_histFileName, 
																		   d_inOut,
			                                                               d_coef,
			                                                               d_hashTables,
			                                                               d_chainTables,
			                                                               d_indexTables,
			                                                               COLS,
			                                                               data.D,
			                                                               data.K,
			                                                               data.L );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaEventRecord(stop, gpuGlobals->streamMgr->kernelStream) );

	run_gpufs_handler(gpuGlobals, 0);

	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "total,time," << msec << ",ms" << endl;

	checkCudaErrors( cudaMemcpy(inOut, d_inOut, IMAGE_SIZE, cudaMemcpyDeviceToHost) );

	int i = 0;

	for( int by = 0; by < ROWS; by += 32 )
	{
		for( int bx = 0; bx < COLS; bx += 32 )
		{
			int bestImage = ((int*)inOut)[by * COLS + bx];

			bests[i++] = bestImage;
		}
	}

	delete gpuGlobals;
}

#ifdef WARP

__global__ void mosaicKernelWarp(const char* histFileName, uchar *inOut, float* coef, int** hashTables, int** chainTables, int** indexTables, const int width, const int D,const int K, const int L)
{
	__shared__ float  hist[3 * 256];

	Pixel* im = (Pixel*)inOut;

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int threadID = ty * 32 + tx;
	int laneID = tx & 0x1f;

	hist[0 * 256 + threadID] = 0.f;
	hist[1 * 256 + threadID] = 0.f;
	hist[2 * 256 + threadID] = 0.f;

	__syncthreads();

	int base = by * BLOCK_SIZE * width + bx * BLOCK_SIZE;

	Pixel myPix0 = im[base + (ty + 0 * 8) * width + tx];
	Pixel myPix1 = im[base + (ty + 1 * 8) * width + tx];
	Pixel myPix2 = im[base + (ty + 2 * 8) * width + tx];
	Pixel myPix3 = im[base + (ty + 3 * 8) * width + tx];

	atomicAdd( (float*)&hist[0 * 256 + myPix0.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix0.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix0.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix1.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix1.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix1.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix2.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix2.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix2.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix3.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix3.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix3.b ], 1.f );

	__syncthreads();

	__shared__ uint keys[32];

	for( int i = 0; i < 4; ++i )
	{
		uint bit = 0;

		if( tx < K )
		{
			float value = 0;

			for(int d = 0; d < D; ++d)
			{
				// We use 32 instead of K for alignment
				value += (hist[d * 4 + 0] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 1] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 2] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 3] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
			}

			bit = ( value > 0 ) ? 1 : 0;
		}

		keys[ty + i * 8] = __ballot( bit );
	}

	__shared__ volatile int candidates[256];
	__shared__ volatile int idx;
	__shared__ volatile int cand[32];
	__shared__ volatile bool found;

	candidates[threadID] = -1;
	idx = 0;

	__syncthreads();

	for( int l = 0; l < 32; ++l )
	{
		__shared__ int numCands;

		if( threadID == 0 )
		{
			bool foundBucket = false;
			numCands = 0;

			uint hIndex = keys[l] % 1000000;
			uint control = keys[l];

			int chainIndex = hashTables[l][hIndex];
			if( ALL_ONES != chainIndex )
			{
				while( true )
				{
					if( chainTables[l][chainIndex] == control )
					{
						foundBucket = true;
						break;
					}

					if( chainTables[l][chainIndex + 1] < 0 )
					{
						// If MSB is set, this means it's the last node in the chain
						break;
					}

					chainIndex += 2;
				}
			}

			if( foundBucket )
			{
				// Remove the MSB, we no longer need it
				int idIndex = chainTables[l][chainIndex + 1] & 0x7FFFFFFF;
				while( true )
				{
					int id = indexTables[l][idIndex];

					// Insert id without the MSB
					cand[numCands] = id & 0x7FFFFFFF;
					numCands++;

					if( id < 0 )
					{
						break;
					}

					idIndex++;
				}
			}
		}

		__syncthreads();

		for( int c = 0; c < numCands; ++c )
		{
			found = false;

			__syncthreads();

			if( candidates[threadID] == cand[c] )
			{
				found = true;
			}

			__syncthreads();

			if( threadID == 0 )
			{
				if( !found && idx < MAX_CAND_LIST_SIZE )
				{
					candidates[idx] = cand[c];
					idx++;
				}
			}

			__syncthreads();
		}

		__syncthreads();
	}

	int histFile = gopen(histFileName, O_GRDONLY);

	size_t best = 0;
	float minDiff = __FLT_MAX__;

	__shared__ size_t bests[8];
	__shared__ float minDiffs[8];

	for( int i = ty; i < MAX_CAND_LIST_SIZE; i += 8 )
	{
		float diff = 0;
		size_t candID = candidates[ i ];

		if( candidates[ i ] < 0 ) break;

		volatile float* cand = (volatile float*)gmmap_warp(NULL, 3 * 256 * sizeof(float), 0, O_GRDONLY, histFile, candID * HIST_SIZE_ON_DISK);
		cand += laneID;

		for( int k = 0; k < 768; k += 32 )
		{
			if( k != 0 )
			{
				cand += 32;
			}

			float candHist = *cand;

			float myDiff = ( hist[k + laneID] - candHist ) * ( hist[k + laneID] - candHist );

			myDiff += __shfl_down( myDiff, 16 );
			myDiff += __shfl_down( myDiff, 8 );
			myDiff += __shfl_down( myDiff, 4 );
			myDiff += __shfl_down( myDiff, 2 );
			myDiff += __shfl_down( myDiff, 1 );

			diff += myDiff;
		}

		gmunmap_warp(cand, NULL);

		if( laneID == 0 )
		{
			if( diff < minDiff )
			{
				minDiff = diff;
				best = candID;
			}
		}
	}

	if( laneID == 0 )
	{
		bests[ty] = best;
		minDiffs[ty] = minDiff;
	}

	__syncthreads();
	gclose(histFile);

	// Now we need to go over all the bests and find the best one
	if( 0 == threadID )
	{
		for( int i = 1; i < 8; ++i )
		{
			if( minDiffs[i] < minDiff )
			{
				minDiff = minDiffs[i];
				best = bests[i];
			}
		}

		((int*)inOut)[base] = best;
	}
}

void mosaic_GPUfs_warp( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS )
{
	checkCudaErrors( cudaSetDeviceFlags(cudaDeviceMapHost) );

	float msec = 0.f;

	const size_t IMAGE_SIZE = ROWS * COLS * 4 * sizeof( uchar );
	const size_t COEF_SIZE = data.D * /*data.K*/32 * data.L * sizeof(float);

	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals);
	init_device_app();

	cudaStream_t mainStream = gpuGlobals->streamMgr->kernelStream;

	cudaEvent_t start, stop;

	checkCudaErrors( cudaEventCreate(&start) );
	checkCudaErrors( cudaEventCreate(&stop) );

	float *h_coef = (float*)malloc( COEF_SIZE );
	data.getCoefficentsTransposed( h_coef );

	// Allocate the device input image
	uchar *d_inOut = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_inOut, IMAGE_SIZE) );

	// Allocate buffer for LSH functions coefficients
	float *d_coef = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_coef, COEF_SIZE) );

	// Copy the host to the device memory
	checkCudaErrors( cudaMemcpy(d_inOut, inOut, IMAGE_SIZE, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_coef, h_coef, COEF_SIZE, cudaMemcpyHostToDevice) );

	size_t tablesAlocationSize = 0;

    void **h_hashTables = (void**)malloc(sizeof(void*) * NUM_TABLES);
    void **h_chainTables = (void**)malloc(sizeof(void*) * NUM_TABLES);
    void **h_indexTables = (void**)malloc(sizeof(void*) * NUM_TABLES);

    for( int i = 0; i < NUM_TABLES; ++i )
    {
    	checkCudaErrors( cudaMalloc((void **)&h_hashTables[i], data.hashTableSizes[i] * sizeof(int)) );
    	checkCudaErrors( cudaMemcpy(h_hashTables[i], data.hashTable[i], data.hashTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

    	checkCudaErrors( cudaMalloc((void **)&h_chainTables[i], data.chainTableSizes[i] * sizeof(int)) );
		checkCudaErrors( cudaMemcpy(h_chainTables[i], data.chainTable[i], data.chainTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaMalloc((void **)&h_indexTables[i], data.indexTableSizes[i] * sizeof(int)) );
		checkCudaErrors( cudaMemcpy(h_indexTables[i], data.indexTable[i], data.indexTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

		tablesAlocationSize += data.hashTableSizes[i] * sizeof(int);
		tablesAlocationSize += data.chainTableSizes[i] * sizeof(int);
		tablesAlocationSize += data.indexTableSizes[i] * sizeof(int);
    }

    int **d_hashTables = NULL;
    int **d_chainTables = NULL;
    int **d_indexTables = NULL;

    checkCudaErrors( cudaMalloc((void **)&d_hashTables, sizeof(void*) * NUM_TABLES) );
    checkCudaErrors( cudaMemcpy(d_hashTables, h_hashTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMalloc((void **)&d_chainTables, sizeof(void*) * NUM_TABLES) );
	checkCudaErrors( cudaMemcpy(d_chainTables, h_chainTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMalloc((void **)&d_indexTables, sizeof(void*) * NUM_TABLES) );
	checkCudaErrors( cudaMemcpy(d_indexTables, h_indexTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

	tablesAlocationSize += sizeof(void*) * NUM_TABLES;
	tablesAlocationSize += sizeof(void*) * NUM_TABLES;
	tablesAlocationSize += sizeof(void*) * NUM_TABLES;

    cout << "Alloc sizes" << endl;
    cout << "Input image: " << (double)IMAGE_SIZE / 1024 / 1024 << "MB" << endl;
    cout << "Coefficients: " << (double)COEF_SIZE / 1024 / 1024 << "MB" << endl;
    cout << "Tables: " << (double)(tablesAlocationSize) / 1024 / 1024 << "MB" << endl;

	char* d_histFileName = allocate_filename(data.histogramsFileName.c_str());

    dim3 block(32, 8);
	dim3 grid(COLS / BLOCK_SIZE, ROWS / BLOCK_SIZE);

#ifdef WARMUP

    // Run warmup
	checkCudaErrors( cudaEventRecord(start, gpuGlobals->streamMgr->kernelStream) );

	// Launch the CUDA Kernel
	mosaicKernelWarp<<<grid, block, 0, gpuGlobals->streamMgr->kernelStream>>>( d_histFileName,
																		   d_inOut,
			                                                               d_coef,
			                                                               d_hashTables,
			                                                               d_chainTables,
			                                                               d_indexTables,
			                                                               COLS,
			                                                               data.D,
			                                                               data.K,
			                                                               data.L );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaEventRecord(stop, gpuGlobals->streamMgr->kernelStream) );

	run_gpufs_handler(gpuGlobals, 0);

	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "warmup time: " << msec << "ms" << endl;

#endif

	checkCudaErrors( cudaEventRecord(start, gpuGlobals->streamMgr->kernelStream) );

	// Launch the CUDA Kernel
	mosaicKernelWarp<<<grid, block, 0, gpuGlobals->streamMgr->kernelStream>>>( d_histFileName,
																	       d_inOut,
																		   d_coef,
																		   d_hashTables,
																		   d_chainTables,
																		   d_indexTables,
																		   COLS,
																		   data.D,
																		   data.K,
																		   data.L );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaEventRecord(stop, gpuGlobals->streamMgr->kernelStream) );

	run_gpufs_handler(gpuGlobals, 0);

	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "total time: " << msec << "ms" << endl;

	checkCudaErrors( cudaMemcpy(inOut, d_inOut, IMAGE_SIZE, cudaMemcpyDeviceToHost) );

	int i = 0;

	for( int by = 0; by < ROWS; by += 32 )
	{
		for( int bx = 0; bx < COLS; bx += 32 )
		{
			int bestImage = ((int*)inOut)[by * COLS + bx];

			bests[i++] = bestImage;
		}
	}

	delete gpuGlobals;
}

#ifdef GPUFS_VM
__global__ void mosaicKernelVM(const char* histFileName, uchar *inOut, float* coef, int** hashTables, int** chainTables, int** indexTables, const int width, const int D,const int K, const int L)
{
	__shared__ TLB<TLB_SIZE> tlb;

	__shared__ float  hist[3 * 256];

	Pixel* im = (Pixel*)inOut;

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int threadID = ty * 32 + tx;
	int laneID = tx & 0x1f;

	hist[0 * 256 + threadID] = 0.f;
	hist[1 * 256 + threadID] = 0.f;
	hist[2 * 256 + threadID] = 0.f;

	__syncthreads();

	int base = by * BLOCK_SIZE * width + bx * BLOCK_SIZE;

	Pixel myPix0 = im[base + (ty + 0 * 8) * width + tx];
	Pixel myPix1 = im[base + (ty + 1 * 8) * width + tx];
	Pixel myPix2 = im[base + (ty + 2 * 8) * width + tx];
	Pixel myPix3 = im[base + (ty + 3 * 8) * width + tx];

	atomicAdd( (float*)&hist[0 * 256 + myPix0.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix0.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix0.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix1.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix1.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix1.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix2.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix2.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix2.b ], 1.f );

	atomicAdd( (float*)&hist[0 * 256 + myPix3.r ], 1.f );
	atomicAdd( (float*)&hist[1 * 256 + myPix3.g ], 1.f );
	atomicAdd( (float*)&hist[2 * 256 + myPix3.b ], 1.f );

	__syncthreads();

	__shared__ uint keys[32];

	for( int i = 0; i < 4; ++i )
	{
		uint bit = 0;

		if( tx < K )
		{
			float value = 0;

			for(int d = 0; d < D; ++d)
			{
				// We use 32 instead of K for alignment
				value += (hist[d * 4 + 0] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 1] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 2] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
				value += (hist[d * 4 + 3] * coef[ (ty + i * 8) * 32 * D + d * 32 + (K - 1 - tx) ]);
			}

			bit = ( value > 0 ) ? 1 : 0;
		}

		keys[ty + i * 8] = __ballot( bit );
	}

	__shared__ volatile int candidates[256];
	__shared__ volatile int idx;
	__shared__ volatile int cand[32];
	__shared__ volatile bool found;

	candidates[threadID] = -1;
	idx = 0;

	__syncthreads();

	for( int l = 0; l < 32; ++l )
	{
		__shared__ int numCands;

		if( threadID == 0 )
		{
			bool foundBucket = false;
			numCands = 0;

			uint hIndex = keys[l] % 1000000;
			uint control = keys[l];

			int chainIndex = hashTables[l][hIndex];
			if( ALL_ONES != chainIndex )
			{
				while( true )
				{
					if( chainTables[l][chainIndex] == control )
					{
						foundBucket = true;
						break;
					}

					if( chainTables[l][chainIndex + 1] < 0 )
					{
						// If MSB is set, this means it's the last node in the chain
						break;
					}

					chainIndex += 2;
				}
			}

			if( foundBucket )
			{
				// Remove the MSB, we no longer need it
				int idIndex = chainTables[l][chainIndex + 1] & 0x7FFFFFFF;
				while( true )
				{
					int id = indexTables[l][idIndex];

					// Insert id without the MSB
					cand[numCands] = id & 0x7FFFFFFF;
					numCands++;

					if( id < 0 )
					{
						break;
					}

					idIndex++;
				}
			}
		}

		__syncthreads();

		for( int c = 0; c < numCands; ++c )
		{
			found = false;

			__syncthreads();

			if( candidates[threadID] == cand[c] )
			{
				found = true;
			}

			__syncthreads();

			if( threadID == 0 )
			{
				if( !found && idx < MAX_CAND_LIST_SIZE )
				{
					candidates[idx] = cand[c];
					idx++;
				}
			}

			__syncthreads();
		}

		__syncthreads();
	}

	int histFile = gopen(histFileName, O_GRDONLY);

	size_t best = 0;
	float minDiff = __FLT_MAX__;

	__shared__ size_t bests[8];
	__shared__ float minDiffs[8];

	{
		// Declare scope for VM to force fat pointer destructor

		FatPointer<volatile float, TLB_SIZE> candPtr = gvmmap<volatile float, TLB_SIZE>(NULL, INT64_MAX, 0, O_GRDONLY, histFile, 0, &tlb);

		for( int i = ty; i < MAX_CAND_LIST_SIZE; i += 8 )
		{
			float diff = 0;
			size_t candID = candidates[ i ];

			if( candidates[ i ] < 0 ) break;

			candPtr.moveTo( candID * HIST_SIZE_ON_DISK );
			candPtr += laneID;

			for( int k = 0; k < 768; k += 32 )
			{
				if( k != 0 )
				{
					candPtr += 32;
				}

				float candHist = *candPtr;

				float myDiff = ( hist[k + laneID] - candHist ) * ( hist[k + laneID] - candHist );

				myDiff += __shfl_down( myDiff, 16 );
				myDiff += __shfl_down( myDiff, 8 );
				myDiff += __shfl_down( myDiff, 4 );
				myDiff += __shfl_down( myDiff, 2 );
				myDiff += __shfl_down( myDiff, 1 );

				diff += myDiff;
			}

			if( laneID == 0 )
			{
				if( diff < minDiff )
				{
					minDiff = diff;
					best = candID;
				}
			}
		}
	}

	if( laneID == 0 )
	{
		bests[ty] = best;
		minDiffs[ty] = minDiff;
	}

	__syncthreads();
	gclose(histFile);

	// Now we need to go over all the bests and find the best one
	if( 0 == threadID )
	{
		for( int i = 1; i < 8; ++i )
		{
			if( minDiffs[i] < minDiff )
			{
				minDiff = minDiffs[i];
				best = bests[i];
			}
		}

		((int*)inOut)[base] = best;
	}
}

void mosaic_GPUfs_VM( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS )
{
	checkCudaErrors( cudaSetDeviceFlags(cudaDeviceMapHost) );

	float msec = 0.f;

	const size_t IMAGE_SIZE = ROWS * COLS * 4 * sizeof( uchar );
	const size_t COEF_SIZE = data.D * /*data.K*/32 * data.L * sizeof(float);

	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals);
	init_device_app();

	cudaStream_t mainStream = gpuGlobals->streamMgr->kernelStream;

	cudaEvent_t start, stop;

	checkCudaErrors( cudaEventCreate(&start) );
	checkCudaErrors( cudaEventCreate(&stop) );

	float *h_coef = (float*)malloc( COEF_SIZE );
	data.getCoefficentsTransposed( h_coef );

	// Allocate the device input image
	uchar *d_inOut = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_inOut, IMAGE_SIZE) );

	// Allocate buffer for LSH functions coefficients
	float *d_coef = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_coef, COEF_SIZE) );

	// Copy the host to the device memory
	checkCudaErrors( cudaMemcpy(d_inOut, inOut, IMAGE_SIZE, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_coef, h_coef, COEF_SIZE, cudaMemcpyHostToDevice) );

	size_t tablesAlocationSize = 0;

    void **h_hashTables = (void**)malloc(sizeof(void*) * NUM_TABLES);
    void **h_chainTables = (void**)malloc(sizeof(void*) * NUM_TABLES);
    void **h_indexTables = (void**)malloc(sizeof(void*) * NUM_TABLES);

    for( int i = 0; i < NUM_TABLES; ++i )
    {
    	checkCudaErrors( cudaMalloc((void **)&h_hashTables[i], data.hashTableSizes[i] * sizeof(int)) );
    	checkCudaErrors( cudaMemcpy(h_hashTables[i], data.hashTable[i], data.hashTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

    	checkCudaErrors( cudaMalloc((void **)&h_chainTables[i], data.chainTableSizes[i] * sizeof(int)) );
		checkCudaErrors( cudaMemcpy(h_chainTables[i], data.chainTable[i], data.chainTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

		checkCudaErrors( cudaMalloc((void **)&h_indexTables[i], data.indexTableSizes[i] * sizeof(int)) );
		checkCudaErrors( cudaMemcpy(h_indexTables[i], data.indexTable[i], data.indexTableSizes[i] * sizeof(int), cudaMemcpyHostToDevice) );

		tablesAlocationSize += data.hashTableSizes[i] * sizeof(int);
		tablesAlocationSize += data.chainTableSizes[i] * sizeof(int);
		tablesAlocationSize += data.indexTableSizes[i] * sizeof(int);
    }

    int **d_hashTables = NULL;
    int **d_chainTables = NULL;
    int **d_indexTables = NULL;

    checkCudaErrors( cudaMalloc((void **)&d_hashTables, sizeof(void*) * NUM_TABLES) );
    checkCudaErrors( cudaMemcpy(d_hashTables, h_hashTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMalloc((void **)&d_chainTables, sizeof(void*) * NUM_TABLES) );
	checkCudaErrors( cudaMemcpy(d_chainTables, h_chainTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMalloc((void **)&d_indexTables, sizeof(void*) * NUM_TABLES) );
	checkCudaErrors( cudaMemcpy(d_indexTables, h_indexTables, sizeof(void*) * NUM_TABLES, cudaMemcpyHostToDevice) );

	tablesAlocationSize += sizeof(void*) * NUM_TABLES;
	tablesAlocationSize += sizeof(void*) * NUM_TABLES;
	tablesAlocationSize += sizeof(void*) * NUM_TABLES;

    cout << "Alloc sizes" << endl;
    cout << "Input image: " << (double)IMAGE_SIZE / 1024 / 1024 << "MB" << endl;
    cout << "Coefficients: " << (double)COEF_SIZE / 1024 / 1024 << "MB" << endl;
    cout << "Tables: " << (double)(tablesAlocationSize) / 1024 / 1024 << "MB" << endl;

	char* d_histFileName = allocate_filename(data.histogramsFileName.c_str());

    dim3 block(32, 8);
	dim3 grid(COLS / BLOCK_SIZE, ROWS / BLOCK_SIZE);

#ifdef WARMUP

	checkCudaErrors( cudaEventRecord(start, gpuGlobals->streamMgr->kernelStream) );

	// Launch the CUDA Kernel
	mosaicKernelVM<<<grid, block, 0, gpuGlobals->streamMgr->kernelStream>>>( d_histFileName, 
																		   d_inOut,
			                                                               d_coef,
			                                                               d_hashTables,
			                                                               d_chainTables,
			                                                               d_indexTables,
			                                                               COLS,
			                                                               data.D,
			                                                               data.K,
			                                                               data.L );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaEventRecord(stop, gpuGlobals->streamMgr->kernelStream) );

	run_gpufs_handler(gpuGlobals, 0);

	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "warmup time: " << msec << "ms" << endl;

#endif

	checkCudaErrors( cudaEventRecord(start, gpuGlobals->streamMgr->kernelStream) );

	// Launch the CUDA Kernel
	mosaicKernelVM<<<grid, block, 0, gpuGlobals->streamMgr->kernelStream>>>( d_histFileName,
																		   d_inOut,
																		   d_coef,
																		   d_hashTables,
																		   d_chainTables,
																		   d_indexTables,
																		   COLS,
																		   data.D,
																		   data.K,
																		   data.L );
	checkCudaErrors( cudaGetLastError() );
	checkCudaErrors( cudaEventRecord(stop, gpuGlobals->streamMgr->kernelStream) );

	run_gpufs_handler(gpuGlobals, 0);

	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "total time: " << msec << "ms" << endl;

	checkCudaErrors( cudaMemcpy(inOut, d_inOut, IMAGE_SIZE, cudaMemcpyDeviceToHost) );

	int i = 0;

	for( int by = 0; by < ROWS; by += 32 )
	{
		for( int bx = 0; bx < COLS; bx += 32 )
		{
			int bestImage = ((int*)inOut)[by * COLS + bx];

			bests[i++] = bestImage;
		}
	}

	delete gpuGlobals;
}
#endif

#endif
