
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <iomanip>

#include "utils.h"
#include "loader.h"

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;

static const int MAX_CAND_LIST_SIZE = 256;

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

struct Pixel
{
	uchar r;
	uchar g;
	uchar b;
	uchar a;
};

__global__ void
calcKeys(uchar *inOut, uint *keys, float* coef, const int width, const int D,const int K, const int L)
{
	__shared__ float hist[3 * 256];

	Pixel* im = (Pixel*)inOut;

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int blockID = by * gridDim.x + bx;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int threadID = ty * blockDim.x + tx;

	if( threadID < 768 )
	{
		hist[threadID] = 0;
	}

	__syncthreads();

	int base = by * blockDim.y * width + bx * blockDim.x;

	Pixel myPix = im[base + ty * width + tx];

	atomicAdd( &hist[0 * 256 + myPix.r ], 1.f );
	atomicAdd( &hist[1 * 256 + myPix.g ], 1.f );
	atomicAdd( &hist[2 * 256 + myPix.b ], 1.f );

	__syncthreads();

	if( threadID < 768 )
	{
		float* globHist = (float*)inOut;

		globHist[base + ty * width + tx] = hist[threadID];
	}

	__syncthreads();

	uint bit = 0;
	if( tx < K )
	{
		float value = 0;

		for(int d = 0; d < D; ++d)
		{
			// We use 32 instead of K for alignment
			value += (hist[d * 4 + 0] * coef[ ty * 32 * D + d * 32 + (K - 1 - tx) ]);
			value += (hist[d * 4 + 1] * coef[ ty * 32 * D + d * 32 + (K - 1 - tx) ]);
			value += (hist[d * 4 + 2] * coef[ ty * 32 * D + d * 32 + (K - 1 - tx) ]);
			value += (hist[d * 4 + 3] * coef[ ty * 32 * D + d * 32 + (K - 1 - tx) ]);
		}

		bit = ( value > 0 ) ? 1 : 0;
	}

	uint key = __ballot( bit );

	// Using __ballot returns the LSH in reverse order from the one we computed offline
	// I think using (k - tx) instead of tx should fix it

	if( 0 == tx )
	{
		keys[ blockID * L + ty ] = key;
	}
}

__global__ void
chooseBest(uchar *inOut, int *candidates, int* hists, const int width, const int D,const int K, const int L)
{
	// Block index
	size_t bx = blockIdx.x;
	size_t by = blockIdx.y;

	size_t blockID = by * gridDim.x + bx;

	// Thread index
	size_t tx = threadIdx.x;
	size_t ty = threadIdx.y;

	size_t threadID = ty * blockDim.x + tx;

	size_t base = by * 32 * width + bx * 32;

	float myBlockHist1 = ((float*)inOut)[base + (ty + 0 * 8) * width + tx];
	float myBlockHist2 = ((float*)inOut)[base + (ty + 1 * 8) * width + tx];
	float myBlockHist3 = ((float*)inOut)[base + (ty + 2 * 8) * width + tx];

	size_t best = 0;
	float minDiff = __FLT_MAX__;

	volatile __shared__ float diff[8][32];

	for( int i = 0; i < MAX_CAND_LIST_SIZE; ++i )
	{
		if( candidates[ blockID * MAX_CAND_LIST_SIZE + i ] < 0 ) break;

		size_t candID = candidates[ blockID * MAX_CAND_LIST_SIZE + i ];
		uchar* candT = (uchar*)hists + candID * HIST_SIZE_ON_DISK;
		float* cand = (float*)candT;

		__syncthreads();

		float myDiff = 0;
		float candHist1 = cand[0 * 256 + threadID];
		float candHist2 = cand[1 * 256 + threadID];
		float candHist3 = cand[2 * 256 + threadID];

		myDiff += ( myBlockHist1 - candHist1 ) * ( myBlockHist1 - candHist1 );
		myDiff += ( myBlockHist2 - candHist2 ) * ( myBlockHist2 - candHist2 );
		myDiff += ( myBlockHist3 - candHist3 ) * ( myBlockHist3 - candHist3 );

//		atomicAdd((float*)&diff[0][0], myDiff);

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

		__syncthreads();
	}

	__syncthreads();

	if( 0 == threadID )
	{
		((int*)inOut)[base] = best;
	}
}


void mosaic_GPU( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS )
{
	const int IMAGE_SIZE = ROWS * COLS * 4L * sizeof( uchar );
	const int BLOCKS = (ROWS / 32) * (COLS / 32);
	const int COEF_SIZE = data.D * /*data.K*/32 * data.L * sizeof(float);
	const int KEYS_SIZE = BLOCKS * data.L * sizeof(uint);
	const int CANDIDATES_SIZE = BLOCKS * MAX_CAND_LIST_SIZE * sizeof(uint);
	
	float *h_coef = (float*)malloc( COEF_SIZE );
	data.getCoefficentsTransposed( h_coef );

	uint* h_keys = (uint*)malloc( KEYS_SIZE );

	int* h_cand = (int*)malloc( CANDIDATES_SIZE );

	float msec = 0.f;
	cudaEvent_t start, stop;
	cudaEvent_t totalStart, totalStop;

	checkCudaErrors( cudaEventCreate(&start) );
	checkCudaErrors( cudaEventCreate(&stop) );
	checkCudaErrors( cudaEventCreate(&totalStart) );
	checkCudaErrors( cudaEventCreate(&totalStop) );

	checkCudaErrors( cudaEventRecord(start, NULL) );

	uchar* h_hists = NULL;
	checkCudaErrors( cudaHostAlloc( &h_hists, 1024 * 1024 * 1024 * 2L, 0) );

	int *d_hists = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_hists, 1024 * 1024 * 1024 * 2L) );

	checkCudaErrors( cudaEventRecord(stop, NULL) );
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Allocating histograms array (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(totalStart, NULL) );

	checkCudaErrors( cudaEventRecord(start, NULL) );

	// Allocate the device input image
	uchar *d_inOut = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_inOut, IMAGE_SIZE) );

	// Allocate buffer for LSH keys
	uint *d_keys = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_keys, KEYS_SIZE) );

	// Allocate buffer for candidates
	int *d_cand = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_cand, CANDIDATES_SIZE) );

	// Allocate buffer for LSH functions coefficients
	float *d_coef = NULL;
	checkCudaErrors( cudaMalloc((void **)&d_coef, COEF_SIZE) );

	checkCudaErrors( cudaEventRecord(stop, NULL) );
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Allocating input for calcKeys (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(start, NULL) );

	// Copy the host to the device memory
	checkCudaErrors( cudaMemcpy(d_inOut, inOut, IMAGE_SIZE, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMemcpy(d_coef, h_coef, COEF_SIZE, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaEventRecord(stop, NULL) );	
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Copying input for calcKeys (" << msec << "ms)" << endl;

	const int N = 32;

	// Running calc keys
	checkCudaErrors( cudaEventRecord(start, NULL) );

	// Launch the CUDA Kernel
	dim3 block(N, N);
	dim3 grid(COLS / N, ROWS / N);
	calcKeys<<<grid, block>>>( d_inOut, d_keys, d_coef, COLS, data.D, data.K, data.L );
	checkCudaErrors( cudaGetLastError() );

	checkCudaErrors( cudaEventRecord(stop, NULL) );	
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "calcKeys (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(start, NULL) );

	// Copy the device result to the host
	checkCudaErrors( cudaMemcpy(h_keys, d_keys, KEYS_SIZE, cudaMemcpyDeviceToHost) );

	checkCudaErrors( cudaEventRecord(stop, NULL) );	
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Copying calcKeys output to host (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(start, NULL) );

	// gather images
	uint* reverseGlobalImageList = new uint[10000000];
	size_t numImages = 0;

	merge_single(BLOCKS, h_keys, h_cand, data, reverseGlobalImageList, numImages);

	cout << "Num images: " << numImages << endl;

	checkCudaErrors( cudaEventRecord(stop, NULL) );
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Merging candidates (" << msec << "ms)" << endl;

	if( (numImages) > (1024L * 1024L * 2L / 4L) )
	{
		cout << "Too many images" << endl;
		exit(-1);
	}

	checkCudaErrors( cudaEventRecord(start, NULL) );

	for( int n = 0; n < numImages; ++n )
	{
		size_t offset = (size_t)reverseGlobalImageList[n] * HIST_SIZE_ON_DISK;

		int res = pread(data.histogramsFile, h_hists + (size_t)n * HIST_SIZE_ON_DISK, HIST_SIZE_ON_DISK, offset);
		assert( res > 0 );
	}

	checkCudaErrors( cudaEventRecord(stop, NULL) );	
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Reading histograms from file (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(start, NULL) );

	checkCudaErrors( cudaMemcpy(d_hists, h_hists, numImages * HIST_SIZE_ON_DISK, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_cand, h_cand, CANDIDATES_SIZE, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaEventRecord(stop, NULL) );	
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Copying histograms to device (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(start, NULL) );

	dim3 histBlock(32, 8);
	chooseBest<<<grid, histBlock>>>( d_inOut, d_cand, d_hists, COLS, data.D, data.K, data.L );
	checkCudaErrors( cudaGetLastError() );

	checkCudaErrors( cudaEventRecord(stop, NULL) );	
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Running chooseBest (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(start, NULL) );

	checkCudaErrors( cudaMemcpy(inOut, d_inOut, IMAGE_SIZE, cudaMemcpyDeviceToHost) );

	checkCudaErrors( cudaEventRecord(stop, NULL) );	
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, start, stop) );
	cout << "Copying chooseBest output to host (" << msec << "ms)" << endl;

	checkCudaErrors( cudaEventRecord(totalStop, NULL) );
	checkCudaErrors( cudaEventSynchronize(totalStop) );
	checkCudaErrors( cudaEventElapsedTime(&msec, totalStart, totalStop) );

	cout << "total time: " << msec << "ms" << endl;

	int i = 0;

	for( int by = 0; by < ROWS; by += 32 )
	{
		for( int bx = 0; bx < COLS; bx += 32 )
		{
			int bestID = ((int*)inOut)[by * COLS + bx];
			int bestImage = reverseGlobalImageList[bestID];

			bests[i++] = bestImage;
		}
	}

	cout << "Total list size: " << numImages << endl;

	checkCudaErrors( cudaFree(d_inOut) );
	checkCudaErrors( cudaFree(d_keys) );
	checkCudaErrors( cudaFree(d_coef) );
}
