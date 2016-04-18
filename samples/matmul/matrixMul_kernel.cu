/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) *((&Bs[0][0])+(i)*(BLOCK_SIZE+1)+(j))
#define CS(i, j) Cs[i][j]
#endif


#include "fs_globals.cu.h"
#include "fs_calls.cu.h"

__device__ volatile INIT_LOCK init_lock;
__device__ volatile LAST_SEMAPHORE last_lock;

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////

__global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep)
	{

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

__device__ float tmp_a[1<<22];
__device__ float tmp_b[1<<22];
__device__ float tmp_c[1<<22];

__global__ void

#define SS_DEBUG() if (0 == TID) printf("%d\n", __LINE__);
matrixMul( int wA, int wB, int perBlockX, int perBlockY, char n)
{
	int f_a,f_b,f_c;

	f_a=0;
	f_a=gopen("mtx_a",O_GRDONLY);
	if (f_a<0) { ERROR("Failed to open a");}


	f_b=0;
	f_b=gopen("mtx_b",O_GRDONLY);
	if (f_b<0) { ERROR("Failed to open B");}

	f_c=0;
	char out[6]="mtx_c"; out[0]=n;


	f_c=gopen(out,O_GWRONLY);
	//   f_c=gopen("mtx_c",O_GWRONCE);
	if (f_c<0) { ERROR("Failed to open c");}

	for (int by=blockIdx.y*perBlockY;by<(blockIdx.y+1)*perBlockY;by++){

		int wC=wB;
		int cBegin = wC*BLOCK_SIZE*by*sizeof(float); 

		volatile float* ptr_c=(volatile float*)gmmap(NULL,wC*BLOCK_SIZE*sizeof(float),0, O_GWRONCE, f_c,cBegin);
		//		volatile float * ptr_c=tmp_c;
		if (ptr_c==GMAP_FAILED) ERROR("GMMAP failed with m_c");


		int aBegin = wA*BLOCK_SIZE*by*sizeof(float);
		volatile float* ptr_a=(volatile float*)gmmap(NULL,wA*BLOCK_SIZE*sizeof(float),0, O_GRDONLY, f_a,aBegin);
		//	    volatile float* ptr_a=tmp_a;
		if (ptr_a==GMAP_FAILED) ERROR("GMMAP failed with m_a");

		for (int bx=0;bx<perBlockX;bx++){

			// Index of the first sub-matrix of A processed by the block
			int hB=wA;

			int bBegin = hB*BLOCK_SIZE*bx*sizeof(float);


			volatile float* ptr_b=(volatile float*)gmmap(NULL,wA*BLOCK_SIZE*sizeof(float),0, O_GRDONLY, f_b,bBegin);
			//		    volatile float * ptr_b=tmp_b;
			//		    if (ptr_b==GMAP_FAILED) ERROR("GMMAP failed with m_b");

			// Block index

			// Thread index
			int tx = threadIdx.x;
			int ty = threadIdx.y;


			// Index of the last sub-matrix of A processed by the block
			int aEnd   =  wA - 1;

			// Step size used to iterate through the sub-matrices of A
			int aStep  = BLOCK_SIZE;

			// Csub is used to store the element of the block sub-matrix
			// that is computed by the thread
			float Csub = 0;

			for( int a = 0, b= 0; a <=aEnd; a += aStep, b += aStep) {
				// Declaration of the shared memory array As used to
				// store the sub-matrix of A
				__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

				// Declaration of the shared memory array Bs used to
				// store the sub-matrix of B
				__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+1];


				// Load the matrices from device memory
				// to shared memory; each thread loads
				// one element of each matrix

				AS(ty, tx) = ptr_a[a + wA * ty + tx];
				BS(ty, tx) = ptr_b[b + wA * ty + tx];
				//	 AS(ty,tx)=1;
				//	 BS(ty,tx)=1;

				// Synchronize to make sure the matrices are loaded
				__syncthreads();

				// Multiply the two matrices together;
				// each thread computes one element
				// of the block sub-matrix
				//	#pragma unroll
				float* bs_ptr=&BS(tx,0);
				for (int k = 0; k < BLOCK_SIZE; ++k){
					Csub+=AS(ty,k)*(*(bs_ptr+k));; 
				}
				//if (Csub!=-1) Csub=AS(0,0);
				// Synchronize to make sure that the preceding
				// computation is done before loading two new
				// sub-matrices of A and B in the next iteration
				__syncthreads();
			}

			ptr_c[bx*BLOCK_SIZE+ wB*ty+tx]=Csub;

			//		}
			// Write the block sub-matrix to device memory;
			// each thread writes one element

			gmunmap(ptr_b,0);
		}
		if(perBlockX>4 ) gmsync(ptr_c,0,0);

		gmunmap(ptr_c,0);
		gmunmap(ptr_a,0);
	}

	gclose(f_a);
	gclose(f_b);
	gclose(f_c);
}


void init_app()
{
	// INITI LOCK   
	void* inited;

	CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,init_lock));
	CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(INIT_LOCK)));

	CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,last_lock));
	CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(LAST_SEMAPHORE)));
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
