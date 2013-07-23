/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>


#include "fs_constants.h"
#include "fs_debug.cu.h"
#include "util.cu.h"
#include "cpu_ipc.cu.h"
#include "mallocfree.cu.h"
#include "fs_structures.cu.h"
#include "timer.h"
#include "hash_table.cu.h"
#include "swapper.cu.h"
#include "fs_calls.cu.h"

/************GLOBALS********/
// CPU Write-shared memory //
__device__ volatile CPU_IPC_OPEN_Queue* g_cpu_ipcOpenQueue;
__device__ volatile CPU_IPC_RW_Queue* g_cpu_ipcRWQueue; 
//
// manager for rw RPC queue

__device__ volatile GPU_IPC_RW_Manager* g_ipcRWManager;

// Open/Close table
__device__ volatile OTable* g_otable;
// Memory pool
__device__ volatile PPool* g_ppool;
// File table with block pointers
__device__ volatile FTable* g_ftable;

// Radix tree memory pool for rt_nodes
__device__ volatile rt_mempool g_rtree_mempool;

// Hash table with all the previously opened files indexed by their inodes
__device__ volatile hash_table g_closed_ftable;

// file_id uniq counter
__device__ int g_file_id;

//pre close table
__device__ volatile preclose_table* g_preclose_table;



#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>


#include "fs_initializer.cu.h"
// INCLUDING CODE INLINE - change later
#include "host_loop.h"

#define MAIN_FS_FILE
#include "bigmatrix_mmap.cu"

void stdavg(double *avg_time, double *avg_thpt, double* std_time, double *std_thpt, const double* times, const double total_data, int arr_len)
{
	*avg_time=*avg_thpt=*std_time=*std_thpt=0;
	int counter=0;

	for( int i=0;i<arr_len;i++){
		if (times[i]<=0) continue;

		*avg_time+=times[i];
		*avg_thpt+=((double)total_data)/times[i];
		counter++;
	}
	if (counter==0) return;
	*avg_time/=(double)counter;
	*avg_thpt/=(double)counter;

	for( int i=0;i<arr_len;i++){
		if (times[i]<=0) continue;
		*std_time=(times[i]-*avg_time)*(times[i]-*avg_time);
	
		double tmp=(((double)total_data)/times[i])-*avg_thpt;
		*std_thpt=tmp*tmp;
	}
	*std_time/=(double)counter;
	*std_thpt/=(double)counter;

	*std_time=sqrt(*std_time);
	*std_thpt=sqrt(*std_thpt);
	
}


char*  update_filename(const char* h_filename){
	int n=strlen(h_filename);
	assert(n>0);
	if (n>FILENAME_SIZE) {
		fprintf(stderr,"Filname %s too long, should be only %d symbols including \\0",h_filename,FILENAME_SIZE);
		exit (-1);
	}
	char* d_filename;
	CUDA_SAFE_CALL(cudaMalloc(&d_filename,n+1));
	CUDA_SAFE_CALL(cudaMemcpy(d_filename, h_filename, n+1,cudaMemcpyHostToDevice));
	return d_filename;
}

#include <assert.h>

// size of the output used for data staging
int output_size=FS_BLOCKSIZE;

#define MAX_TRIALS (10)
double time_res[MAX_TRIALS];

int main( int argc, char** argv)
{

	
	if(argc<5) {
		fprintf(stderr,"<kernel_iterations> <blocks> <threads> f1 f2 ... f_#files\n");
		return -1;
	}
	int trials=atoi(argv[1]);
	assert(trials<=MAX_TRIALS);
	int nblocks=atoi(argv[2]);
	int nthreads=atoi(argv[3]);

	fprintf(stderr," iterations: %d blocks %d threads %d\n",trials, nblocks, nthreads);	

	int num_files=argc-1-3;
	char** d_filenames=NULL;
	

	double total_time=0;
	size_t total_size;
	
	CUDA_SAFE_CALL(cudaSetDevice(2));

	memset(time_res,0,MAX_TRIALS*sizeof(double));
for(int i=0;i<trials+1;i++){


	

	volatile GPUGlobals* gpuGlobals;
	initializer(&gpuGlobals);
	
	init_device_app();
	init_app();


	if (num_files>0){
		d_filenames=(char**)malloc(sizeof(char*)*num_files);
		for(int i=0;i<num_files;i++){
			d_filenames[i]=update_filename(argv[i+4]);
			fprintf(stderr,"file -%s\n",argv[i+4]);
		}
	}
	double time_before=_timestamp();
	if (!i) time_before=0;
	// vector, matrix, out
        bigmatrix_mmap<<<nblocks,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(d_filenames[0],d_filenames[1],d_filenames[2]);
	
	
	
	int device_num=0;
	while(true)
	{
		open_loop(gpuGlobals,2);
		rw_loop(gpuGlobals);
		if ( cudaErrorNotReady != cudaStreamQuery(gpuGlobals->streamMgr->kernelStream)) {
			fprintf(stderr,"kernel is complete\n");
			fprintf(stderr,"Max pending requests: %d\n",max_req);
			fprintf(stderr,"Transfer time: %.3f\n",transfer_time);
			transfer_time=0;
			break;
		}
	}


    cudaError_t error = cudaDeviceSynchronize();
	double time_after=_timestamp();
	if(!i) time_after=0;
	total_time+=(time_after-time_before);
	if (i>0) {time_res[i]=time_after-time_before;
		fprintf(stderr," t-%.3f-us\n",time_res[i]);
	}
	

    //Check for errors and failed asserts in asynchronous kernel launch.
    if(error != cudaSuccess )
    {
        printf("Device failed, CUDA error message is: %s\n\n", cudaGetErrorString(error));
    }
	

    //PRINT_DEBUG;

	fprintf(stderr,"\n");
	delete gpuGlobals;

	PRINT_MALLOC;
	PRINT_FREE;
	PRINT_PAGE_ALLOC_RETRIES;
	PRINT_LOCKLESS_SUCCESS;
	PRINT_WRONG_FILE_ID;

	PRINT_RT_MALLOC;
	PRINT_RT_FREE;
	PRINT_HT_MISS;
	PRINT_PRECLOSE_PUSH;
	PRINT_PRECLOSE_FETCH;
	PRINT_HT_HIT;
	PRINT_FLUSHED_READ;
	PRINT_FLUSHED_WRITE;
	PRINT_TRY_LOCK_FAILED;


//	cudaFree(d_output);	
	cudaDeviceReset();
	if(error) break;

	

}

	

	if (d_filenames) free(d_filenames);

	double thpt=post_app(total_time,trials);
	struct stat s1,s2,s3;
	if (stat(argv[4],&s1)) perror("stat failed");
	if (stat(argv[5],&s2)) perror("stat failed");
	if (stat(argv[6],&s3)) perror("stat failed");
	total_size=s1.st_size+s2.st_size+s3.st_size;
	double d_size=total_size/1024.0/1024.0/1024.0;

	
	double avg_time,avg_thpt,std_time,std_thpt;
	
	stdavg(&avg_time,&avg_thpt, &std_time, &std_thpt, time_res, d_size, MAX_TRIALS);

	fprintf(stderr,"Performance: %.3f usec +/- %.3f, %.3f GB,  %.3f GB/s +/- %.3f, FS_BLOCKSIZE %d FS_LOGBLOCKSIZE %d\n",avg_time,std_time, d_size,
				avg_thpt*1e6,std_thpt*1e6,FS_BLOCKSIZE, FS_LOGBLOCKSIZE );
//((double)output_size*(double)nblocks*(double)read_count)/(total_time/TRIALS)/1e3 );
	return 0;
}



