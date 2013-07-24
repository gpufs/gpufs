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

#include "fs_debug.cu.h"
#include "fs_initializer.cu.h"

// INCLUDING CODE INLINE - change later
#include "host_loop.h"
//



#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

#define MAIN_FS_FILE
#include "grep_text_gpu.cu"


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

#define TRIALS 1.0
int main( int argc, char** argv)
{

	
	if(argc<5) {
		fprintf(stderr,"<kernel_iterations> <blocks> <threads> f1 f2 ... f_#files\n");
		return -1;
	}
	int trials=atoi(argv[1]);
	int nblocks=atoi(argv[2]);
	int nthreads=atoi(argv[3]);

	fprintf(stderr," iterations: %d blocks %d threads %d\n",trials, nblocks, nthreads);	

	int num_files=argc-1-3;
	char** d_filenames=NULL;
	

	double total_time=0;
//	int scratch_size=128*1024*1024*4;

for(int i=1;i<trials+1;i++){


	
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

        grep_text<<<nblocks,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(d_filenames[0],d_filenames[1],d_filenames[2]);
	
	
	run_gpufs_handler(gpuGlobals,0);



    cudaError_t error = cudaDeviceSynchronize();
	double time_after=_timestamp();
	if(!i) time_after=0;
	total_time+=(time_after-time_before);

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

	fprintf(stderr,"Performance: %.3f usec FS_BLOCKSIZE %d FS_LOGBLOCKSIZE %d\n",total_time/trials,FS_BLOCKSIZE, FS_LOGBLOCKSIZE );
//((double)output_size*(double)nblocks*(double)read_count)/(total_time/TRIALS)/1e3 );
	return 0;
}



