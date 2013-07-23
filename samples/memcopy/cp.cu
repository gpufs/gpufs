/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/



#ifndef MAIN_FS_FILE
#error "This file must be included into main CU file. It cannot be used separately"
#endif


__shared__ int zfd;

#define ONE_BLOCK_READ (1<<26 )

__device__ LAST_SEMAPHORE sync_sem;
__global__ void test_cpy(char* src)
{
        __shared__ uchar* scratch;
        BEGIN_SINGLE_THREAD;
                zfd=0;
                zfd=single_thread_open(src,O_GRDONLY);
		if (zfd<0) ERROR("Cant open file");
               /* 
                zfd1=0;
                zfd1=single_thread_open(dst,O_GWRONCE);
                if (zfd1<0) { atomicMin(&OK,-2); ERROR("Failed to open dst");}
                */
                //scratch=(uchar*)malloc(1<<20);
                //GPU_ASSERT(scratch!=NULL);
                
        END_SINGLE_THREAD;

        int filesize=fstat(zfd);

        for(size_t me=0; me< ONE_BLOCK_READ; me+=FS_BLOCKSIZE)
	{
		int my_offset=blockIdx.x*ONE_BLOCK_READ;

                int toRead=min((unsigned int)FS_BLOCKSIZE,(unsigned int)(filesize-me-my_offset));
		assert(toRead);
        	volatile void* p=gmmap(NULL, toRead,0,O_GRDONLY,zfd,my_offset+me);
		if (p==MAP_FAILED) ERROR("MAP FAILED");
//		gmunmap(p,0);
        }

/*
        BEGIN_SINGLE_THREAD;
        	if (sync_sem.is_last()) single_thread_fsync(zfd1);
                close_ret=single_thread_close(zfd);
                if (close_ret!=0) {atomicMin(&OK,-5); }
                close_ret=single_thread_close(zfd1);
                if (close_ret!=0) {atomicMin(&OK,-6); }
                free(scratch);
        END_SINGLE_THREAD;
  */
	BEGIN_SINGLE_THREAD
		single_thread_close(zfd);
	END_SINGLE_THREAD              
}
void init_device_app(){

        CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<30));
}

void init_app()
{
	// INITI LOCK   
	void* inited;

/*        CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,init_lock));
        CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(INIT_LOCK)));
	
	CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,last_lock));
        CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(LAST_SEMAPHORE)));
*/
}

double post_app(double time, int trials){
   //   int res;
//      CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&res,OK,sizeof(int),0,cudaMemcpyDeviceToHost));
  //    if(res!=0) fprintf(stderr,"Test Failed, error code: %d \n",res);
//      else  fprintf(stderr,"Test Success\n");
 
     return 0;
}

