
#include "fs_calls.cu.h"
__device__ int OK;
__shared__ int zfd,zfd1, zfd2, close_ret;

__device__ LAST_SEMAPHORE sync_sem;
__global__ void test_cpy(char* src, char* dst)
{
	__shared__ uchar* scratch;
	__shared__ size_t filesize;
	BEGIN_SINGLE_THREAD
	        scratch=(uchar*)malloc(FS_BLOCKSIZE);
                GPU_ASSERT(scratch!=NULL);
	END_SINGLE_THREAD


       zfd=0;
       zfd=gopen(src,O_GRDONLY);
	
                
       zfd1=0;
       zfd1=gopen(dst,O_GWRONCE);
	filesize=fstat(zfd);


        for(size_t me=blockIdx.x*FS_BLOCKSIZE;me<filesize;me+=FS_BLOCKSIZE*gridDim.x){
                int toRead=min((unsigned int)FS_BLOCKSIZE,(unsigned int)(filesize-me));
                if (toRead!=gread(zfd,me,toRead,scratch)){
			assert(NULL);
		}
                
                if (toRead!=gwrite(zfd1,me,toRead,scratch)){
			assert(NULL);
                }
        
        }


	
	gclose(zfd);
        gclose(zfd1);
                
}
void init_device_app(){

        CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<30));
}

void init_app()
{
      void* d_OK;
      CUDA_SAFE_CALL(cudaGetSymbolAddress(&d_OK,OK));
      CUDA_SAFE_CALL(cudaMemset(d_OK,0,sizeof(int)));
	// INITI LOCK   
	void* inited;

	
	CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited,sync_sem));
        CUDA_SAFE_CALL(cudaMemset(inited,0,sizeof(LAST_SEMAPHORE)));
}

double post_app(double time, int trials){
      int res;
      CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&res,OK,sizeof(int),0,cudaMemcpyDeviceToHost));
      if(res!=0) fprintf(stderr,"Test Failed, error code: %d \n",res);
      else  fprintf(stderr,"Test Success\n");
 
     return 0;
}

