/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/




#ifndef MAIN_FS_FILE
#error "This file must be included in the fs.cu"
#endif
 #include <sys/mman.h>
#include "fs_constants.h"
#include "fs_calls.cu.h"
__device__ volatile INIT_LOCK init_lock;
__device__ volatile LAST_SEMAPHORE last_lock;

__shared__ float res;

__device__ void inner_product(volatile float* a, volatile float* b, int size)
{
	float tmp=0;
	__syncthreads();
	if (threadIdx.x==0) {
			res=0;
	}
	__syncthreads();
	int i=0;
	for( i=threadIdx.x;i<size;i+=blockDim.x){

		tmp+=(a[i]*b[i]);
	}
	
	atomicAdd(&res,tmp);
	__syncthreads();

}



__device__ volatile float* get_row(volatile uchar** cur_page_ptr, size_t* cur_page_offset, size_t req_file_offset, int max_file_size, int fd, int type)
{
	if (*cur_page_ptr!=NULL && *cur_page_offset+FS_BLOCKSIZE>req_file_offset) 
		return (volatile float*)(*cur_page_ptr+(req_file_offset&(FS_BLOCKSIZE-1)));

	// remap
	if (*cur_page_ptr && gmunmap(*cur_page_ptr,0)) ERROR("Unmap failed");

	int mapsize=(max_file_size-req_file_offset)>FS_BLOCKSIZE?FS_BLOCKSIZE:(max_file_size-req_file_offset);

	*cur_page_offset=(req_file_offset& (~(FS_BLOCKSIZE-1)));// round to the beg. of the page
	*cur_page_ptr=(volatile uchar*) gmmap(NULL, mapsize,0,type, fd,*cur_page_offset);
	if (*cur_page_ptr == GMAP_FAILED) ERROR("MMAP failed");

	return (volatile float*)(*cur_page_ptr+(req_file_offset&(FS_BLOCKSIZE-1)));
}
struct _pagehelper{
	volatile uchar* page;
	size_t file_offset;
};

__global__ void bigmatrix_mmap( char* f_v, char* f_m, char* f_out )
{
	__shared__ int zfd_v;
	__shared__ int zfd_m;
	__shared__ int zfd_o;

#define MB (1<<20)
	
	__shared__ int toInit;
	
	zfd_m=gopen(f_m,O_GRDONLY);
	if (zfd_m<0) ERROR("Failed to open matrix");

	zfd_o=gopen(f_out,O_GWRONCE);
	if (zfd_o<0) ERROR("Failed to open output");
		
	zfd_v=gopen(f_v,O_GRDONLY);
	if (zfd_v<0) ERROR("Failed to open vector");


	volatile float* ptr_v=(volatile float*)gmmap(NULL, fstat(zfd_v),0, O_GRDONLY, zfd_v, 0);

	if (ptr_v==GMAP_FAILED) ERROR("GMMAP failed");
	
	BEGIN_SINGLE_THREAD
		toInit=init_lock.try_wait();
	
		if (toInit == 1)
		{
			single_thread_ftruncate(zfd_o,0);
			__threadfence();
			init_lock.signal();
		}
	END_SINGLE_THREAD

	size_t size_v=fstat(zfd_v)/sizeof(float);
			
	size_t size_m=fstat(zfd_m)/sizeof(float);
			
	size_t size_o=size_m/size_v;
			
			
	
	int floats_per_page=MB/sizeof(float);
	if (size_v>floats_per_page) ERROR("Vector must be smaller than one page");
	if (floats_per_page%size_v!=0) ERROR("Vector size must divide page size");

	int rows_per_chunk;

	rows_per_chunk=size_o/gridDim.x;

	if (rows_per_chunk<1) rows_per_chunk=1;

	if (rows_per_chunk*size_v>MB/sizeof(float)) rows_per_chunk=MB/sizeof(float)/size_v; 

	
	_pagehelper ph_m={NULL,0};
	_pagehelper ph_o={NULL,0};
	

	
	size_t rows_per_block=size_o/gridDim.x;
	

	for( size_t data_idx=blockIdx.x*rows_per_block; data_idx<(blockIdx.x+1)*rows_per_block; data_idx+=rows_per_chunk)
	{

		volatile float* ptr_row_m=get_row(&ph_m.page,&ph_m.file_offset,data_idx*size_v<<2,size_m<<2,zfd_m,O_GRDONLY);
		volatile float* ptr_val_o=get_row(&ph_o.page,&ph_o.file_offset,data_idx<<2,size_o<<2,zfd_o,O_GWRONCE);
		for( int subrow=0 ;subrow<rows_per_chunk;subrow++){
			
			
			size_t _req_offset=(data_idx+subrow)<<2;

			if (_req_offset - ph_o.file_offset >= FS_BLOCKSIZE) 
				ptr_val_o=get_row(&ph_o.page,&ph_o.file_offset,_req_offset,size_v<<2,zfd_o,O_GWRONCE);
						
			_req_offset*=size_v;
			if (_req_offset - ph_m.file_offset >= FS_BLOCKSIZE) {
				ptr_row_m=get_row(&ph_m.page,&ph_m.file_offset,_req_offset,size_m<<2,zfd_m,O_GRDONLY);
			}
			
			inner_product(ptr_v,ptr_row_m, size_v);

			 *ptr_val_o=res;
			__syncthreads();

			ptr_row_m+=size_v; //move on one row, maybe its still on the page
			ptr_val_o++; // move output one value
		}
			
	}
	if (gmunmap(ph_m.page,0)) ERROR("Failed to unmap big matrix");
	if (gmunmap(ph_o.page,0)) ERROR("Failed to unmap output");
	if (gmunmap(ptr_v,0)) ERROR("Failed to unmap vector");

	gclose(zfd_m);
		
	gclose(zfd_v);

	gclose(zfd_o);
	
}

void init_device_app(){

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

double post_app(double total_time, float trials )
{
	return 0;
}
