/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/


#include "fs_calls.cu.h"
#define GREP_ROW_WIDTH (1024)

__device__ volatile INIT_LOCK init_lock;
__device__ volatile LAST_SEMAPHORE last_lock;

__shared__ float input_img_row[GREP_ROW_WIDTH];
__shared__ float input_db_row[GREP_ROW_WIDTH];

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
#define ACCUM_N 512
__shared__ volatile float s_reduction[ACCUM_N];

__device__ float inner_product( volatile float* a, volatile float* b, int size)
{
        float tmp=0;
//      __syncthreads();
//      if (threadIdx.x==0) {
//                      *res=0;
//      }
//      __syncthreads();
        int i=0;
        for( i=threadIdx.x;i<size;i+=blockDim.x){

		tmp+=(a[i]-b[i])*(a[i]-b[i]); 
        }
        s_reduction[threadIdx.x]=tmp;

         __syncthreads();
        for (int stride = ACCUM_N / 2; stride > 32; stride >>= 1)
        {
            if(threadIdx.x<stride) s_reduction[threadIdx.x] += s_reduction[stride + threadIdx.x];
            __syncthreads();
        }
        for (int stride = 32; stride > 0 && threadIdx.x<32 ; stride >>=1 )
        {
                if(threadIdx.x<stride) s_reduction[threadIdx.x] += s_reduction[stride + threadIdx.x];
        }

        __syncthreads();

        return s_reduction[0];

}

__device__ bool match(volatile float* a, volatile float* b, int size,float match_threshold){
	return sqrt(inner_product(a,b,size))<match_threshold;
}
/*
__device__ bool match(volatile float* a, volatile float* b, int size,float match_threshold)
{
        float tmp=0;
	__shared__ float res;
        __syncthreads();
        if (threadIdx.x==0) {
                        res=10;
        }
        __syncthreads();
        int i=0;
        for( i=threadIdx.x;i<size;i+=blockDim.x){

        //        assert((a[i]<1 && a[i]>0 ) );
          //      assert((b[i]<1 && b[i]>0 ) );
         //       tmp+=(a[i]-b[i])*(a[i]-b[i]); // eucledian distance
	      // assert(a[i]*b[i]<1);
        }

 	       atomicAdd(&res,tmp);
        __syncthreads();
	return sqrt(res)<match_threshold;
}
*/



void __global__ img_gpu(char* src, int src_row_len, int num_db_files, float match_threshold, int start_offset,
			 char* out, char* out2, char* out3, char* out4, char *out5, char* out6, char* out7)
{
	__shared__ int zfd_src;
	__shared__ int zfd_o;
	__shared__ char* db_files[6];
	__shared__ int* out_buffer;
	__shared__ size_t in_size;
	__shared__ int rows_per_chunk;
	__shared__ int toInit;
	__shared__ int rows_to_process;
	__shared__ int total_rows;
	src_row_len=GREP_ROW_WIDTH;
		db_files[0]=out2;
		db_files[1]=out3;
		db_files[2]=out4;
		db_files[3]=out5;
		db_files[4]=out6;
		db_files[5]=out7;
	
		zfd_o=gopen(out,O_GWRONCE);
		if (zfd_o<0) ERROR("Failed to open output");
		
		zfd_src=gopen(src,O_GRDONLY);
		if (zfd_src<0) ERROR("Failed to open input");
	
		in_size=fstat(zfd_src);
		total_rows=in_size/src_row_len>>2;

		rows_per_chunk=total_rows/gridDim.x;
		if (rows_per_chunk==0) rows_per_chunk=1;
		
		rows_to_process=rows_per_chunk;

		if (blockIdx.x==gridDim.x-1) rows_to_process=(total_rows - blockIdx.x*rows_per_chunk);
	
	BEGIN_SINGLE_THREAD
		out_buffer=(int*)malloc(rows_to_process*sizeof(int)*3);
		toInit=init_lock.try_wait();

		if (toInit == 1)
		{
			single_thread_ftruncate(zfd_o,0);
			__threadfence();
			init_lock.signal();
		}
	END_SINGLE_THREAD

	/*
	1. decide how many strings  each block does
	2. map input line
	3. map db
	4. scan through 
	5. write to output
	*/

	

	_pagehelper ph_input={NULL,0};
	_pagehelper ph_db={NULL,0};
	
	int out_count=0;
	volatile float* ptr_row_in=NULL;
	int found=0;
	int start=blockIdx.x*rows_per_chunk;

	for (size_t data_idx=blockIdx.x*rows_per_chunk ; data_idx<start+rows_to_process; data_idx++,out_count+=3)
	{
		found=0;
		//ptr_row_in=get_row(&ph_input.page,&ph_input.file_offset,
		//				   data_idx*src_row_len<<2,in_size<<2,zfd_src,O_GRDONLY);
		
		int bytes_read=gread(zfd_src,data_idx*src_row_len<<2,GREP_ROW_WIDTH*4,(uchar*)input_img_row);
		if (bytes_read!=GREP_ROW_WIDTH*4) ERROR("Failed to read src");

		for( int db_idx=0;db_idx<num_db_files;db_idx++ )
		{
			
			int zfd_db;
			zfd_db=gopen(db_files[db_idx],O_GRDONLY);
			if (zfd_db<0) ERROR("Failed to open DB file");
			size_t db_rows=(fstat(zfd_db)/src_row_len)>>2;
			
			volatile float* ptr_row_db=get_row(&ph_db.page,&ph_db.file_offset,0,fstat(zfd_db),zfd_db,O_GRDONLY);
			for (int _cursor=0;_cursor<db_rows;_cursor++,ptr_row_db+=src_row_len)
			{

				size_t _req_offset=(_cursor*src_row_len)<<2;
				if (_req_offset - ph_db.file_offset >= FS_BLOCKSIZE) {
					ptr_row_db=get_row(&ph_db.page,&ph_db.file_offset,_req_offset,fstat(zfd_db),zfd_db,O_GRDONLY);
				}
				found=match(input_img_row,ptr_row_db,src_row_len,match_threshold);
				//found=match(ptr_row_in,ptr_row_db,src_row_len,match_threshold);
				BEGIN_SINGLE_THREAD
					if (found){
						out_buffer[out_count]=data_idx+start_offset*total_rows;
						out_buffer[out_count+1]=db_idx;
						out_buffer[out_count+2]=_cursor;
					}
				END_SINGLE_THREAD
				if (found) break;
			}
			if(gmunmap(ptr_row_db,0)) ERROR("Failed to unmap db");
			ph_db.page=NULL; ph_db.file_offset=0;
			
			gclose(zfd_db);
		
			if (found) break;
		}
		if (!found){
			BEGIN_SINGLE_THREAD
				out_buffer[out_count]=data_idx+start_offset*total_rows;
				out_buffer[out_count+1]=-1;
				out_buffer[out_count+2]=-1;
			END_SINGLE_THREAD
		}
	}
	//we are done.
	//write the output and finish
	//if (gmunmap(ptr_row_in,0)) ERROR("Failed to unmap input");
	int write_size=rows_to_process*sizeof(int)*3;
	if (gwrite(zfd_o,blockIdx.x*rows_per_chunk*sizeof(int)*3,write_size,(uchar*)out_buffer)!=write_size) ERROR("Failed to write output");
	
        gclose(zfd_src);
	BEGIN_SINGLE_THREAD
		free(out_buffer);
        END_SINGLE_THREAD
        gclose(zfd_o);
}




void init_device_app(){
//	CUDA_SAFE_CALL(cudaSetDevice(global_devicenum));
      CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<25));
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
        //return  sizeof(float)*VEC_FLOAT*((double)VEC_FLOAT)*2/ (total_time/trials);
}

