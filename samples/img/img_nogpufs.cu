

#define GREP_ROW_WIDTH (4*1024)
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

__device__ int matched_data;

void __global__ img_gpu_nofs(float* src, int src_len,
			 float *db, size_t* db_len, 
			 float match_threshold,  
			 char* outbuf, int* matched_entries )
{
	int total_rows=src_len>>2;

	int rows_per_chunk=total_rows/gridDim.x;
	if (rows_per_chunk==0) rows_per_chunk=1;
		
	int rows_to_process=rows_per_chunk;

	if (blockIdx.x==gridDim.x-1) rows_to_process=(total_rows - blockIdx.x*rows_per_chunk);
	

	
	int out_count=sizeof(int)*3;
	int found=0;
	int start=blockIdx.x*rows_per_chunk;
	int matched_count=0;
	for ( size_t _cursor=0;_cursor<db_len;_cursor+=(GREP_ROW_WIDTH<<2)){
		
		prefetch(db_line,db+_cursor,(GREP_ROW_WIDTH<<2));
			
		for (size_t data_idx=start ; data_idx<start+rows_to_process; data_idx++,out_count+=3)
		{
			found=0;
			
			found=match(db_line, src+data_idx*GREP_ROW_WIDTH,GREP_ROW_WIDTH,match_threshold);
			BEGIN_SINGLE_THREAD
			if (found){
				out_buffer[out_count]=data_idx;
				matched_entries[matched_count++]=data_idx;
				out_buffer[out_count+1]=0;
				out_buffer[out_count+2]=_cursor;
			}
			END_SINGLE_THREAD
			if (found) break;
		}
		if (!found){
			BEGIN_SINGLE_THREAD
				out_buffer[out_count]=data_idx;
				out_buffer[out_count+1]=-1;
				out_buffer[out_count+2]=-1;
			END_SINGLE_THREAD
		}
	}
}


#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
       #include <sys/stat.h>
       #include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

 #include <stdio.h>

void setupGpu(char** d_content, char* h_content, size_t content_size,
                size_t **d_sizes, size_t* h_sizes,
                char** d_output, int num_files,
                char** d_words, char* h_words, int num_words ){

                CUDA_SAFE_CALL(cudaMalloc(d_content,content_size));
                CUDA_SAFE_CALL(cudaMalloc(d_output,BUF_SIZE));
                CUDA_SAFE_CALL(cudaMalloc(d_sizes,num_files*sizeof(size_t)));
                CUDA_SAFE_CALL(cudaMemcpy(*d_content, h_content, content_size,cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(*d_sizes, h_sizes, num_files*sizeof(size_t),cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemset(*d_output,0, BUF_SIZE));

                CUDA_SAFE_CALL(cudaMalloc(d_words,num_words*32));
                CUDA_SAFE_CALL(cudaMemcpy(*d_words,h_words,num_words*32,cudaMemcpyHostToDevice));


                CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<30));

                void* global_output_ptr;

                CUDA_SAFE_CALL(cudaGetSymbolAddress(&global_output_ptr,global_output));
                CUDA_SAFE_CALL(cudaMemset(global_output_ptr,0,sizeof(int)));

}

#define MAX_BUF_SIZE (1<<31L)

int main(int argc, char** argv){
	if(argc<7) {
		fprintf(stderr,"<src_in><out><db_1><db_2><db_3><threshold>\n");
		return -1;
	}

	int src_fd=open(argv[1],O_RDONLY);
	assert(src_fd>=0);
	stat s;
	fstat(src_fd,&s);
	size_t src_size=s.st_size;

	int out_fd=open(argv[2],O_WRONLY,O_CREAT|O_TRUNC,S_IRUSR|S_IWUSR);
	assert(out_fd>=0);

	int db_fds[3];
	int db_sizes[3];

	int db1_fd=open(argv[3],O_RDONLY);
	assert(db1_fd>=0);
	fstat(db1_fd,&s);
	size_t db1_size=s.st_size;
	db_fds[0]=db1_fd;
	db_sizes[0]=db1_size;


	int db2_fd=open(argv[4],O_RDONLY);
	assert(db2_fd>=0);
	fstat(db2_fd,&s);
	size_t db2_size=s.st_size;
	db_fds[1]=db2_fd;
	db_sizes[1]=db2_size;

	int db3_fd=open(argv[5],O_RDONLY);
	assert(db3_fd>=0);
	fstat(db3_fd,&s);
	size_t db3_size=s.st_size;
	db_fds[2]=db1_fd;
	db_sizes[2]=db1_size;

	float match_threshold=strtof(argv[6],NULL);
	fprintf(stderr, "Match threshold %f\n",match_threshold);

/////////////////////////

	size_t total_size=db1_size+db2_size+db3_size;

	size_t buf_size= min(MAX_BUF_SIZE,total_size);
	
	char* d_db;
	CUDA_SAFE_CALL(cudaMalloc(&d_db,buf_size));
	
	char* h_d_db;
	CUDA_SAFE_CALL(cudaHostAlloc(&h_d_db, buf_size,  cudaHostAllocDefault));

	char* d_src,*h_d_src;
	CUDA_SAFE_CALL(cudaMalloc(&d_src,src_size));
        CUDA_SAFE_CALL(cudaHostAlloc(&h_d_src, src_size,  cudaHostAllocDefault));

	char* d_matched, h_d_matched;
	CUDA_SAFE_CALL(cudaMalloc(&d_matched,src_size/GREP_ROW_WIDTH));
        CUDA_SAFE_CALL(cudaHostAlloc(&h_d_matched, src_size/GREP_ROW_WIDTH,  cudaHostAllocDefault));


	char* d_out, *h_out;
	CUDA_SAFE_CALL(cudaMalloc(&d_out,3*src_size/GREP_ROW_WIDTH));
	h_out=(char*)malloc(3*src_size/GREP_ROW_WIDTH);
	
	size_t buf_size_left=buf_size;
	int cur_db=0;
	size_t cur_db_left=db_sizes[0];

	
	if (src_size!=read(src_fd,d_src,src_size)){ perror("Failed to read src"); return -1;}
	CUDA_SAFE_CALL(cudaMemcpy(d_src

	for( size_t i=0;i<total_size;)
	{
		if (buf_size_left>=cur_db_left){
			if(cur_db_left!=read(db_fds[cur_db],d_src+i,cur_db_left)) { perror("Failed to read"); return -1;}
			buf_size_left-=cur_db_left;
			i+=cur_db_left;
			cur_db++;
		}else{
			if(buf_size_left!=read(db_fds[cur_db],d_src+i,buf_size_left)) { perror("Failed to read"); return -1;}
		}
		
		CUDA_SAFE_CALL(cudaMemset(d_matched, src_size/GREP_ROW_WIDTH));

		CUDA_SAFE_CALL(cudaMemcpu(
	
	}
	

}
