/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/


#include "util.cu.h"
#include <sys/mman.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>


#include <sys/time.h>

      double _timestamp(){
                struct timeval tv;
                gettimeofday(&tv,0);
                return 1e6*tv.tv_sec+tv.tv_usec;
        }


__forceinline__ __device__ void memcpy_thread(volatile char* dst, const volatile char* src, uint size)
{
	for( int i=0;i<size;i++)
		dst[i]=src[i];
}



//#define alpha(src)      (((src)>=65 && (src)<=90)||( (src)>=97 && (src)<=122)|| (src)==95 || (src)==39)
#define alpha(src)      (((src)>=65 && (src)<=90)||( (src)>=97 && (src)<=122)|| (src)==95)
#define INPUT_PREFETCH_ARRAY (128*33)
#define INPUT_PREFETCH_SIZE (128*32)

#define CORPUS_PREFETCH_SIZE (16384)
__shared__ char input[INPUT_PREFETCH_ARRAY];
__shared__ char corpus[CORPUS_PREFETCH_SIZE+32+1]; // just in case we need the leftovers

		
__device__ void bzero_block(char*dst, int total_buf){
	for( int i=threadIdx.x;i<total_buf;i+=blockDim.x) dst[i]=0;
}
__device__ void prefetch_banks(char *dst, char *src, int data_size, int total_buf)
{
	__syncthreads();
	int i=0;

	for(i=threadIdx.x;i<data_size;i+=blockDim.x)
	{
		int offset=(i>>5)*33+(i&31);
		dst[offset]=src[i];
	}
	for(;i<total_buf;i+=blockDim.x) {
		int offset=(i>>5)*33+(i&31);
		dst[offset]=0;
	}
	__syncthreads();
}

__device__ void prefetch(char *dst, char *src, int data_size)
{
	__syncthreads();
	int i=0;
	for(i=threadIdx.x;i<data_size;i+=blockDim.x)
	{
		dst[i]=src[i];
	}
	__syncthreads();
}
#define WARP_COPY(dst,src) (dst)[threadIdx.x&31]=(src)[threadIdx.x&31];
#define LEN_ZERO (-1)
#define NO_MATCH 0
#define MATCH  1


__device__ int match_string( char* a, char*data, int data_size, char* wordlen)
{
	int matches=0;
	char sizecount=0;
	char word_start=1;
	if (*a==0) return -1;
	
	for(int i=0;i<data_size;i++)
	{
		if (!alpha(data[i])) { 
			if ((sizecount == 32 || a[sizecount]=='\0' ) && word_start ) { matches++; *wordlen=sizecount;}
			word_start=1;
			sizecount=0;
		}else{

			if (a[sizecount]==data[i]) { sizecount++; }
			else {	word_start=0;	sizecount=0;}
		}
	}

	return matches;
}

#define ROW_SIZE (128*32)
#define PREFETCH_SIZE 16384

__device__ int global_output;
__shared__ int output_count;
__shared__ char* output_buffer;



__device__ int find_overlap(char* dst)
{
	__shared__ int res;
	if(threadIdx.x==0){
		res=0;
		int i=0;
		for(;i<32&&alpha(dst[i]);i++);
		res=i;
	}
	__syncthreads();
	return res;
	
}
		
__device__ char* get_next(char* data, size_t* filesizes, int file_counter, int& last_offset)
{
	if (file_counter==0) return data;
	char* newptr=data+last_offset+filesizes[file_counter-1];
	last_offset+=filesizes[file_counter-1];
	return newptr;
}

void __global__ grep_text_nofiles(char* src, int total_words, char* out, char* data,  size_t* filesizes, int num_files )
{

	int data_to_process=0;

	int words_per_chunk=total_words/gridDim.x;

	if (words_per_chunk==0) 
	{
		words_per_chunk=1;
		if (blockIdx.x>total_words) {
			words_per_chunk=0;
		}
	}
		
			

	if (blockIdx.x==gridDim.x-1){
		data_to_process=32*(total_words-words_per_chunk*blockIdx.x);
	}else{
		data_to_process=32*words_per_chunk;
	}
	if(threadIdx.x==0){
		output_buffer=(char*)malloc(data_to_process/32*(32+3*sizeof(int)));
		assert(output_buffer);
		output_count=0;
	}
	__syncthreads();


	// for every DB
	// 	read the input
	//	for every DB block as long as the input buffer is not empty
	//		for every input block
	// 			match the block
	//			throw away matched 
	//			put unmatched into the other buffer
	//		done
	//		switch input buffers
	//	done
	//done
	int db_idx=0;
	int global_offset=0;

	for(db_idx=0;db_idx<num_files;db_idx++)
	{
		char* filecontent=get_next(data,filesizes,db_idx,global_offset);
		size_t db_size=filesizes[db_idx];

		for (size_t _cursor=0;_cursor< db_size;)
		{
			bool last_iter=db_size-_cursor<(CORPUS_PREFETCH_SIZE+32);
			int db_left=last_iter?db_size-_cursor: CORPUS_PREFETCH_SIZE+32;
			
			corpus[db_left]='\0';
			prefetch(corpus,filecontent+_cursor,db_left);


			// take care of the stitches
			int overlap=0;
			
			if(!last_iter){
				overlap=find_overlap(corpus+CORPUS_PREFETCH_SIZE);
				_cursor+=overlap;
			}
			_cursor+=CORPUS_PREFETCH_SIZE;
	
			///////////////////// NOW WE ARE DEALING WITH THE INPUT							
			//
			// indexing is in chars, not in row size
			char* input_tmp=src+blockIdx.x*words_per_chunk*32;				
			for(int input_block=0;input_block<data_to_process;input_block+=INPUT_PREFETCH_SIZE){

				int data_left=data_to_process-input_block;
				prefetch_banks(input,input_tmp + input_block,min(data_left,INPUT_PREFETCH_SIZE),INPUT_PREFETCH_SIZE);
				
				char word_size=0;				
				int res= match_string(input+threadIdx.x*33,corpus,last_iter?db_left+1:CORPUS_PREFETCH_SIZE+overlap+1,&word_size);


				if (!__syncthreads_or(res!=LEN_ZERO && res )) continue;
				if(res!=LEN_ZERO && res){
						char tmp_add=(~(word_size&3)+1)&3; // make sure int assignment will be warp-aligned on 4.
						int offset=atomicAdd(&output_count,(word_size+tmp_add+3*sizeof(int)));
						char* outptr=output_buffer+offset;

						*(int*)(outptr)=word_size;
						*(int*)(outptr+4)=res;
						*(int*)(outptr+8)=db_idx;	
						memcpy_thread(outptr+12,input_tmp+input_block+threadIdx.x*32,word_size);
				}
				__syncthreads();
					
				if (output_count){
					__shared__ int old_offset;
					if (threadIdx.x==0) old_offset=atomicAdd(&global_output,output_count);
					__syncthreads();
					prefetch(out+old_offset,output_buffer,output_count);
				}
				__syncthreads();

				/// how many did we find
				if(threadIdx.x==0){ 
					output_count=0;
				}
				__syncthreads();
			}
		}
	}
	//we are done.
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

#include <omp.h>


size_t* file_sizes;
char** file_names;
 
#define BUF_SIZE (1024*1024*1024)
	
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


int print_line(char* output, char** files){
	
	int word_size=*(int*)(output);
	int res=*(int*)(output+4);
	int idx=*(int*)(output+8);
	
	char tmp_add=(~((word_size&3))+1)&3;
	for(int i=0;i<word_size;i++){
		printf("%c",output[i+12]);
	}
	printf(" %d %s\n",res,files[idx]);
	return word_size+tmp_add+12;
}
		
		
int main(int argc, char** argv){
	
	double time_before=_timestamp();
	file_sizes=new size_t[60000];
	file_names=new char*[60000];
	char* output=new char[BUF_SIZE];

	for(int i=0;i<60000;i++){
		file_names[i]=new char[128];
		file_names[i][0]=0;
	}
	
	int fd_words=open(argv[1],O_RDWR);
	assert(fd_words>=0);

	struct stat s;
	fstat(fd_words,&s);
	if (s.st_size<=0) assert(NULL);
	int words_size=s.st_size;


	FILE* fd_files=fopen(argv[3],"r");
		char* words=(char*)mmap(NULL, words_size,PROT_READ,MAP_SHARED|MAP_POPULATE,fd_words,0);
		perror("mmap");
		assert(words!=MAP_FAILED);
	
	
	char* line=(char*)malloc(256);
	size_t n=256;


	char* all_bufs[1000];
	memset(all_bufs,0,1000);

	char* buf=all_bufs[0]=new char[BUF_SIZE];
	
	int cur_file=0;

	int cur_buf=0;

	int cur_size=0;

	while ( getline(&line, &n,fd_files)>0){
		if (strlen(line)<4) continue;
		line[strlen(line)-1]='\0';
		

		int fd_src=open(line,O_RDONLY);
		if (fd_src<0 ) perror("open");
		assert(fd_src>=0);
	
	
		fstat(fd_src,&s);
		if (s.st_size<=0) assert(NULL);
		int src_size=s.st_size;
		file_sizes[cur_file]=src_size;
		strcpy(file_names[cur_file],line);
		cur_file++;
	
		char* src=(char*)mmap(NULL, src_size,PROT_READ,MAP_SHARED|MAP_POPULATE,fd_src,0);
		if(src==MAP_FAILED) perror("mmap");
		assert(src!=MAP_FAILED);
	
		if (BUF_SIZE-cur_size> src_size)
		{
			memcpy(buf+cur_size,src,src_size);
			cur_size+=src_size;
			
		}else{
			memcpy(buf+cur_size,src,BUF_SIZE-cur_size);
			cur_buf++;
			all_bufs[cur_buf]=new char[BUF_SIZE];
			buf=all_bufs[cur_buf];
			memcpy(buf,src+BUF_SIZE-cur_size,BUF_SIZE-cur_size);
			cur_size=src_size-(BUF_SIZE-cur_size);
		}
		
			
		munmap(src,src_size);
		close(fd_src);
	}
		
		
	char* d_content,*d_output,*d_words;
	size_t *d_sizes;

	setupGpu(&d_content, buf, cur_size,
                 &d_sizes, file_sizes,&d_output, cur_file,
                &d_words, words, words_size/32);
	

	grep_text_nofiles<<<28,128,0,0>>>(d_words, words_size/32, d_output, d_content,  d_sizes, cur_file );
	
	cudaError_t error = cudaDeviceSynchronize();
	
	if(error != cudaSuccess )
	    {
        	printf("Device failed, CUDA error message is: %s\n\n", cudaGetErrorString(error));
	    }
	CUDA_SAFE_CALL(cudaMemcpy(output,d_output,BUF_SIZE,cudaMemcpyDeviceToHost));
	void* global_output_ptr;
	int total_out_size;
	CUDA_SAFE_CALL(cudaGetSymbolAddress(&global_output_ptr,global_output));
	CUDA_SAFE_CALL(cudaMemcpy(&total_out_size,global_output_ptr, sizeof(int),cudaMemcpyDeviceToHost));

	for(int i=0;i<total_out_size;)
	{	
			i+=print_line(output+i,file_names);
	}
	double time_after=_timestamp();
	printf("Total time: %.0f\n",time_after-time_before);
	
}



