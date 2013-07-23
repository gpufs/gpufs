/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/


#include "fs_constants.h"
#include "util.cu.h"
#include "fs_calls.cu.h"
#include <sys/mman.h>
#include <stdio.h>


__device__ volatile INIT_LOCK init_lock;
__device__ volatile LAST_SEMAPHORE last_lock;

__shared__ char int_to_char_map[10];
__device__ void init_int_to_char_map()
{
	int_to_char_map[0]='0';
	int_to_char_map[1]='1';
	int_to_char_map[2]='2';
	int_to_char_map[3]='3';
	int_to_char_map[4]='4';
	int_to_char_map[5]='5';
	int_to_char_map[6]='6';
	int_to_char_map[7]='7';
	int_to_char_map[8]='8';
	int_to_char_map[9]='9';
}
	
__device__ void print_uint(char* tgt, int input, int *len){
        if (input<10) {tgt[0]=int_to_char_map[input]; tgt[1]=0; *len=1; return;}
        char count=0;
        while(input>0)
        {
                tgt[count]=int_to_char_map[input%10];
                count++;
                input/=10;
        }
        *len=count;
        count--;
        char reverse=0;
        while(count>0)
        {
                char tmp=tgt[count];
                tgt[count]=tgt[reverse];
                count--;
                tgt[reverse]=tmp;
                reverse++;
        }
}


__device__ volatile char* get_row(volatile uchar** cur_page_ptr, size_t* cur_page_offset, size_t req_file_offset, int max_file_size, int fd, int type)
{
        if (*cur_page_ptr!=NULL && *cur_page_offset+FS_BLOCKSIZE>req_file_offset)
                return (volatile char*)(*cur_page_ptr+(req_file_offset&(FS_BLOCKSIZE-1)));

        // remap
        if (*cur_page_ptr && gmunmap(*cur_page_ptr,0)) ERROR("Unmap failed");

        int mapsize=(max_file_size-req_file_offset)>FS_BLOCKSIZE?FS_BLOCKSIZE:(max_file_size-req_file_offset);

        *cur_page_offset=(req_file_offset& (~(FS_BLOCKSIZE-1)));// round to the beg. of the page
        *cur_page_ptr=(volatile uchar*) gmmap(NULL, mapsize,0,type, fd,*cur_page_offset);
        if (*cur_page_ptr == GMAP_FAILED) ERROR("MMAP failed");

        return (volatile char*)(*cur_page_ptr+(req_file_offset&(FS_BLOCKSIZE-1)));
}
struct _pagehelper{
        volatile uchar* page;
        size_t file_offset;
};

//#define alpha(src)      (((src)>=65 && (src)<=90)||( (src)>=97 && (src)<=122)|| (src)==95 || (src)==39)
#define alpha(src)      (((src)>=65 && (src)<=90)||( (src)>=97 && (src)<=122)|| (src)==95)
#define INPUT_PREFETCH_ARRAY (128*33)
#define INPUT_PREFETCH_SIZE (128*32)

#define CORPUS_PREFETCH_SIZE (16384)
__shared__ char corpus[CORPUS_PREFETCH_SIZE+32]; // just in case we need the leftovers
__shared__ char input[INPUT_PREFETCH_ARRAY];

__device__ int prefetch_word(char* dst,volatile char *src){
	__syncthreads();
	__shared__ int res;
	if (!(threadIdx.x>>5)){
		dst[threadIdx.x]=src[threadIdx.x];
		int v=__clz(__ballot(!alpha(dst[threadIdx.x])));
	//	if (threadIdx.x>=v) src[threadIdx.x]='\0';
		if(threadIdx.x==0) res=v;
	}
	__syncthreads();
	return res;
}
	
		

__device__ void prefetch_banks(char *dst, volatile char *src, int data_size, int total_buf)
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

__device__ void prefetch(char *dst, volatile char *src, int data_size, int total_buf)
{
	__syncthreads();
	int i=0;
	for(i=threadIdx.x;i<data_size;i+=blockDim.x)
	{
		dst[i]=src[i];
	}
	for(;i<total_buf;i+=blockDim.x) dst[i]=0;
	__syncthreads();
}
__device__ bool match(volatile char a, char* data, bool disabled, int data_size, int myId)
{
	//warp based approach
	int found=0;
	//for(int i=myId;i<data_size-(slen-myId);i++)
	for(int i=myId;i<data_size+myId;i++)
	{
		// all values are ok - match
		found = __all(disabled ||(a==data[i]));
		if (found) return found;
	}
	return false;
}
__device__ uchar strlen32(const char* str )
{
	int vote=__ballot(str[threadIdx.x&31]!=0);
	return 32-__clz(vote);
}
#define WARP_COPY(dst,src) (dst)[threadIdx.x&31]=(src)[threadIdx.x&31];
#define LEN_ZERO (-1)
#define NO_MATCH 0
#define MATCH  1

__device__ int match_string( char* a, char*data, int data_size, char* wordlen)
{
	int matches=0;
	char sizecount=0;
	char word_start=0;
	if (*a==0) return -1;
	
	for(int i=0;i<data_size;i++)
	{
		if (!alpha(data[i])) { 
			if ((sizecount == 32 || a[sizecount]=='\0') && word_start ) { matches++; *wordlen=sizecount;}
			word_start=1;
			sizecount=0;
		}else{

			if (a[sizecount]==data[i]) { sizecount++; }
			else {	word_start=0;	sizecount=0;}
		}
	}
	return matches;
}

__shared__ char current_db_name[FILENAME_SIZE+1];
__device__ char* get_next(char* str, char** next, int* db_strlen){
	__shared__ int beg;
	__shared__ int i;
	char db_name_ptr=0;
	if (str[0]=='\0') return NULL;

	BEGIN_SINGLE_THREAD
	beg=-1;
	for(i=0; (str[i]==' '||str[i]=='\t'||str[i]==','||str[i]=='\r'||str[i]=='\n');i++);
	beg=i; 
	for(;str[i]!='\n' && str[i]!='\r' && str[i]!='\0' && str[i]!=',' && i<64 ;i++,db_name_ptr++)
		current_db_name[db_name_ptr]=str[i];

	current_db_name[db_name_ptr]='\0';
	*db_strlen=i-beg;

	END_SINGLE_THREAD

	if (i-beg==64) return NULL;
	if (i-beg==0) return NULL;
	
//	str[i]='\0';
	*next=&str[i+1];
	return current_db_name;
}
#define ROW_SIZE (128*32)
#define PREFETCH_SIZE 16384

__device__ int global_output;
__shared__ int output_count;
void __global__ grep_text(char* src, char* out, char* dbs)
{
	__shared__ int zfd_src;
	__shared__ int zfd_o;
	__shared__ int zfd_dbs;
	//__shared__ volatile char* db_files;
	__shared__ char* db_files;
	__shared__ size_t in_size;
	__shared__ int words_per_chunk;
	__shared__ int toInit;
	__shared__ int data_to_process;
	__shared__ char* input_tmp;
	__shared__ int input_tmp_counts;
	__shared__ char *output_buffer;

	int total_words;
	BEGIN_SINGLE_THREAD

		init_int_to_char_map();
	
		zfd_dbs=single_thread_open(dbs,O_GRDONLY);
		if (zfd_dbs<0) ERROR("Failed to open output");


		zfd_o=single_thread_open(out,O_GWRONCE);
		if (zfd_o<0) ERROR("Failed to open output");
		
		zfd_src=single_thread_open(src,O_GRDONLY);
		if (zfd_src<0) ERROR("Failed to open input");
	
		in_size=fstat(zfd_src);
	
		total_words=in_size/32;
		
		if (total_words==0) ERROR("empty input");

		
		words_per_chunk=total_words/gridDim.x;

		if (words_per_chunk==0) 
		{
			words_per_chunk=1;
			if (blockIdx.x>total_words) {
				words_per_chunk=0;
			}
		}
		
	END_SINGLE_THREAD

		if (words_per_chunk==0) {
			BEGIN_SINGLE_THREAD
				single_thread_close(zfd_o);
				single_thread_close(zfd_src);
			END_SINGLE_THREAD
			return;
		}

	BEGIN_SINGLE_THREAD
			
		data_to_process=words_per_chunk*32;
	
		if (blockIdx.x==gridDim.x-1)  data_to_process=fstat(zfd_src)-data_to_process*blockIdx.x;
		
		input_tmp=(char*)malloc(data_to_process);
		assert(input_tmp);
		output_buffer=(char*)malloc(data_to_process/32*(32+FILENAME_SIZE+sizeof(int)));
		assert(output_buffer);
		output_count=0;

		db_files=(char*) malloc(3*1024*1024);
		assert(db_files);

		toInit=init_lock.try_wait();
	
	END_SINGLE_THREAD
	if (toInit == 1)
	{
		BEGIN_SINGLE_THREAD
	

			global_output=0;
			single_thread_ftruncate(zfd_o,0);
			__threadfence();
			init_lock.signal();
		END_SINGLE_THREAD
	}

	
	

	int db_bytes_read=gread(zfd_dbs,0,fstat(zfd_dbs),(uchar*)db_files);
	if(db_bytes_read!=fstat(zfd_dbs)) ERROR("Failed to read dbs");
	
	char* current_db;
	char* next_db;

	int db_idx=-1;
		
	int to_read=min(data_to_process,(int)fstat(zfd_src));
	int bytes_read=gread(zfd_src,blockIdx.x*words_per_chunk*32,to_read,(uchar*)input_tmp);
	if (bytes_read!=to_read) ERROR("FAILED to read input");
	input_tmp_counts=to_read;
	__shared__ int db_strlen;
	int db_data_read;
	while(current_db=get_next(db_files,&next_db,&db_strlen))
	{
		
		db_files=next_db;
		db_idx++;

		__shared__ int zfd_db;
		BEGIN_SINGLE_THREAD
			zfd_db=single_thread_open((char*)current_db,O_GRDONLY);
			if (zfd_db<0) ERROR("Failed to open DB file");
		END_SINGLE_THREAD
		size_t db_size=fstat(zfd_db);
			
		_pagehelper ph_db={NULL,0};
		volatile char* ptr_row_db=get_row(&ph_db.page,&ph_db.file_offset,0,db_size,zfd_db,O_GRDONLY);
	
		for (size_t _cursor=0;_cursor< db_size;)
		{
			
			if (_cursor - ph_db.file_offset >= FS_BLOCKSIZE) 
			{
				ptr_row_db=get_row(&ph_db.page,&ph_db.file_offset,_cursor,db_size,zfd_db,O_GRDONLY);
			}

			int db_left=db_size-_cursor<CORPUS_PREFETCH_SIZE?db_size-_cursor: CORPUS_PREFETCH_SIZE;

			prefetch(corpus,ptr_row_db,db_left,CORPUS_PREFETCH_SIZE);

			// take care of the stitches
			int overlap=0;
			
			if(db_left==CORPUS_PREFETCH_SIZE){
				_cursor+=CORPUS_PREFETCH_SIZE;
				ptr_row_db+=CORPUS_PREFETCH_SIZE;
				overlap=db_size-_cursor>32? 32:db_size-_cursor;
				// get the overlap
			
				if (_cursor > ph_db.file_offset + FS_BLOCKSIZE ){
					ptr_row_db=get_row(&ph_db.page,&ph_db.file_offset,_cursor,db_size,zfd_db,O_GRDONLY);
				}
				overlap=prefetch_word(corpus+CORPUS_PREFETCH_SIZE,ptr_row_db);
			}else{
				_cursor+=CORPUS_PREFETCH_SIZE;
			}
			///////////////////// NOW WE ARE DEALING WITH THE INPUT							
			//
			// indexing is in chars, not in row size
			for(int input_block=0;input_block<input_tmp_counts;input_block+=INPUT_PREFETCH_SIZE){
				
				int data_left=input_tmp_counts-input_block;

				prefetch_banks(input,input_tmp + input_block,min(data_left,INPUT_PREFETCH_SIZE),INPUT_PREFETCH_SIZE);
								
				char word_size=0;
				int res= match_string(input+threadIdx.x*33,corpus,CORPUS_PREFETCH_SIZE+overlap,&word_size);
				//int res=0;
					
				if (!__syncthreads_or(res!=LEN_ZERO && res )) continue;
				
				if(res!=LEN_ZERO && res ){
					char numstr[4]; int numlen;
					print_uint(numstr,res,&numlen);
						
					int offset=atomicAdd(&output_count,(numlen+1+word_size+1+db_strlen+1));

						char* outptr=output_buffer+offset;
						memcpy_thread(outptr,
						            input+threadIdx.x*33,word_size);
						outptr[word_size]=' ';
						
						memcpy_thread(outptr+word_size+1,numstr,numlen);
						outptr[word_size+numlen+1]=' ';
						
						memcpy_thread(outptr+word_size+numlen+2,current_db,db_strlen);
						outptr[word_size+numlen+db_strlen+2]='\n';
				}
				__syncthreads();
				if (output_count){
					__shared__ int old_offset;
					if (threadIdx.x==0) old_offset=atomicAdd(&global_output,output_count);
					__syncthreads();
					if(gwrite(zfd_o, old_offset, output_count,(uchar*) output_buffer)!=output_count)
					{
						ERROR("Write to output failed");
					}
				}
				__syncthreads();

				/// how many did we find
				if(threadIdx.x==0){ 
					output_count=0;
				}
				__syncthreads();

					
			}
		}
		if(gmunmap(ptr_row_db,0)) ERROR("Failed to unmap db");
		
		ph_db.page=NULL; ph_db.file_offset=0;
			BEGIN_SINGLE_THREAD
				single_thread_close(zfd_db);
				output_count=0;
			END_SINGLE_THREAD
		// at this point we scanned through all the DBs and found no matches for the strings in the prefetched memory
	}
	//we are done.
	//write the output and finish
		
	BEGIN_SINGLE_THREAD
        	single_thread_close(zfd_src);
        	single_thread_close(zfd_dbs);
		free(output_buffer);
		free(input_tmp);
                if (last_lock.is_last()){
                        single_thread_fsync(zfd_o);
                }
                single_thread_close(zfd_o);
        END_SINGLE_THREAD
}




void init_device_app(){
      CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize,1<<30));
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


