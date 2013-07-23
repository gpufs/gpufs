/* 
 * * This expermental software is provided AS IS. 
 * * Feel free to use/modify/distribute, 
 * * If used, please retain this disclaimer and cite 
 * * "GPUfs: Integrating a file system with GPUs", 
 * * M Silberstein,B Ford,I Keidar,E Witchel
 * * ASPLOS13, March 2013, Houston,USA
 * */


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

#define alpha(src)	(((src)>=65 && (src)<=90)||( (src)>=97 && (src)<=122)|| (src)==95 )

int* file_sizes;
char** file_names;
 

int main(int argc, char** argv){
	
	file_sizes=new int[60000];
	file_names=new char*[60000];
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

#define BUF_SIZE (1024*1024*1024)

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

		

	size_t src_size=cur_size;

	int total_words=words_size/32;
#pragma omp parallel for schedule(static,1)
for(int iii=0;iii<8;iii++)
{
		int threadId=omp_get_thread_num();
		int per_thread=total_words/omp_get_num_threads();
		fprintf(stderr, "threads: %d,  threadId: %d\n",omp_get_num_threads(), omp_get_thread_num());

		for(int w_ptr=per_thread*threadId*32;w_ptr<per_thread*(threadId+1)*32;w_ptr+=32){
			int letter_counter=0;
			int word_counter=0;
			int file_counter=0;
			int total_size=file_sizes[0];
			
			int src_ptr=0;
			int word_start=1;
			for( src_ptr=0;src_ptr<src_size;src_ptr++){


				if (words[w_ptr+letter_counter]==buf[src_ptr] && buf[src_ptr]!='\n' ){
					letter_counter++;
					if (words[w_ptr+letter_counter]=='\0'  || letter_counter==32) {
						
						if (  src_ptr+1==src_size ||
							! (alpha(buf[src_ptr+1]))  ){
							letter_counter=0;
							word_counter++;
							src_ptr++;
							continue;
						}else{ 
							for(;src_ptr<src_size&& alpha(buf[src_ptr]);src_ptr++);
							letter_counter=0;
							// jump to the next file
							src_ptr++;
							continue;
						}
					}
				}else{
					for(;src_ptr<src_size && alpha(buf[src_ptr]);src_ptr++);
					letter_counter=0;
				}
				src_ptr++;
				
				if (src_ptr+1>=total_size || src_ptr+1 >=src_size )
				{	
					if(word_counter) printf("%s %d %s\n",&words[w_ptr], word_counter,file_names[file_counter]);
					word_counter=0;
					
					file_counter++;
					total_size+=file_sizes[file_counter];
				}
			}
		}
}
	return 0;

}
