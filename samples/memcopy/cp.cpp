
/* 
 * * This expermental software is provided AS IS. 
 * * Feel free to use/modify/distribute, 
 * * If used, please retain this disclaimer and cite 
 * * "GPUfs: Integrating a file system with GPUs", 
 * * M Silberstein,B Ford,I Keidar,E Witchel
 * * ASPLOS13, March 2013, Houston,USA
 * */

#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "timer.h"
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"
#define TRIALS 5
#define CUDA_SAFE_CALL(x) if((x)!=cudaSuccess) { fprintf(stderr,"CUDA ERROR %d %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); exit(-1); }
	

void*  open_map_file(const char* f, int* fd, size_t* size, int type)
{
        int open_fd=open(f,type==O_RDONLY?type:type|O_CREAT|O_TRUNC,S_IRUSR|S_IWUSR);



        if (open_fd<0){
                 perror("open failed");
                return NULL;
        }
        if (type!=O_RDONLY) {
                assert(*size>0);
                if (ftruncate(open_fd,*size)){
                        perror("ftrunc failed");
                        return NULL;
                }
        }
        struct stat s;
        if (fstat(open_fd,&s)) {
                        fprintf(stderr,"Problem with fstat the file on CPU: %s \n ",strerror(errno));
        }

        if (s.st_size==0) {
                fprintf(stderr,"file with zero lenght, skipping %s\n",f);
                close(open_fd);
                return NULL;
        }
        void* data=mmap(NULL,s.st_size,type==O_RDONLY?PROT_READ:PROT_READ|PROT_WRITE,MAP_SHARED,open_fd,0);
        if (data==MAP_FAILED)   {
                perror("mmap");
                close(open_fd);
                return NULL;
        }
        *fd=open_fd;
        *size=s.st_size;
        return data;
}

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

#define MAX_TRIALS (10)
double time_res[MAX_TRIALS];


int main(int argc, char** argv)
{
	if(argc<5) { printf("<transfer size>, <block size>,<filename> <option 1-4> "); return 0;}
	
	size_t transfer_size=atoll(argv[1]);
	int block_size=atoi(argv[2]);
	char* filename=argv[3];
	int option=atoi(argv[4]);
	assert(block_size>0);

	
	char* h_mem=0;
	char* d_mem=0;
	CUDA_SAFE_CALL(cudaHostAlloc(&h_mem,transfer_size,  cudaHostAllocDefault));
	CUDA_SAFE_CALL(cudaMalloc(&d_mem,transfer_size));

	int fd;
	size_t size;
	char* fmem=(char*)open_map_file(filename, &fd, &size, O_RDONLY);
	transfer_size=size;
	assert(transfer_size>0);
	fprintf(stderr,"Running with transfer size=%ld, block size %d, filename %s\n", transfer_size,block_size,filename);

	cudaStream_t s;
	CUDA_SAFE_CALL(cudaStreamCreate(&s));

       double avg_time,avg_thpt,std_time,std_thpt;
	char* prefix;
	memset(time_res,0,MAX_TRIALS*sizeof(double));

	assert(fmem);
	double total_time=0;
	switch(option){
	case 0:

	prefix="LOOP_FULL";
	for(int t=0;t<TRIALS+1;t++){
	
		double beg=_timestamp();
		for(unsigned int i=0;i<transfer_size;i+=block_size){
		        size_t r=pread(fd,h_mem+i,block_size,i);
			assert(r==block_size);
		//	memcpy(h_mem+i,fmem+i,block_size);
			CUDA_SAFE_CALL(cudaMemcpyAsync(d_mem+i,h_mem+i,block_size,cudaMemcpyHostToDevice,s));
		}
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		double end=_timestamp();
		if (t!=0) time_res[t]=(end-beg);
	}
	cudaFreeHost(h_mem);

	break;
	case 1:
	total_time=0;
	for(int t=0;t<TRIALS+1;t++){
		double beg=_timestamp();
		for(int i=0;i<transfer_size;i+=block_size){
			CUDA_SAFE_CALL(cudaMemcpyAsync(d_mem+i,h_mem,block_size,cudaMemcpyHostToDevice,s));
		}
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		double end=_timestamp();
		if (t!=0) time_res[t]=(end-beg);
	}
	cudaFreeHost(h_mem);
	prefix="LOOP_NO_MEMCPY";

	
	break;
	case 2:
	total_time=0;
	for(int t=0;t<TRIALS+1;t++){
		
		double beg=_timestamp();
		size_t r=pread(fd,h_mem,transfer_size,0);

		assert(r==transfer_size);
	//	memcpy(h_mem,fmem,transfer_size);
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_mem,h_mem,transfer_size,cudaMemcpyHostToDevice,s));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		double end=_timestamp();
		if (t!=0) time_res[t]=(end-beg);
	}
	prefix="ONE_GO_FULL";
	cudaFreeHost(h_mem);
	break;
	case 3:
	total_time=0;
	for(int t=0;t<TRIALS+1;t++){
		
		double beg=_timestamp();
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_mem,h_mem,transfer_size,cudaMemcpyHostToDevice,s));
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		double end=_timestamp();
		if (t!=0) time_res[t]=(end-beg);
	}
	prefix="ONE_GO_NO_MEMCPY";
	}
        stdavg(&avg_time,&avg_thpt, &std_time, &std_thpt, time_res, transfer_size/(1<<20), MAX_TRIALS);
        fprintf(stderr,"Total time: %s: %.3f usec +/- %.3f, %ld MB,  %.3f MB/s +/- %.3f\n",prefix,avg_time,std_time, transfer_size/(1<<20), avg_thpt*1e6,std_thpt*1e6);


	return 0;
}


