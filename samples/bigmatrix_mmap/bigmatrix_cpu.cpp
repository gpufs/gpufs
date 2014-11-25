/* 
 * * This expermental software is provided AS IS. 
 * * Feel free to use/modify/distribute, 
 * * If used, please retain this disclaimer and cite 
 * * "GPUfs: Integrating a file system with GPUs", 
 * * M Silberstein,B Ford,I Keidar,E Witchel
 * * ASPLOS13, March 2013, Houston,USA
 * */



#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

      double _timestamp(){
                struct timeval tv;
                gettimeofday(&tv,0);
                return 1e6*tv.tv_sec+tv.tv_usec;
        }




void die(const char* msg){
	perror(msg);
	exit(-1);

}

float* rpool;

void rpool_populate()
{
	rpool=new float[1<<20];
	for(int i=0;i<1<<20;i++) rpool[i]=rand()/(float)INT_MAX;
}

void generate(long row_size, long num_rows, const char* filename )
{	
	assert(row_size>0);
	
	float* line=new float[row_size];

	FILE* f=fopen(filename,"w+");
	if (!f) die("file open failed");
	
	int fd=fileno(f);

	for(long i=0;i<num_rows;i++){
		for(long j=0;j<row_size;j++){
			line[j]=rpool[(j+i)%(1<<20)];
//			line[j-1]=(float)((i+j)%2);
		}
		long towrite=sizeof(float)*row_size;
		if(towrite!= write(fd,line,towrite)){
			die("write failed");
		}
	}
	fclose(f);
	delete []line;
}


int main(int argv, char** argc)
{
	if (argv<4) {
		fprintf(stderr,"<basefile><row_size><num_rows> [force regeneration]\n");
		return -1;
	}
	int force=0;
	if (argv>4) {
		fprintf(stderr,"Regenerating matrices\n"); force=1;
	}

	char v[256];
	char m[256];
	char o[256];
	
	char* filename=argc[1];

	long row_size=atol(argc[2]);
	long num_rows=atol(argc[3]);
	
	if (row_size*num_rows<=0) 	die("size is wrong");
	
	double total_time=0;	

int trials=1;
for(int t=0;t<trials+1;t++){
	
	double before=_timestamp();
	if(!t) { before=0;total_time=0;};

	sprintf((char*)v,"%s_vector",filename);
	sprintf((char*)m,"%s_matrix",filename);
	sprintf((char*)o,"%s_out",filename);
	
	
	int v_fd=open((char*)v,O_RDONLY);
	close(v_fd);
	if (v_fd<0||force) {
		rpool_populate();
		fprintf(stderr,"generating %s -  %ldx%ld\n",(char*)v,row_size,1L);
		generate(row_size,1,(char*)v);
		fprintf(stderr,"generating %s -  %ldx%ld\n",(char*)m,row_size,num_rows);
		generate(row_size,num_rows,(char*)m);
		return 0;
	}else{
		fprintf(stderr,"NOT regenerating \n");
	}
	
	int fd_out=open((char*)o,O_WRONLY|O_TRUNC|O_CREAT,S_IRUSR|S_IWUSR);
	if (fd_out<0) die("cant open output");
	int fd_in_v=open((char*)v,O_RDONLY);
	if (fd_in_v<0) die("cant open input vector");
	int fd_in_m=open((char*)m,O_RDONLY);
	if (fd_in_m<0) die("cant open input matrix");
	
	//posix_fadvise(fd_in_m, 0, 0, POSIX_FADV_SEQUENTIAL );
	
	
	float* vec=new float[row_size];
	float* matrix=new float[row_size*8];
	assert(vec&&matrix);
	
	long to_read=row_size*sizeof(float);

	long size_read=read(fd_in_v,vec,to_read);
	if (to_read!=size_read) die("read not enough data for the vector");

	float* out_vec=new float[num_rows];

	if (!vec) die("cant allocate memory");
#pragma omp parallel for
	for(long i=0;i<num_rows;i++){

	        int threadId=omp_get_thread_num();
//                fprintf(stderr, "threads: %d,  threadId: %d\n",omp_get_num_threads(), omp_get_thread_num());

		if(pread(fd_in_m, matrix+threadId*row_size,to_read,i*row_size*4)!=to_read){
			perror("Problem with pread\n");
			die("read not enough data for the matrix");
		}
		float res=0;
		for(long j=0;j<row_size;j++){
			res+=(matrix[j+threadId*row_size]*vec[j]);
		}
		out_vec[i]=res;
	}

	if(num_rows*sizeof(float)!=write(fd_out,out_vec,num_rows*sizeof(float))){
		die("failed to write result back");
	}
	
	close(fd_out);close(fd_in_v);close(fd_in_m);

	double after=_timestamp()-before;
	if (t) total_time+=after;
	
	if (t) fprintf(stderr,"total time %.0f us  %.3f GB \n ",total_time/t, t*4*(row_size*num_rows+row_size)/total_time/1000);

	delete []out_vec;
	delete [] vec;
	delete []matrix;
//	delete[] rpool;
}
	return 0;

}
