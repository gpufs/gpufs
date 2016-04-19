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
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

      double _timestamp(){
                struct timeval tv;
                gettimeofday(&tv,0);
                return 1e6*tv.tv_sec+tv.tv_usec;
        }



#define TH (0.01)
float* rpool;
void rpool_populate()
{
        rpool=new float[1<<20];
        for(int i=0;i<1<<20;i++) rpool[i]=rand()/(float)INT_MAX;
}

int v_len=1024;
//int num_rows=3*32*5;
int num_db_files=3;
int max_rows_per_db=10000;
int min_rows_per_db=8000;

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
 	void* data=mmap(NULL,s.st_size,type==O_RDONLY?PROT_READ:PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,open_fd,0);	
	if (data==MAP_FAILED)	{
		perror("mmap");
		close(open_fd);
		return NULL;
	}
	*fd=open_fd;
	*size=s.st_size;
	return data;
}

void unmap_close_file(int fd, void* ptr,int len)
{
	
	if(munmap(ptr,len)) { perror("unmap"); return;}
	close(fd);
}

struct fdata{
	int fd;
	size_t size;
	float* ptr;
};
	
int search(float* src,float* db,int v_len, long db_size)
{
	for(long dbi=0;dbi<db_size;dbi++){
        	float res=0;
	        for(long j=0;j<v_len;j++){
			float val=(src[j]-db[v_len*dbi+j]);
        	        res+=val*val;
	        }
		res=sqrt(res);
		if (res <= TH ) return dbi;
	}
	return -1;
}

void generate_matrix(float* tgt, int row_size, int num_rows)
{
        for(long i=0;i<num_rows;i++,tgt+=row_size)
	{
                for(long j=0;j<row_size;j++){
                        tgt[j]=rand()/((float)INT_MAX);
			if (tgt[j]>=1) tgt[j]=0.9;
			if (tgt[j]==0) tgt[j]=0.1;

//rpool[(j+i)%(1<<20)];
                }
	}
}

void generate(const char* prefix, int v_len, int num_rows, int num_db_files, int max_rows_per_db_file, int min_rows_per_db_file)
{
	char fname[256];
	sprintf(fname,"%s_in",prefix);
	
	fdata in_file;
	in_file.size=v_len*num_rows*sizeof(float);
        in_file.ptr=(float*)open_map_file(fname,&in_file.fd,&in_file.size,O_RDWR);
	if (!in_file.ptr) assert(0);

	generate_matrix(in_file.ptr,v_len,num_rows);

	for( int i=0;i<num_db_files;i++)
	{
		fdata db_file;
	
		int  db_rows=(max_rows_per_db_file-min_rows_per_db_file)*(rand()/(float)INT_MAX)+min_rows_per_db_file;
		db_file.size=db_rows*sizeof(float)*v_len;
		
		sprintf(fname,"%s_o_%d",prefix,i);
		db_file.ptr=(float*)open_map_file(fname,&db_file.fd,&db_file.size,O_RDWR);
		assert(db_file.ptr);
		
		generate_matrix(db_file.ptr,v_len,db_rows);
		// now add our rows
	//	int num_hits=(rand()/((float)INT_MAX) * num_rows) ;
		int num_hits= num_rows ;
		
		for(int hits=0;hits<num_hits;hits++){
			int hit_row_src=rand()/((float)INT_MAX)*num_rows;
			int hit_row_tgt=rand()/((float)INT_MAX)*db_rows;
			memcpy(db_file.ptr+hit_row_tgt*v_len, 
			       in_file.ptr+hit_row_src*v_len,
				v_len*sizeof(float));
		}
		printf("File %s has total %d exact hits\n",fname,num_hits);
		unmap_close_file(db_file.fd, db_file.ptr,db_file.size);
	}
	unmap_close_file(in_file.fd,in_file.ptr,in_file.size);
}

#define USAGE "-g(enerate) <prefix> <num_rows_input> <max_rows_db> <min_rows_db>|-r(ead db in text) <dbfile> | -o(utput to translate  into text) output_file | <source_f>,source_row_len, <outfile>, <f1>,<f2>,<f3>.....\n"
int main(int argc, char** argv)
{
	fprintf(stderr, "Threshold: %.5f\n", TH);
	if (argc<3) { printf(USAGE); return -1; }
	int num_rows=0;
		if (!strcmp(argv[1],"-g")) {
			assert(argc==6);
			rpool_populate();
	
			num_rows=atoi(argv[3]);
			max_rows_per_db=atoi(argv[4]);
			min_rows_per_db=atoi(argv[5]);
			generate(argv[2], v_len, num_rows, num_db_files, max_rows_per_db,min_rows_per_db);
			delete []rpool;
			return 0;
		}
		if (!strcmp(argv[1],"-r"))
		{
			fdata f;
			f.ptr=(float*)open_map_file(argv[2],&f.fd,&f.size, O_RDONLY);
			assert(f.ptr);
			
			int v_len=16;
			for(int i=0;i<f.size/v_len/sizeof(float);i++){
				for( int k=0;k<v_len;k++){
					printf("%.3f ",f.ptr[v_len*i+k]);
				}
				printf("\n");
			}
			return 0;
		}	
		if (!strcmp(argv[1],"-o"))
		{	printf("src_line\tfile\tline\n");
			fdata f;
			f.ptr=(float*)open_map_file(argv[2],&f.fd,&f.size, O_RDONLY);
			assert(f.ptr);
			int* p=(int*)f.ptr;
			for(int i=0;i<f.size/(sizeof(int));i+=3)
			{
				printf("%d\t%d\t%d\n",p[i],p[i+1],p[i+2]);
			}
			return 0;
		}
			
		 printf(USAGE); return -1;
		


	int v_len=atoi(argv[2]);
	if (v_len<=0) { fprintf(stderr,"len <0\n"); return -1; }
	char* out_file_name=argv[3];
	
	int out_fd=open(out_file_name, O_WRONLY|O_CREAT|O_TRUNC,S_IRUSR|S_IWUSR);

	if (out_fd<0 ) { perror("failed open output"); return -1;}

	fdata src;
		
	src.ptr=(float*)open_map_file(argv[1],&src.fd,&src.size,O_RDONLY);
	if (!src.ptr) assert(NULL);
	
/*
#pragma omp parallel for schedule(static,1)
for(int iii=0;iii<8;iii++)
{
int threadId=iii;
int per_thread=total_rows/omp_get_num_threads();
int last_thread=(iii==7)?total_rows%omp_get_num_threads():0;


fprintf(stderr, "thread: %d, id %d, per_thread %d, last_thread %d\n",omp_get_num_threads(),threadId, per_thread,last_thread );
	
*/	
//	for( int seq=per_thread*threadId;seq<per_thread*(threadId+1)+last_thread;seq++)
int total_rows=src.size/v_len/sizeof(float);
#pragma omp parallel for	
	for (int seq=0;seq<total_rows;seq++)
	{
		int found=0;
		for(int i=4;i<argc;i++)
		{
			fdata tgt;
			tgt.ptr=(float*)open_map_file(argv[i],&tgt.fd,&tgt.size,O_RDONLY);
			if(!tgt.ptr) assert(NULL);

			int ln=search(&src.ptr[seq*v_len],tgt.ptr,v_len,tgt.size/v_len/sizeof(float));
			if (ln>=0)
			{
				int d[]={seq,i-4,ln};
				#pragma omp critical
				if(write(out_fd,d,sizeof(d))!=sizeof(d)) {
					perror("output failed\n");  
					assert(NULL);
				}
				found=1;
			}
			unmap_close_file(tgt.fd,tgt.ptr,tgt.size);
			if (found) break;
		}
		int d[]={seq,-1,-1};
		#pragma omp critical
		if (found==0) write(out_fd,d,sizeof(d));
	}

	close(out_fd);
	return 0;
}
