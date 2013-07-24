/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/

/* 
* This expermental software is provided AS IS. 
* Feel free to use/modify/distribute, 
* If used, please retain this disclaimer and cite 
* "GPUfs: Integrating a file system with GPUs", 
* M Silberstein,B Ford,I Keidar,E Witchel
* ASPLOS13, March 2013, Houston,USA
*/


#ifndef HOST_LOOP_CPP
#define HOST_LOOP_CPP

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include<stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include "gpufs_con_lib.h"

void fd2name(const int fd, char* name, int namelen){

        char slink[100];
        pid_t me=getpid();
	name[0]=0;
        sprintf(slink,"/proc/%d/fd/0",me);

        int s=readlink(slink,name, namelen-1);
	if (s>=0)  name[s]='\0';
}

double transfer_time=0;


bool debug_expecting_close=0;

double total_stat=0;
double total_stat1=0;

void open_loop(volatile GPUGlobals* globals,int gpuid)
{
	char* use_gpufs_lib=getenv("USE_GPUFS_DEVICE");
	
	for (int i=0;i<FSTABLE_SIZE;i++)
	{
		char filename[FILENAME_SIZE];
		volatile CPU_IPC_OPEN_Entry* e=&globals->cpu_ipcOpenQueue->entries[i];
	// we are doing open
		if (e->status == CPU_IPC_PENDING && e->cpu_fd < 0 ) 
		{
double vvvv=_timestamp();
			memcpy(filename,(char*)e->filename,FILENAME_SIZE);
			// OPEN
			if (e->flags&O_GWRONCE) {
					e->flags=O_RDWR|O_CREAT;
			}
			char pageflush=0;
			int cpu_fd=-1;
			struct stat s;
			if (e->do_not_open){
			
				if ( stat(filename,&s) <0 ) { fprintf(stderr," problem with STAT file %s on CPU: %s\n",filename, strerror(errno));}
			//	fprintf(stderr,"Do not open for inode %d, time %d\n",s.st_ino, s.st_mtime);
			}else{
				if (use_gpufs_lib) cpu_fd=gpufs_file_open(globals->gpufs_fd,gpuid,filename,e->flags,S_IRUSR|S_IWUSR,&pageflush);
				else {  cpu_fd=open(filename, e->flags,S_IRUSR|S_IWUSR);}

				if (cpu_fd < 0) { fprintf(stderr, "Problem with opening file %s on CPU: %s \n ",filename, strerror(errno)); }
		
				if (fstat(cpu_fd,&s)) { fprintf(stderr,"Problem with fstat the file %s on CPU: %s \n ",filename,strerror(errno));}
			}
	
			//fprintf(stderr, "FD %d,  inode %ld, size %ld, Found file %s\n",i, s.st_ino, s.st_size, filename);
			
			e->cpu_fd=cpu_fd;
			e->flush_cache=pageflush;
			e->cpu_inode=s.st_ino;
			e->size=s.st_size;
			e->cpu_timestamp=s.st_ctime;
			__sync_synchronize();
			e->status=CPU_IPC_READY;
			__sync_synchronize();
total_stat+=(_timestamp()-vvvv);		
		}
		if (e->status == CPU_IPC_PENDING && e->cpu_fd>=0 )
		{
	double vvvv1=_timestamp();
			// do close
			// fprintf(stderr, "FD %d, closing file %s\n",i, e->filename);

			if (use_gpufs_lib) {
				if (e->is_dirty){ // if dirty, update gpufs device, but keep the file open
					e->cpu_fd=gpufs_file_close_stay_open(globals->gpufs_fd,gpuid,e->cpu_fd); 
				}else{
					e->cpu_fd=gpufs_file_close(globals->gpufs_fd,gpuid,e->cpu_fd); 
				}
				gpufs_drop_residence(globals->gpufs_fd, gpuid, e->drop_residence_inode);
			}else{
				if (!e->is_dirty){
					e->cpu_fd=close(e->cpu_fd);
				}
			}
			__sync_synchronize();
			e->status=CPU_IPC_READY;
			__sync_synchronize();
			total_stat1+=_timestamp()-vvvv1;
		}
		
	}
} 

Page diff_page;

uchar* diff_and_merge(const Page* page, uint req_cpu_fd, size_t req_size, size_t req_file_offset){

	struct stat s;
				
      	if (fstat(req_cpu_fd,&s)) s.st_size=0;
					
	int data_read;
	if ( s.st_size<req_file_offset) {
		data_read=0;
	}
	else {
		data_read=pread(req_cpu_fd,&diff_page,req_size,req_file_offset);
	}

	//fprintf(stderr,"read %d, offset %d\n",data_read,req_file_offset);
	if (data_read<0) {
		perror("pread failed while diff\n");
		req_size=(size_t)-1;
	}
	//if (data_read==0) fprintf(stderr,"empty read\n");
					

//	uchar* tmp=(uchar*)page;
//	uchar* data_ptr=(uchar*)&diff_page;			
//	if (data_read==0){
	
//		data_ptr=tmp; // copy directly from the buffer
	
//	}else 
	typedef char v32c __attribute__ ((vector_size (16)));
	uchar* data_ptr=(uchar*)page;
        v32c* A_v=(v32c*)data_ptr;
        v32c* B_v=(v32c*)&diff_page;;
	if (data_read>0){
	

		// perform diff-ed write
//		for(int zzz=0;zzz<data_read;zzz++)
//		{
//			if (tmp[zzz]) { 
//				((uchar*)diff_page)[zzz]=tmp[zzz];
///			}
//		}	
		
		int left=data_read%sizeof(v32c);
	        for(int zzz=0;zzz<(data_read/sizeof(v32c)+(left!=0));zzz++)
                {
                  // new is new OR old
                   //data_ptr[zzz]=data_ptr[zzz]|((uchar*)diff_page)[zzz];
                   A_v[zzz]=A_v[zzz]|B_v[zzz];
                }
	
		//memcpy(((char*)&diff_page)+data_read,tmp+data_read,req_size-data_read);
	}
	return data_ptr;
}						

void async_close_loop(volatile GPUGlobals* globals)
{
	async_close_rb_t* rb=globals->async_close_rb;
	
	char* no_files=getenv("GPU_NOFILE"); 
/*
data must be read synchronously from GPU, but then __can be__ written asynchronously by CPU. -- TODO!
The goal is to read as much as possible from GPU in order to make CPU close as fast as possible
*/

	Page* page=globals->streamMgr->async_close_scratch;
	page_md_t md;

	while(rb->dequeue(page,&md, globals->streamMgr->async_close_stream)){ 
		// drain the ringbuffer
		int res;

		if (md.last_page==1){ 
		// that's the last 
			fprintf(stderr,"closing  dirty file %d\n",md.cpu_fd);
			res=close(md.cpu_fd); 
			if (res<0) perror("Async close failed, and nobody to report to:\n");
		}else{
			if (!no_files){
				//fprintf(stderr,"writing async close at offset: %d content; %d\n",md.file_offset,md.content_size);
				uchar* to_write;
				if (md.type == RW_IPC_DIFF ){
					to_write= diff_and_merge(page, md.cpu_fd, md.content_size, md.file_offset);
				}else{
					to_write=(uchar*)page;
				}

				int ws=pwrite(md.cpu_fd, to_write, md.content_size,md.file_offset);
				if (ws!=md.content_size){ 
					perror("Writing while async close failed, and nobody to report to:\n");
				}
			}
		}
	}
	

}



int max_req=0;
int report=0;
void rw_loop(volatile GPUGlobals* globals)
{
	char* no_pci=getenv("GPU_NOPCI");
	if (no_pci&&!report) fprintf(stderr,"Warning: no data will be transferred in and out of the GPU\n");

	char* no_files=getenv("GPU_NOFILE"); 
	if (no_files&&!report) fprintf(stderr,"Warning: no file reads/writes will be performed\n");
	report=1;
	int cur_req=0;
	
	for (int i=0;i<RW_IPC_SIZE;i++)
	{
		volatile CPU_IPC_RW_Entry* e=&globals->cpu_ipcRWQueue->entries[i];
		if(e->status == CPU_IPC_PENDING)
		{

			cur_req++;
	/*		fprintf(stderr, "FD %d, cpu_fd %d, buf_offset %d, size "
					"%d, type %s, ret_val %d\n",i, 
				e->cpu_fd,
				e->buffer_offset, 
				e->size, 
				e->type==RW_IPC_READ?"read":"write",
				e->return_value 
			);
	*/	    	int req_cpu_fd		=e->cpu_fd;
                        size_t req_buffer_offset	=e->buffer_offset;
			size_t req_file_offset	=e->file_offset;
                        size_t req_size		=e->size;
                        int req_type		=e->type;
			assert(req_type == RW_IPC_READ || req_type == RW_IPC_WRITE || req_type == RW_IPC_DIFF || req_type == RW_IPC_TRUNC );
			if (req_type!=RW_IPC_TRUNC){
				assert(req_cpu_fd>=0 && req_size>0 );
			}
			
			if(globals->streamMgr->task_array[i]!=-1) 
			{
				// we only need to check the stream
		                cudaError_t cuda_status= cudaStreamQuery(globals->streamMgr->memStream[i]);

				if ( cudaErrorNotReady == cuda_status ) 
				{
					// rush to the next request, this one is not ready
					continue;
				}
				if (  cuda_status != cudaSuccess)
				{
					fprintf(stderr, "Error in the host loop.\n ");
					 cudaError_t error = cudaDeviceSynchronize();
        				fprintf(stderr,"Device failed, CUDA error message is: %s\n\n", cudaGetErrorString(error));
					exit(-1);
				}

				// we are here only if success
			}


			switch(req_type)
			{	
				case RW_IPC_READ:
				{
					
				// read
					int cpu_read_size=0;
					if (globals->streamMgr->task_array[i]==-1)
					// the request only started to be served
					{
if(!no_files){
						transfer_time-=_timestamp();
						cpu_read_size=pread(req_cpu_fd,
							globals->streamMgr->scratch[i],
							req_size,req_file_offset);
						transfer_time+=_timestamp();
 }else{
	cpu_read_size=req_size;
}
						char fname[256];
						if (cpu_read_size < 0) { 
							fd2name(req_cpu_fd,fname,256);
							fprintf(stderr, "Problem with reading file %s on CPU: %s \n ", fname, strerror(errno)); 
						}
					//	if (cpu_read_size != req_size ) { fprintf(stderr, "Read %d required %d on CPU\n ", cpu_read_size, req_size); }
					//	if (cpu_read_size ==0 ) { fprintf(stderr,"Nothing has been read\n");}
					
						e->return_value=cpu_read_size;

						if (cpu_read_size > 0)
						{
							globals->streamMgr->task_array[i]=req_type;
if (!no_pci){
								CUDA_SAFE_CALL(cudaMemcpyAsync(((char*)globals->rawStorage)+req_buffer_offset,
								globals->streamMgr->scratch[i],
								cpu_read_size,cudaMemcpyHostToDevice,globals->streamMgr->memStream[i]));
}
						}
					}
					// if read failed or we did not update cpu_read_size since we didn't take the previous if
					if (cpu_read_size <=0)
					{
						// complete the request
						globals->streamMgr->task_array[i]=-1;
						__sync_synchronize();
						e->status=CPU_IPC_READY;
						__sync_synchronize();
					}
					
				}
				break;
			
				case RW_IPC_TRUNC:
					e->return_value=ftruncate(req_cpu_fd,0);
					__sync_synchronize();
					e->status=CPU_IPC_READY;
					__sync_synchronize();
				break;
				case RW_IPC_DIFF:
				{	
					if (globals->streamMgr->task_array[i]==-1)
					{
						globals->streamMgr->task_array[i]=req_type; // enqueue
if (!no_pci){
	//						fprintf(stderr,"RW_IPC_DIFF buf_offset %llu, size %llu\n", req_buffer_offset, req_size);
							CUDA_SAFE_CALL(cudaMemcpyAsync(
								globals->streamMgr->scratch[i],
								((char*)globals->rawStorage)+req_buffer_offset,
								req_size,cudaMemcpyDeviceToHost,globals->streamMgr->memStream[i]));
}
					
					}else{
						globals->streamMgr->task_array[i]=-1;
						// request completion
if (!no_files){
						uchar* to_write=diff_and_merge((Page*)globals->streamMgr->scratch[i],req_cpu_fd,req_size,req_file_offset);
						
						int res=pwrite(req_cpu_fd,to_write,req_size,req_file_offset);
						if (res!=req_size) {
							perror("pwrite failed on diff\n");
							req_size=(size_t)-1;
						}
}// end of no_files
						e->return_value=req_size;
						__sync_synchronize();
						e->status=CPU_IPC_READY;
						__sync_synchronize();
					}
				}
				break;
				case RW_IPC_WRITE:
				{
					if (globals->streamMgr->task_array[i]==-1)
					{
if (!no_pci){
				   	
						CUDA_SAFE_CALL(cudaMemcpyAsync(globals->streamMgr->scratch[i],
								((char*)globals->rawStorage)+req_buffer_offset,
								req_size,cudaMemcpyDeviceToHost,globals->streamMgr->memStream[i]));
}
						globals->streamMgr->task_array[i]=req_type; // enqueue
					}else{
						globals->streamMgr->task_array[i]=-1; // compelte
						int cpu_write_size=req_size;
if(!no_files){			
						cpu_write_size=pwrite(req_cpu_fd,
									globals->streamMgr->scratch[i],
									req_size,req_file_offset);
}

						if (cpu_write_size < 0) { 
							
							char fname[256];
							fd2name(req_cpu_fd,fname,256);
							fprintf(stderr, "Problem with writing  file %s on CPU: %s \n ",fname, strerror(errno)); 
						}
						if (cpu_write_size != req_size ) { 
							char fname[256];
							fd2name(req_cpu_fd,fname,256);
							fprintf(stderr, "Wrote less than expected on CPU for file %s\n ",fname); }
						e->return_value=cpu_write_size;
						__sync_synchronize();
						e->status=CPU_IPC_READY;
						__sync_synchronize();
					}
				}
				break;
				default:
					assert(NULL);
			}	
		}
	}
	if (max_req<cur_req) max_req=cur_req;

}

void run_gpufs_handler(volatile GPUGlobals* gpuGlobals, int deviceNum){
       int device_num=0;
        int done=0;
        while(!done)
        {
                open_loop(gpuGlobals,device_num);
                rw_loop(gpuGlobals);
                if ( cudaErrorNotReady != cudaStreamQuery(gpuGlobals->streamMgr->kernelStream)) {
                        fprintf(stderr,"kernel is complete\n");
                        fprintf(stderr,"Max pending requests: %d\n",max_req);
                        fprintf(stderr,"Transfer time: %.3f\n",transfer_time);
                        transfer_time=0;
                        done=1;
                }
                async_close_loop(gpuGlobals);
        }

}

#endif
