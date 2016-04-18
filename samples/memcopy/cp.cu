#include <stdio.h>
#include <errno.h>

#include "fs_calls.cu.h"
#include "host_loop.h"

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#endif

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int) err, cudaGetErrorString(err));
		exit(-1);
	}
}

#define MAX_TRIALS (10)
double time_res[MAX_TRIALS];

char* update_filename(const char* h_filename)
{
	int n = strlen(h_filename);
	assert(n > 0);
	if (n > FILENAME_SIZE)
	{
		fprintf(stderr, "Filname %s too long, should be only %d symbols including \\0", h_filename, FILENAME_SIZE);
		exit(-1);
	}
	char* d_filename;
	CUDA_SAFE_CALL(cudaMalloc(&d_filename, n + 1));
	CUDA_SAFE_CALL(cudaMemcpy(d_filename, h_filename, n + 1, cudaMemcpyHostToDevice));
	return d_filename;
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

__device__ int OK;
__shared__ int zfd, zfd1, zfd2, close_ret;

__device__ LAST_SEMAPHORE sync_sem;
__global__ void test_cpy(char* src, char* dst)
{
	__shared__ uchar* scratch;
	__shared__ size_t filesize;
	BEGIN_SINGLE_THREAD scratch = (uchar*) malloc(FS_BLOCKSIZE);
	GPU_ASSERT(scratch != NULL);
	END_SINGLE_THREAD

	zfd = 0;
	zfd = gopen(src, O_GRDONLY);

	zfd1 = 0;
	zfd1 = gopen(dst, O_GWRONCE);
	filesize = fstat(zfd);

	for (size_t me = blockIdx.x * FS_BLOCKSIZE; me < filesize;
			me += FS_BLOCKSIZE * gridDim.x)
	{
		int toRead = min((unsigned int) FS_BLOCKSIZE,
				(unsigned int) (filesize - me));
		if (toRead != gread(zfd, me, toRead, scratch))
		{
			assert(NULL);
		}

		if (toRead != gwrite(zfd1, me, toRead, scratch))
		{
			assert(NULL);
		}

	}

	gclose(zfd);
	gclose(zfd1);

}
void init_device_app()
{

	CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30));
}

void init_app()
{
	void* d_OK;
	CUDA_SAFE_CALL(cudaGetSymbolAddress(&d_OK, OK));
	CUDA_SAFE_CALL(cudaMemset(d_OK, 0, sizeof(int)));
	// INITI LOCK   
	void* inited;

	CUDA_SAFE_CALL(cudaGetSymbolAddress(&inited, sync_sem));
	CUDA_SAFE_CALL(cudaMemset(inited, 0, sizeof(LAST_SEMAPHORE)));
}

double post_app(double time, int trials)
{
	int res;
	CUDA_SAFE_CALL(
			cudaMemcpyFromSymbol(&res, OK, sizeof(int), 0,
					cudaMemcpyDeviceToHost));
	if (res != 0)
		fprintf(stderr, "Test Failed, error code: %d \n", res);
	else
		fprintf(stderr, "Test Success\n");

	return 0;
}

int main(int argc, char** argv)
{
	int device = 0;
	char* gpudev = getenv("GPUDEVICE");
	if (gpudev != NULL)
		device = atoi(gpudev);

	CUDA_SAFE_CALL(cudaSetDevice(device));

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

	printf("Running on device %d: \"%s\"\n", device, deviceProp.name);

	if (argc < 5)
	{
		fprintf(stderr, "<kernel_iterations> <blocks> <threads> f1 f2 ... f_#files\n");
		return -1;
	}

	int trials = atoi(argv[1]);
	assert(trials <= MAX_TRIALS);
	int nblocks = atoi(argv[2]);
	int nthreads = atoi(argv[3]);

	fprintf(stderr, "\titerations: %d blocks %d threads %d\n", trials, nblocks, nthreads);

	int num_files = argc - 1 - 3;
	char** d_filenames = NULL;

	double total_time = 0;
	size_t total_size = 0;

	memset(time_res, 0, MAX_TRIALS * sizeof(double));
	for (int i = 1; i < trials + 1; i++)
	{
		volatile GPUGlobals* gpuGlobals;
		initializer(&gpuGlobals);

		init_device_app();
		init_app();

		if (num_files > 0)
		{
			d_filenames = (char**) malloc(sizeof(char*) * num_files);
			for (int i = 0; i < num_files; i++)
			{
				d_filenames[i] = update_filename(argv[i + 4]);
				fprintf(stderr, "file -%s\n", argv[i + 4]);
			}
		}

		double time_before = _timestamp();
		if (!i)
			time_before = 0;

		double c_open, c_rw, c_close;
		c_open = c_rw = c_close = 0;

		test_cpy<<<nblocks,nthreads,0,gpuGlobals->streamMgr->kernelStream>>>(d_filenames[0], d_filenames[1]);

		run_gpufs_handler(gpuGlobals, device);
		cudaError_t error = cudaDeviceSynchronize();
		double time_after = _timestamp();
		if (!i)
			time_after = 0;
		total_time += (time_after - time_before);
		if (i > 0)
		{
			time_res[i] = time_after - time_before;
			fprintf(stderr, " t-%.3f-us\n", time_res[i]);
		}
		fprintf(stderr, "open: %.0f, rw %.0f, close %.0f usec\n", c_open, c_rw,
				c_close);

		//Check for errors and failed asserts in asynchronous kernel launch.
		if (error != cudaSuccess)
		{
			printf("Device failed, CUDA error message is: %s\n\n",
					cudaGetErrorString(error));
		}

		//PRINT_DEBUG;

		fprintf(stderr, "\n");
		delete gpuGlobals;

		cudaDeviceReset();
		if (error)
			break;

	}

	if (d_filenames)
		free(d_filenames);

	double thpt = post_app(total_time, trials);
	struct stat s1, s2, s3;
	if (stat(argv[4], &s1))
		perror("stat failed");
	if (stat(argv[5], &s2))
		perror("stat failed");
	total_size = s1.st_size;
	double d_size = total_size / 1024.0 / 1024.0 / 1024.0;

	double avg_time, avg_thpt, std_time, std_thpt;

	stdavg(&avg_time, &avg_thpt, &std_time, &std_thpt, time_res, d_size,
			MAX_TRIALS);

	fprintf(stderr,
			"Performance: %.3f usec +/- %.3f, %.3f GB,  %.3f GB/s +/- %.3f, FS_BLOCKSIZE %d FS_LOGBLOCKSIZE %d\n",
			avg_time, std_time, d_size, avg_thpt * 1e6, std_thpt * 1e6,
			FS_BLOCKSIZE, FS_LOGBLOCKSIZE);
//((double)output_size*(double)nblocks*(double)read_count)/(total_time/TRIALS)/1e3 );
	return 0;
}

