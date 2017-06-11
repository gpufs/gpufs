/*
 * test.cpp
 *
 *  Created on: Aug 6, 2014
 *      Author: sagi
 */

#include <fstream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <limits.h>
#include <sched.h>
#include "tbb/tbb.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
//#include "histLib.h"

#include "loader.h"

static const char* DEFAULT_HISTOGRAMS_FILE_NAME	 = "tiny_2M.hist";
static const char* DEFAULT_DATASET_FILE_NAME	 = "tiny_2M.bin";
static const char* DEFAULT_PARAM_FILE			 = "tiny_2M.param";
static const char* DEFAULT_TABLE_FILE			 = "tiny_2M";

typedef unsigned int uint;
typedef unsigned char uchar;

using namespace std;
using namespace cv;

void mosaic_GPU( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS );
void mosaic_GPUfs( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS );
void mosaic_CPU( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS );

#ifdef WARP

void mosaic_GPUfs_warp( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS );

#ifdef GPUFS_VM
void mosaic_GPUfs_VM( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS );
#else
void mosaic_GPUfs_VM( uchar* inOut, int* bests, AlgData &data, const int ROWS, const int COLS ){}
#endif

#endif

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

char* getCmdOption(char ** begin, char ** end, const string& option)
{
    char ** itr = find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return NULL;
}

bool cmdOptionExists(char** begin, char** end, const string& option)
{
    return find(begin, end, option) != end;
}

int main(int argc, char** argv)
{
	int dev = 0;

	if (cmdOptionExists(argv, argv + argc, "-dev")) {
		dev = stol(getCmdOption(argv, argv + argc, "-dev"));
	}

	checkCudaErrors( cudaSetDevice(dev) );

	cudaDeviceProp deviceProp;
	checkCudaErrors( cudaGetDeviceProperties(&deviceProp, dev) );

	printf("Running on device %d: \"%s\"\n", dev, deviceProp.name);

	const char* datasetFileName = DEFAULT_DATASET_FILE_NAME;

	if( cmdOptionExists(argv, argv + argc, "-dataset") )
	{
		datasetFileName = getCmdOption(argv, argv + argc, "-dataset");
	}

	const char* histogramsFileName = DEFAULT_HISTOGRAMS_FILE_NAME;

	if( cmdOptionExists(argv, argv + argc, "-histograms") )
	{
		histogramsFileName = getCmdOption(argv, argv + argc, "-histograms");
	}

	const char* input = "1024x768.jpg";
	const char* output = "mosaic.jpg";

	if( cmdOptionExists(argv, argv + argc, "-i") )
	{
		input = getCmdOption(argv, argv + argc, "-i");
	}

	if( cmdOptionExists(argv, argv + argc, "-o") )
	{
		output = getCmdOption(argv, argv + argc, "-o");
	}

	Mat old = imread(input);
	int width = old.cols;
	int hight = old.rows;

	AlgData data;

	if( cmdOptionExists(argv, argv + argc, "-load") )
	{
		char* paramFile = getCmdOption(argv, argv + argc, "-load");
		ifstream stream(paramFile);

		data.loadParams( stream );
	}
	else
	{
		const char* paramFile = DEFAULT_PARAM_FILE;
		ifstream stream(paramFile);

		data.loadParams( stream );
	}

	if( cmdOptionExists(argv, argv + argc, "-load-tables") )
	{
		char* tablesFile = getCmdOption(argv, argv + argc, "-load-tables");

		data.loadTables( tablesFile );
	}
	else if( cmdOptionExists(argv, argv + argc, "-load-table") )
	{
		char* tablesFile = getCmdOption(argv, argv + argc, "-load-table");

		data.loadTables( tablesFile );
	}
	else
	{
		const char* tablesFile = DEFAULT_TABLE_FILE;

		data.loadTables( tablesFile );
	}

	data.openFiles( datasetFileName, histogramsFileName );

	cvtColor(old, old, CV_RGB2RGBA, 4);

	int *bests = new int[(hight / 32) * (width / 32)];
cout << "Running on image " << input << endl;
	if( cmdOptionExists(argv, argv + argc, "-gpu") )
	{
		cout << "start mosaic on GPU:" << endl;
		mosaic_GPU( old.data, bests, data, hight, width );
	}
	else if( cmdOptionExists(argv, argv + argc, "-gpufs") )
	{
		cout << "start mosaic on GPUFS:" << endl;
		mosaic_GPUfs( old.data, bests, data, hight, width );
	}
	else if( cmdOptionExists(argv, argv + argc, "-cpu") )
	{
		cout << "start mosaic on CPU:" << endl;
		mosaic_CPU( old.data, bests, data, hight, width );
	}
	else if( cmdOptionExists(argv, argv + argc, "-warp") )
	{
#ifdef WARP
		cout << "start mosaic GPUFS-warp" << endl;
		mosaic_GPUfs_warp( old.data, bests, data, hight, width );
#else
		cout << "Warp not supported in this configuration" << endl;
		exit(-1);
#endif
	}
	else if( cmdOptionExists(argv, argv + argc, "-vm") )
	{
#ifdef WARP
		cout << "start mosaic GPUFS-vm" << endl;
		mosaic_GPUfs_VM( old.data, bests, data, hight, width );
#else
		cout << "VM not supported in this configuration" << endl;
		exit(-1);
#endif
	}
	else
	{
		mosaic_CPU( old.data, bests, data, hight, width );
	}

	cout << "Finished mosaic" << endl;

	cout << "Create output image" << endl;

	Mat mosaic(hight, width, CV_8UC4);

	int i = 0;
	for( int by = 0; by < hight; by += 32 )
	{
		for( int bx = 0; bx < width; bx += 32 )
		{
			uchar imageData[IMAGE_SIZE_ON_DISK];
			size_t offset = (size_t)bests[i] * IMAGE_SIZE_ON_DISK;
			assert( offset >= 0 );

			fseek(data.imagesFile, offset, SEEK_SET);
			size_t read = fread(imageData, IMAGE_SIZE_ON_DISK, 1, data.imagesFile);
			if( read != 1 )
			{
				cout << "Offset: " << offset << " index: " << bests[i] << endl;
			}
			assert( read == 1 );

			Mat newBlockMat(32, 32, CV_8UC4, imageData);
			Mat tmp = mosaic(Rect(bx,by,32,32));
			newBlockMat.copyTo(tmp);

			i++;
		}
	}

	imwrite(output, mosaic);
}
