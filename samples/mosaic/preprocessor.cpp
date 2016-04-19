
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef unsigned int uint;
typedef unsigned char uchar;

using namespace std;
using namespace cv;

static const uint IN_IMAGE_SIZE = 32 * 32 * 3;
static const uint OUT_IMAGE_SIZE = 32 * 32 * 4;

static const uint HIST_BINS_PER_CHANEL = 256;
static const uint HIST_VALS_PER_BIN = 256 / HIST_BINS_PER_CHANEL;
static const uint OUT_HIST_SIZE = 4096;

static const uint DEFAULT_NUM_IMAGES = 2000000U;
static const string DEFAULT_DATASET = "tiny_images.bin";
static const string DEFAULT_OUTPUT_DATASET = "tiny_2M.bin";
static const string DEFAULT_OUTPUT_HISTOGRAMS = "tiny_2M.hist";

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
	string datasetName = DEFAULT_DATASET;
	if( cmdOptionExists(argv, argv + argc, "-dataset") )
	{
		datasetName = getCmdOption(argv, argv + argc, "-dataset");
	}

	string outputDatasetName = DEFAULT_OUTPUT_DATASET;
	if( cmdOptionExists(argv, argv + argc, "-out") )
	{
		outputDatasetName = getCmdOption(argv, argv + argc, "-out");
	}

	string outputHistogramsName = DEFAULT_OUTPUT_HISTOGRAMS;
	if( cmdOptionExists(argv, argv + argc, "-histOut") )
	{
		outputHistogramsName = getCmdOption(argv, argv + argc, "-histOut");
	}

	int numImages = DEFAULT_NUM_IMAGES;
	if( cmdOptionExists(argv, argv + argc, "-num") )
	{
		numImages = atol(getCmdOption(argv, argv + argc, "-num"));
	}

	FILE* imagesFile = fopen(datasetName.c_str(), "r");
	FILE* newImagesFile = fopen(outputDatasetName.c_str(), "w");
	FILE* newHistogramsFile = fopen(outputHistogramsName.c_str(), "w");

	fseek(imagesFile, 0, SEEK_END);
	size_t size = ftell(imagesFile);
	fseek(imagesFile, 0, SEEK_SET);

	uint nInImages = size / IN_IMAGE_SIZE;
	uint nOutImages = MIN( nInImages, numImages );

	uint precent = nOutImages / 100;

	uchar data[ 4096 ];
	uchar imageData[4096];

	float *hist = (float*) malloc(OUT_HIST_SIZE);

	for( uint i = 0; i < nOutImages; ++i )
	{
		if( 0 == (i % precent) )
		{
			cout << (float)i / nOutImages * 100 << "%" << endl;
		}

		size_t read = fread(data, 1, IN_IMAGE_SIZE, imagesFile);
		assert( IN_IMAGE_SIZE == read );

		memset( hist, 0, OUT_HIST_SIZE );

		for (int i = 0; i < 32 * 32; ++i)
		{
			uchar r = data[2 * 32 * 32 + (i % 32) * 32 + (i / 32)];
			uchar g = data[1 * 32 * 32 + (i % 32) * 32 + (i / 32)];
			uchar b = data[0 * 32 * 32 + (i % 32) * 32 + (i / 32)];
			uchar a = 0;	// Alpha channel

			imageData[i * 4 + 0] = r;
			imageData[i * 4 + 1] = g;
			imageData[i * 4 + 2] = b;
			imageData[i * 4 + 3] = a;

			hist[0 * HIST_BINS_PER_CHANEL + ( r / HIST_VALS_PER_BIN ) ] += 1;
			hist[1 * HIST_BINS_PER_CHANEL + ( g / HIST_VALS_PER_BIN ) ] += 1;
			hist[2 * HIST_BINS_PER_CHANEL + ( b / HIST_VALS_PER_BIN ) ] += 1;
		}

		fwrite(imageData, 1, OUT_IMAGE_SIZE, newImagesFile);
		fwrite(hist, 1, OUT_HIST_SIZE, newHistogramsFile);
	}

	fclose(imagesFile);
	fclose(newImagesFile);
	fclose(newHistogramsFile);

	return 0;
}
