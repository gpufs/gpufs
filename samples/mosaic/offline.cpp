/*
 * test.cpp
 *
 *  Created on: Aug 6, 2014
 *      Author: sagi
 */

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <limits.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LSH.h"

using namespace std;
using namespace cv;

extern int seed;

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

static const string DEFAULT_DATASET = "tiny_2M";

int main(int argc, char** argv)
{
	string datasetName = DEFAULT_DATASET;
	if( cmdOptionExists(argv, argv + argc, "-dataset") )
	{
		datasetName = getCmdOption(argv, argv + argc, "-dataset");
	}

	uint D = HIST_BINS_PER_CHANEL * HIST_NUM_CHANNELS;
	uint K = 19;
	uint L = 32;
	uint TableSize = 1000000;

	LSH* lsh = NULL;

	if( cmdOptionExists(argv, argv + argc, "-seed") )
	{
		seed = atoi(getCmdOption(argv, argv + argc, "-seed"));
	}

	if( cmdOptionExists(argv, argv + argc, "-load") )
	{
		char* paramFile = getCmdOption(argv, argv + argc, "-load");
		ifstream stream(paramFile);

		lsh = new LSH(stream, datasetName);
	}
	else
	{
		if( cmdOptionExists(argv, argv + argc, "-K") )
		{
			K	= atoi(getCmdOption(argv, argv + argc, "-K"));
		}

		if( cmdOptionExists(argv, argv + argc, "-L") )
		{
			L 	= atoi(getCmdOption(argv, argv + argc, "-L"));
		}

		lsh = new LSH(D, 	// D
					  K,		// K
					  L,		// L
					  TableSize,
					  datasetName);
	}

	if( cmdOptionExists(argv, argv + argc, "-dump") )
	{
		char* paramFile = getCmdOption(argv, argv + argc, "-dump");
		ofstream stream(paramFile);

		lsh->dump(stream);
		stream.close();

		return 0;
	}

	if( cmdOptionExists(argv, argv + argc, "-load-table") )
	{
		assert( false );
	}
	else if( cmdOptionExists(argv, argv + argc, "-load-tables") )
	{
		assert( false );
	}
	else
	{
		// Populate database
		lsh->populate();
	}

	if( cmdOptionExists(argv, argv + argc, "-info") )
	{
		lsh->printStatistics();
	}

	if( cmdOptionExists(argv, argv + argc, "-dump-tables") )
	{
		char* tableFile = getCmdOption(argv, argv + argc, "-dump-tables");

		lsh->dumpTables(tableFile);

		return 0;
	}
	else if( cmdOptionExists(argv, argv + argc, "-dump-table") )
	{
		char* tableFile = getCmdOption(argv, argv + argc, "-dump-table");

		lsh->dumpTables(tableFile);

		return 0;
	}

	delete lsh;

	return 0;
}

