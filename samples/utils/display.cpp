#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
       #include <sys/stat.h>
       #include <fcntl.h>
#include <string.h>
#include <stdlib.h>


int main(int argc, char** argv)
{
	if (argc<2) { printf("<input file 32 byte + 4 byte  >\n"); return -1; }

	char* in=argv[1];

	int in_fd=open(in,O_RDONLY);
	assert(in_fd);
	char line[36+64];
	
	int size=0;
	size_t n;
	while(read(in_fd,line,36+64)>0){	
		fprintf(stdout,"%s %d %s\n",line,*((int*)&line[32]),line+36);
	}
	
	close(in_fd);
	return 0;
}
