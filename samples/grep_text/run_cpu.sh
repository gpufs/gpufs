#!/bin/bash
set -x
#sudo su -c 'echo 3 > /proc/sys/vm/drop_caches'
t=`date +%s`
OMP_NUM_THREADS=8  ./grep_text_cpu  input/dict.gpu  asdf db_shakespere.txt  > res_shakespere 2>&1 
t1=`date +%s`
echo $t1-$t | bc -l > cpu_timing
cat res_shakespere | awk '{print $1" "$3" "$7;}' | sort  > res_cpu
