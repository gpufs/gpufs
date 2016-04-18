
# this script runs 1 iteration of every configuration of different matrix sizes
# 

# rm experiment_res; rm all_results

for i in 104 208 416 832 1664 3328; do
	for j in 64 128; do
	    for k in 64 128 256;  do
		echo RUN HA=$i  WA=$j  WB=$k ./mm >> experiment_res
		for z in 0 1 2; do
			NUM_ITER=1 HA=$i  WA=$j  WB=$k ./mm 2>&1 | tee -a all_results |grep RESULT >>experiment_res
		done
	   done
	done
done
