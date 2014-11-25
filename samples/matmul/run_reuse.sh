
# this run runs 10 iterations of the same matrix product including reading/writing input/output files
# it is used to estimate the benefits of a cross-application buffer cache when files get reused across different applications


rm experiment_res; rm all_results

for i in 104 208 416 832 1664 3328; do
	for j in 64 128; do
	    for k in 1 2 4 8;  do
		echo RUN HA=$i  WA=$j  WB=$k ./mm >> experiment_res
		for z in 0 1 2; do
			NUM_ITER=10 HA=$i  WA=$j  WB=$k ./mm 2>&1 | tee -a all_results |grep RESULT >>experiment_res
		done
	   done
	done
done
