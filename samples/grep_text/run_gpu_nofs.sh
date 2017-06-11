./fs_gpu_nofs  1 28 128 input/dict.gpu out db_shakespere.txt > gpu_nofs 2>&1 
cat gpu_nofs |sort |  awk '{if ($1==old){count+=$2;}else{ print old" "count" "$3; old=$1;count=$2; }}' |sort > gpu_nofs.sorted
#cat gpu_nofs | sort | awk 'BEGIN{c=0; w=0;}{if ($1!=w) { print w" "c; c=$2;w=$1;} else{ c+=$2;}}' > res_gpu_nofs
