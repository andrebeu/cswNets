#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/cswNets"

declare -a arr=(10 25 50 75 90)

## now loop through the above array
for i in {1..25}; do 
	for PRSHIFT in "${arr[@]}"; do 
		sbatch ${wd_dir}/gpu_jobsub.cmd "${PRSHIFT}" 
	done
done
