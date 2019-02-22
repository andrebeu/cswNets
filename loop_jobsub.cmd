#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/cswNets"

declare -a stsize_arr=(5 10 20 50)
declare -a shift_arr=(10 50 90)
declare -a cswpr_arr=(90 80 70 60)

## now loop through the above array
for i in {1..2}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for prshift in "${shift_arr[@]}"; do 
			for cswpr in "${cswpr_arr[@]}"; do 
				printf "st ${stsize} csw ${cswpr} shift ${prshift}\n"
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "${cswpr}" "${prshift}" 
			done
		done
	done
done
