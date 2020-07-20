#!/bin/bash

declare -a blocklen_arr=(1 40)
declare -a stsize_arr=(20 25 30 35 40)
declare -a lr_arr=('0.005' '0.01' '0.05' '0.1')

## now loop through the above array

for stsize in "${stsize_arr[@]}"; do 
	for lr in "${lr_arr[@]}"; do 
    for blocklen in "${blocklen_arr[@]}"; do 
			python cswnets19-regression-sweep2.py "${blocklen}" "${stsize}" "${lr}" 
		done
	done
done
