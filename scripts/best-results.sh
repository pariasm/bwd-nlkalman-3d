#!/bin/bash
# This script is used to generate the best results found during 
# training, with the aim of inspecting the visual quality.

# we assume that the binaries are in the same folder as the script
BIN=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# test sequences
seqs=(\
derf-hd/park_joy \
derf-hd/speed_bag \
derf-hd/station2 \
derf-hd/sunflower \
derf-hd/tractor \
)

# seq folder
sf='/home/pariasm/Remote/lime/denoising/data/'

# first and last frame
f0=70
f1=85

# function to run the algorithm
function run_rnlm {

	s="$1"
	px="$2"
	pt="$3"
	dth="$4"
	bx="$5"
	wx=3
	wt=4

	# format as string
	s=$(printf "%02d" $s)
	px=$(printf "%d" $px)
	pt=$(printf "%d" $pt)
	wx=$(printf "%d" $wx)
	wt=$(printf "%d" $wt)
	dth=$(printf "%04.1f" $dth)
	bx=$(printf "%3.1f" $bx)

	trialfolder=$(printf "s%sp%sx%sw%sx%sdth%sbx%s\n" \
		$s $px $pt $wx $wt $dth $bx)

	params=$(printf " -p %s --patch_t %s -w %s --search_t %s --dth %s --beta_x %s" \
		$px $pt $wx $wt $dth $bx)

	nseqs=${#seqs[@]}
	for seq in ${seqs[@]}
	do
		echo $BIN/rnldct-train.sh ${sf}${seq} $f0 $f1 $s $trialfolder/$seq \"$params\"
		time $BIN/rnldct-train.sh ${sf}${seq} $f0 $f1 $s $trialfolder/$seq  "$params"
	done

}

# run with optimal parameters
run_rnlm 10 8 1 13.9 1.6
run_rnlm 20 8 1 18.7 1.9
run_rnlm 40 8 1 28.2 2.5

run_rnlm 10 6 2 20.2 0.7
run_rnlm 20 6 2 25.1 0.8
run_rnlm 40 6 2 35.0 1.0

# run varying the optimal parameters
#                                   #       bus_mono    football_mono
#                                   #       PSNR 29.27  PSNR 28.54
# run_rnlm 20 8 10 35.0 1.2 4.5 1.0 # dth-- PSNR 28.58  PSNR 28.39
# run_rnlm 20 8 10 55.0 1.2 4.5 1.0 # dth++ PSNR 29.30  PSNR 28.50
# run_rnlm 20 8 10 45.0 0.5 4.5 1.0 # b_x-- PSNR 29.20  PSNR 28.50
# run_rnlm 20 8 10 45.0 4.2 4.5 1.0 # b_x++ PSNR 29.25  PSNR 28.57
# run_rnlm 20 8 10 45.0 1.2 2.5 1.0 # b_t-- PSNR 28.91  PSNR 28.53
# run_rnlm 20 8 10 45.0 1.2 6.5 1.0 # b_t++ PSNR 29.22  PSNR 28.31

# conclusions
# dth : tiene que estar cerca de 40 (lo cual parece un umbral muy grande)
#       por debajo de este umbral aparecen empieza a aparecer ruido en algunos
#       patches. 
# b_x : denoising espacial. La incidencia del denoising espacial dura pocos frames
#       (4 frames?). Por medio del promediado temporal, la secuencia o bien
#       gana detalle, si se empieza de un frame demasiado suave, o bien pierde 
#       ruido 
# b_t : controla el promediado temporal.
# l_x : aparentemente no tiene mucha influencia

