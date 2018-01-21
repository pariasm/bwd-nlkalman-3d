#!/bin/bash
# Tune the algorithm's parameters

# noise levels
sigmas=(10 20 40 40)

# fixed parameters
pxs=(8 6)
pts=(1 2)
wxs=(3 4 5 6)
wts=(4 3 2 1)

# number of trials
ntrials=1000

# test sequences
seqs=(\
derf-hd/park_joy \
derf-hd/speed_bag \
derf-hd/station2 \
derf-hd/sunflower \
derf-hd/tractor \
)

# seq folder
#sf='/mnt/nas-pf/'
sf='/home/pariasm/denoising/data/'

output=${1:-"trials"}

export OMP_NUM_THREADS=1

# we assume that the binaries are in the same folder as the script
BIN=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo $BIN

for ((i=0; i < $ntrials; i++))
do
	# randomly draw noise level and parameters

	# noise level
	r=$(awk -v M=4 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*M)}')
	s=${sigmas[$r]}

	# patch size
	r=$(awk -v M=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*M)}')
	px=${pxs[$r]}
	pt=${pts[$r]}

	# search region
#	r=$(awk -v M=3 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*M)}')
#	wx=${wxs[$r]}
#	wt=${wts[$r]}
	wx=3
	wt=4

	# spatial and temporal weights
	dth=$(awk -v M=60 -v S=5 -v s=$RANDOM 'BEGIN{srand(s); print rand()*(M - S) + S}')
	bx=$(awk -v M=4 -v s=$RANDOM 'BEGIN{srand(s); print rand()*M}')

	# format as string
	s=$(printf "%02d" $s)
	px=$(printf "%d" $px)
	pt=$(printf "%d" $pt)
	wx=$(printf "%d" $wx)
	wt=$(printf "%d" $wt)
	dth=$(printf "%04.1f" $dth)
	bx=$(printf "%3.1f" $bx)

	trialfolder=$(printf "$output/s%sp%sx%sw%sx%sdth%sbx%s\n" \
		$s $px $pt $wx $wt $dth $bx)

	params=$(printf " -p %s --patch_t %s -w %s --search_t %s --dth %s --beta_x %s" \
		$px $pt $wx $wt $dth $bx)

	mpsnr=0
	nseqs=${#seqs[@]}
	ff=70
	lf=85
	if [ ! -d $trialfolder ]
	then
		for seq in ${seqs[@]}
		do
			echo  "$BIN/rnldct-train.sh ${sf}${seq} $ff $lf $s $trialfolder \"$params\""
			psnr=$($BIN/rnldct-train.sh ${sf}${seq} $ff $lf $s $trialfolder  "$params")
			mpsnr=$(echo "$mpsnr + $psnr/$nseqs" | bc -l)
		done
	fi
	
	printf "$s $px $pt $wx $wt $dth $bx %7.4f\n" $mpsnr >> $output/table
	rm $trialfolder/*.tif
done
