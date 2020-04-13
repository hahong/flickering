#!/bin/bash

projroot="/mnt"
trap ctrl_c SIGINT

function ctrl_c() {
	echo
	echo -e '\e[43m\e[30mInfo:\e[49m\e[39m Prepare to halt...'
	touch $projroot/Full/00_HALT
	touch $projroot/Fast/00_HALT
	while true
	do
		if [ ! -e "$projroot/Full/00_WORKING" -a ! -e "$projroot/Fast/00_WORKING" ]; then
			break
		fi
		echo -e '\e[43m\e[30mInfo:\e[49m\e[39m Waiting tasks to finish...'
		killall python
		sleep 5
	done
	echo -e '\e[43m\e[30mInfo:\e[49m\e[39m Halting...'
	echo '-------------------------------------------------------------------------------'
	poweroff
}

clear
if ! mount | grep flickering &> /dev/null; then
	echo -e '\e[42m\e[30mInfo:\e[49m\e[39m Mounting shared volume.'
	if ! mount -t vboxsf flickering $projroot; then
		echo -e '\e[41m\e[30mError:\e[49m\e[39m Cannot access shared drive. Halting in 60s. (Ctrl+C to halt immediately)'
		sleep 60
		poweroff
	fi
fi

mkdir -p $projroot/Full
mkdir -p $projroot/Fast
if [ ! -d $projroot/Full -o ! -d $projroot/Fast ]; then
	echo -e '\e[41m\e[30mError:\e[49m\e[39m Cannot access watch folder(s). Halting in 60s. (Ctrl+C to halt immediately)'
	sleep 60
	poweroff
fi       
rm -f $projroot/Full/00_HALT
rm -f $projroot/Fast/00_HALT

echo -e '\e[42m\e[30mReady:\e[49m\e[39m Initialization done.'
export PATH=$PATH:/root/flickering/scripts:/root/flickering/support
export PYTHONPATH=/root/flickering
watch_analysis.sh $projroot/Fast/ --n_jobs=-1 --verbose=1 &
watch_analysis.sh $projroot/Full/ --n_jobs=-1 --verbose=1 --full &

while true
do
	echo -e '\e[42m\e[30mInfo:\e[49m\e[39m Press Ctrl+C to completely halt the service.'
	sleep 14400
done
wait
poweroff
