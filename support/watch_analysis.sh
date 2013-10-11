#!/bin/bash

if [ $# -lt 1 ]; then
	echo 'Usgage: watch_analysis.sh <watch dir> [options] [...]'
	exit 1
fi

srcdir="$1"                    # source watch dir
wrkdir="$srcdir/.working"      # working dir
outdir="$srcdir/output"        # output dir
donedir="$srcdir/done"         # output dir
faildir="$srcdir/failed"       # output dir

while true
do
	sleep 1
	mkdir -p "$srcdir"
	mkdir -p "$wrkdir"
	mkdir -p "$outdir"
	mkdir -p "$donedir"
	mkdir -p "$faildir"
	chmod 777 "$srcdir" "$outdir" "$donedir" "$faildir"

	# -- check if there's any target file
	if ! ls "$srcdir"/*.tiff &> /dev/null; then
		# no target file not exists
		continue
	fi
	
	#DBG echo '** tiff exists'
	# -- make sure no file activity
	while true
	do
		fntmp1=`mktemp /tmp/fsev1.XXXXXX`
		fntmp2=`mktemp /tmp/fsev2.XXXXXX`

		ls -lT "$srcdir" > $fntmp1
		sleep 1
		ls -lT "$srcdir" > $fntmp2

		if diff $fntmp1 $fntmp2 &> /dev/null; then
			rm -f $fntmp1 $fntmp2
			break
		fi
		rm -f $fntmp1 $fntmp2
		#DBG echo '** file activity dectected'
	done
	
	# -- check one more time if there's any target file
	if ! ls "$srcdir"/*.tiff &> /dev/null; then
		# no target file not exists
		continue
	fi

	# -- begin analysis
	#DBG echo 'do the analysis'
	files=("$srcdir"/*.tiff)
	fno="${files[0]}"
	fnb="`basename "$fno"`"
	fnb0="`basename "$fno" .tiff`"
	fnlog="${fnb0}.log"

	mv "$fno" $wrkdir
	echo '** Processing:' $fnb
	touch "$srcdir/00_WORKING"
	if fa.py $2 $3 $4 $5 $6 $7 $8 $9 "$wrkdir/$fnb" "$outdir/$fnb0" &> "$outdir/$fnlog"; then
		# success
		mv "$wrkdir/$fnb" "$donedir"
	else
		mv "$wrkdir/$fnb" "$faildir"
	fi
	rm -f "$srcdir/00_WORKING"
done
