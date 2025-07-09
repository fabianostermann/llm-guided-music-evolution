#!/bin/bash

tmpfile=".evoclamp-tmpfile-"$(date +%N)
		
for abcfile in $@
do
	echo "X:0" > $tmpfile; cat $abcfile >> $tmpfile
	abcm2ps $tmpfile -O $abcfile.ps
	abc2midi $tmpfile -o $abcfile.mid
done

rm $tmpfile

