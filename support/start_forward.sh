#!/bin/bash

while [ 1 ]; do
	ssh -R 12122:localhost:22 -N devthor@dicarlo2.mit.edu
	echo Retrying...
done
