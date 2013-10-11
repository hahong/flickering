#!/bin/bash

while [ 1 ]; do
	ssh -R :12188:localhost:18888 -N devthor@dicarlo2.mit.edu
	echo Retrying...
done
