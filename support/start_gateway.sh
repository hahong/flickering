#!/bin/bash

sleep 150
ssh -R :$2:localhost:$1 -N devthor@dicarlo2.mit.edu
