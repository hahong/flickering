#!/bin/bash

rsync -av --exclude '*~' --exclude '.*' /mnt/flickering/* ./
