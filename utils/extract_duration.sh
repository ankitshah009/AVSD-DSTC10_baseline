#!/bin/bash

### 
# $1 - here is the list of the files for which the duration needs to be extracted 
# $2 - Location where the files are located (actual location/ can also be relative path)
# $3 - output file to be generated for the filename and duration information. 
###

###
# Sample run command - bash ~/extract_duration.sh ~/Charades_v1_480  $PWD/Charades_v1_480/  ~/duration_Charades_v1_480 & 
###


for i in `cat $1`
	do
		echo "$i" `ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 $2/$i` >> "$3".csv
	done

