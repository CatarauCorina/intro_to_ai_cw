#!/bin/bash

foldersplit='./img_alig_split'
cd $foldersplit

for i in *; do
  count=$(ls $i | wc -l)
  if [ $count -lt 14 ]; then
    echo $count
    cp -r $i '../img_alig_split_lt_15'
  fi
done
