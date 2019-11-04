#!/bin/bash
foldersplit='img_alig_split'
cd $foldersplit
for i in *; do
  echo $i
  v2=${i::${#i}-1}
  mv $i $v2;
done

