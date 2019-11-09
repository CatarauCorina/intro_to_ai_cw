#!/bin/bash
crtdir='.'
foldersplit='../img_alig_split'
cd $foldersplit
#counts=$(find . -maxdepth 1 -type d -exec bash -c "echo -ne '{} '; ls '{}' | wc -l" \; | awk '$NF>=15')
#echo $counts
for i in *; do
  count=$(ls $i | wc -l)
  if [ $count -gt 15 ]; then
    echo $count
    cp -r $i '../img_alig_split_gt_15'
  fi
done
