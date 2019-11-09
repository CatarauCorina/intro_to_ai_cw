#!/bin/bash
filename='identity_CelebA.txt'
foldertocreate='img_alig_split'
imgfolder='img_align_celeba'
exec 4<$filename
echo Start
while read -u4 p ; do
    stringarray=($p)
    mkdir -p $foldertocreate/${stringarray[1]}
    echo ${stringarray[0]}
    if [ -f "$imgfolder/${stringarray[0]}" ]; then
      mv $imgfolder/${stringarray[0]} $foldertocreate/${stringarray[1]}
    fi
done