#!/bin/bash

# fetch md5 item
function fetch()
{
  md5=$1
  item=$2
  file=${item#*/}

  ## Download
  URL_ROOT=http://host.robots.ox.ac.uk/pascal/VOC
  if [ ! -f $file ]; then
    echo "Downloading $file ..."
    wget $URL_ROOT/$item -O $file
  fi

  ## Checksum
  if [ -f $file ]; then
    os=`uname -s`
    if [ "$os" = "Linux" ]; then
      checksum=`md5sum $file | awk '{ print $1 }'`
    elif [ "$os" = "Darwin" ]; then
      checksum=`cat $file | md5`
    fi
    if [ "$checksum" = "$md5" ]; then
      echo "Checksum is correct."
    else
      echo "Checksum is incorrect."
      exit 0
    fi
  else
    echo "download failure."
    exit 0
  fi

  ## Extract
  tar -xf $file
}

cd data
fetch b6e924de25625d8de591ea690078ad9f  voc2007/VOCtest_06-Nov-2007.tar
fetch c52e279531787c972589f7e41ab4ae64  voc2007/VOCtrainval_06-Nov-2007.tar
fetch 6cd6e144f989b92b3379bac3b3de84fd  voc2012/VOCtrainval_11-May-2012.tar
cd ..
