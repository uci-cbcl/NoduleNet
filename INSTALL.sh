#/usr/bin/bash

set -e
WORKSPACE=`pwd`
NVI_CP="/usr/local/cuda-9.0/bin/nvcc" # Absolute path to your nvcc(symbolic link not supported)

#Compile nms
cd net/lib/box/nms/torch_nms/src
"$NVI_CP"  -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ..
python build.py

#Compile overlap
cd $WORKSPACE
cd net/lib/box/overlap/torch_overlap/src
"$NVI_CP"  -c -o overlap_kernel.cu.o overlap_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ..
python build.py

cd $WORKSPACE
