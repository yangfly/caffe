if [ -d build ]; then
    rm -rf build
fi

# cmake find nccl
export NCCL_DIR=$HOME/.software/nccl/build

mkdir build && cd build
cmake \
    -Dpython_version=3 \
    -DUSE_NCCL=ON \
    -DUSE_LEVELDB=OFF \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_only_tests="bbox_util,detection_evaluate_layer,detection_output_layer,im_transforms,normalize_layer,permute_layer,prior_box_layer,annotated_data_layer,multibox_loss_layer" \
    ..

make -j $(nproc)
# make runtest -j $(nproc)
make pycaffe -j $(nproc)
make pytest -j $(nproc)
make install
