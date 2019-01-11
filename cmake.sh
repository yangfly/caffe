if [ -d build ]; then
    rm -rf build
fi
mkdir build && cd build
cmake \
    -Dpython_version=3 \
    -DUSE_LEVELDB=OFF \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_BUILD_TYPE=Release \
    ..

make -j $(nproc)
make runtest -j $(nproc)
make pycaffe -j $(nproc)
# make pytest -j $(nproc)
# make install

