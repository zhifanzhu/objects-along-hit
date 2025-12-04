mkdir -p externals

# install manopth
pip install "git+https://github.com/hassony2/manopth.git"

# install neural_renderer for HOMAN
git clone https://github.com/hassony2/multiperson.git externals/multiperson
CC=gcc-9 CXX=g++-9 pip install externals/multiperson/neural_renderer --no-build-isolation

# install sdf
CC=gcc-9 CXX=g++-9 pip install git+https://github.com/zhifanzhu/sdf_pytorch --no-build-isolation

# install Roma rotation library
pip install git+https://github.com/naver/roma@22806dfb43329b9bf1dd2cead7e96720330e3151 --no-build-isolation


# Install torch_scatter (we use scatter_min)
# E.g. for python3.10 torch2.0.0 cuda 11.8:
pip install \
    https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl \
    --no-build-isolation


# Luckily there exist certain pre-build wheels for pytorch3d
# e.g. for python3.10 torch2.0.0 cuda 11.8, pytorch 0.7.3
#  also see: https://miropsota.github.io/torch_packages_builder/pytorch3d/  but untested
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/pytorch3d-0.7.3-cp310-cp310-linux_x86_64.whl