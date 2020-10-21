# Parallel Subgraph Enumeration on GPUs and CPUs

This project provides a framework for parallel subgraph enumeration utilizing GPUs and multi-core CPUs on a single machine [1,2].

## File organization

- `scripts`: provide some useful scripts to preprocess the data graphs and run the programs.
- `lib`: the library used in this project.
- `src/cpu`: the implementation on multi-core CPUs.
- `src/gpu`: the implementation on GPUs.
- `src/preprocess`: the utils to preprocess the data graphs. 

## Prerequisites
### Common dependency
- gtest (optional): needed to compile the `src/test` folder.

### CPU dependency
- g++ 5.4.0

### GPU dependency
You need the following deps only if you want to run GPU solutions.
- CUDA 10.0
- [moderngpu 1.0](https://github.com/guowentian/moderngpu)
- [cnmem](https://github.com/guowentian/cnmem)
- [cudpp](https://github.com/guowentian/cudpp) (optional)
- [cub 1.8](https://github.com/guowentian/cub)
- [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) (optional)

#### Build GPU dependency
The GPU deps except for CUDA are configured as submodule in the `lib` folder.
You can easily fetch the respositories by the following command.
```
git submodule init
git submodule update
```
To compile moderngpu
```
cd lib/moderngpu; make -j
```
To compile cnmem 
```
cd lib/cnmem 
mkdir build 
cmake -DCMAKE_INSTALL_PREFIX={your_path_to_cnmem} ..
make -j
make install
```
Add the path to the `libcnmem.so` to your LD_LIBRARY_PATH.

To compile cudpp, refer to 
[this guide](http://cudpp.github.io/cudpp/2.3/building-cudpp.html) and 
[this guide](https://github.com/cudpp/cudpp/wiki/BuildingCUDPPwithCMake).
```
cd lib/cudpp 
git submodule init
git submodule update
mkdir build
cmake -DCMAKE_INSTALL_PREFIX={your_path_to_cudpp} ..
make -j
make install
```

## Compile 
After configuring the dependencies, simply go to the folder `src/cpu`, `src/gpu`, `src/preprocess`,
and `make -j`. 

Note that the solutions for CPUs and GPUs are separated. So you can still compile and run the CPU solutions even if you don't have CUDA on your machine. 

### GPU gencode
Depending on your GPU architecture, you may need to 
- In [src/gpu/Makefile](src/gpu/Makefile), add `-gencode arch=compute_61,code=sm_xx` to `ARCH`.
- In [src/gpu/utils/context/MGPUContext.cuh](src/gpu/utils/context/MGPUContext.cuh), add `mgpu::arch_xx_cta<128,8,8>` to `MGPULaunchBox` and `mgpu::arch_xx_cta<128,1>` to `MGPULaunchBoxVT1`. 

If your arch is not in [lib/moderngpu/src/moderngpu/launch_box.hxx](lib/moderngpu/src/moderngpu/launch_box.hxx), you need to configure there also.

## Run

### Getting started
[scripts/demo.sh](scripts/demo.sh) shows an example of how to run the preprocessing, CPU and GPU solutions.

### Data formats
Our framework supports two formats of data graphs.

- SNAP: the edge list input. The datasets downloaded from [SNAP website](https://snap.stanford.edu/data/) adopt this format. 
- BIN: our preprocessed binary format. The BIN format reorganizes the vertex ids into the continous integer starting from 0. It removes any self loops and parallel edges if needed. The BIN file is written in binary to improve read/write speed. 

### Queries
The set of queries supported can be seen in [doc/queries.pdf](doc/queries.pdf).

#### Register new queries
1. In [src/query/common/Query.h](src/query/common/Query.h), add a new function specifying the structure of a new pattern. Refer to other patterns in the same file as examples. 
2. In [src/Meta.h](src/Meta.h), add your new query under `QueryType`.
3. Recompile.

### Preprocess
See [src/preprocess/README.md](src/preprocess/README.md) for details.

[scripts/preprocess.h](scripts/preprocess.h) provide some example runs.

### Run GPU solutions
See [src/gpu/README.md](src/gpu/README.md).

### Run CPU solutions
See [src/cpu/README.md](src/cpu/README.md).

## References

[1]. GPU-Accelerated Subgraph Enumeration on Partitioned Graphs. SIGMOD 2020. [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3318464.3389699?casa_token=6YJkJ4c7b_UAAAAA:JbNWDytqd6kY8hdktAp0FQsXGTFTaWQxAih16Q-lJZd_qzKlE3TV06HOB1brW9ThFqllWR9FqRY) [\[Bib entry\]](https://scholar.googleusercontent.com/scholar.bib?q=info:F6fuEJ0tqPIJ:scholar.google.com/&output=citation&scisdr=CgXYWi02EL6ftXcuxRA:AAGBfm0AAAAAX5Ar3RD-F_5o4Eu-2ejzNDHMIG7taZup&scisig=AAGBfm0AAAAAX5Ar3b8tdj-daz05wgRPHlYYWPf_O-GH&scisf=4&ct=citation&cd=-1&hl=en)

[2]. Exploiting Reuse for GPU Subgraph Enumeration. Under submission.
