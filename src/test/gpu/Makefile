NVCC = nvcc
PROJ_DIR = ../..
LIB_DIR = ${PROJ_DIR}/../lib

CUB_DIR = ${LIB_DIR}/cub
MGPU_DIR = ${LIB_DIR}/moderngpu
CNMEM_DIR = ${LIB_DIR}/cnmem
CUDPP_DIR = ${LIB_DIR}/cudpp
GTEST_DIR = /home/wentian/programs/googletest/googletest

PROJ_INCLUDE = $(shell find ${PROJ_DIR} -type d | sed s/^/-I/)
LIB_INCLUDE = -I${CUB_DIR} -I${GTEST_DIR}/include -I${MGPU_DIR}/src -I${CNMEM_DIR}/include -I${CUDPP_DIR}/include
INCLUDE = ${PROJ_INCLUDE} ${LIB_INCLUDE} 

CUDPP_LIB = -L${CUDPP_DIR}/lib -lcudpp -lcudpp_hash
GTEST_LIB = -L${GTEST_DIR} -lgtest 
CNMEM_LIB = -L${CNMEM_DIR}/build -lcnmem
LIBS = ${GTEST_LIB} ${CNMEM_LIB} ${CUDPP_LIB} 

#MGPU_SRC = ${MGPU_DIR}/src/mgpucontext.cu ${MGPU_DIR}/src/mgpuutil.cpp ${MGPU_DIR}/src/mmio.cpp ${MGPU_DIR}/src/sparsematrix.cpp

ARCH=\
		 -gencode arch=compute_61,code=sm_70
		 #-gencode arch=compute_61,code=compute_61

MGPU_FLAGS = --expt-extended-lambda -use_fast_math -Wno-deprecated-declarations
OPENMP_FLAGS=-DOPENMP -Xcompiler -fopenmp -lgomp

CFLAGS += -O3 -std=c++11 \
					-DPROFILE -DPTHREAD_LOCK ${MGPU_FLAGS} ${OPENMP_FLAGS} ${ARCH} 
					#-lineinfo -DCACHED_SET_INTERSECT
					#-DSET_INTERSECT_PROFILE 
					#-DHASH_PROFILE \
					-DBLOOM_FILTER_BLOCK_BST_SEARCH_PROFILE

all: test_utils test_pattern_match test_cuda_memory 
build: test_utils test_pattern_match test_cuda_memory 

test_utils: TestUtils.cu
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ ${MGPU_SRC} $^ $(LIBS)

test_pattern_match: TestPatternMatchGPU.cu
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ ${MGPU_SRC} $^ $(LIBS)

test_cuda_memory: TestCudaMemory.cu
	$(NVCC) $(CFLAGS) $(INCLUDE) -o $@ ${MGPU_SRC} $^ $(LIBS)

clean:
	rm -rf test_utils test_pattern_match test_cuda_memory *.o 
