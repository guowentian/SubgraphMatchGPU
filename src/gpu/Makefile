NVCC = nvcc
PROJ_DIR = ..
LIB_DIR = ${PROJ_DIR}/../lib

CUB_DIR = ${LIB_DIR}/cub
MGPU_DIR = ${LIB_DIR}/moderngpu
CNMEM_DIR = ${LIB_DIR}/cnmem

PROJ_INCLUDE = $(shell find ${PROJ_DIR} -type d | sed s/^/-I/)
LIB_INCLUDE = -I${CUB_DIR} -I${MGPU_DIR}/src -I${CNMEM_DIR}/include 
INCLUDE = ${PROJ_INCLUDE} ${LIB_INCLUDE} 

CNMEM_LIB = -L${CNMEM_DIR}/build -lcnmem
LIBS = ${CNMEM_LIB} 
 
MGPU_FLAGS = --expt-extended-lambda -use_fast_math -Wno-deprecated-declarations
OPEN_FLAGS=-DOPENMP -Xcompiler -fopenmp -lgomp -lpthread
# To support backtrace, which is useful for debugging
BACKTRACE_FLAGS = -Xcompiler -rdynamic

# PROFILE: print out verbose info during execution, which is believed not cause large overheads in execution. The information includes memory size, #instances, intra-partition ratios and etc.
# CUDA_CONTEXT_PROFILE: profile my memory allocator
# REUSE_PROFILE: profile reuse related info
# UNTRACK_ALLOC_LARGE_VERTEX_NUM: for a large graph with a large number of vertices, use UnTrackMalloc to allocate the data structures in GraphDevTracker to avoid reach the GPU memory limit. This is just a temporary fix. Need to have generalized implementation.
# NVPROFILE: for nvprof focused profiling

CFLAGS += -O3 -std=c++11 -DUNTRACK_ALLOC_LARGE_VERTEX_NUM -DPTHREAD_LOCK \
					-DPROFILE # -DCUDA_CONTEXT_PROFILE \
					#-DDEBUG  \
					#-DREUSE_PROFILE

ARCH = -gencode arch=compute_61,code=sm_70 -gencode arch=compute_61,code=compute_61

all: patternmatchgpu
build: patternmatchgpu

patternmatchgpu: PatternMatchGPU.cu
	$(NVCC) $(CFLAGS) ${OPEN_FLAGS} ${MGPU_FLAGS} ${ARCH} $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf patternmatchgpu *.o 
