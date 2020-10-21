This README explains how to run `patterngraphgpu`. 

Simply run `patterngraphgpu` without arguments would show the helper message as follows.
```
./patternmatchcpu -f FILENAME -d IS_DIRECTRED -a ALGORITHM -q QUERY -e PARTITION_FILENAME -p PARTITION_NUM -t THREAD_NUM -v DEVICE_NUM -m EXECUTE_MODE -o VARIANT [ -l MATERIALIZE -r ENABLE_ORDERING -c LAZY_TRAVERSAL_COMPRESS_LEVEL -x GPU_LIGHT_ITP_VARIANT -y GPU_LIGHT_EXT_VARIANT]
ALGORITHM: ,GPU_GPSM=4,GPU_GPSM_REUSE=5,GPU_LIGHT=6
QUERY: Q0 (TRIANGLE) 0, Q1 (square) 1, Q2 (chordal square) 2, Q3 (4 clique) 3, Q4 (house) 4, Q5 (quad triangle) 5, Q6 (near5clique) 6, Q7 (5 clique) 7, Q8 (chordal roof) 8, Q9 (three triangle) 9, Q10 (solar square) 10, Q11 (6 clique) 11
MODE: HYBRID_CPU_GPU=0, IN_MEMORY_GPU=1, EXTERNAL_GPU=2
variant: O0(SEPARATE WITHOUT SHARING)=0, O1(SEPARATE WITH SHARING)=1, O2(INTERPLAY)=2, O3(COPROCESSING)=3
LAZY_TRAVERSAL_COMPRESS_LEVEL: COMPRESS_LEVEL_MATERIALIZE=0, COMPRESS_LEVEL_NON_MATERIALIZE=1, COMPRESS_LEVEL_NON_MATERIALIZE_OPT=2, COMPRESS_LEVEL_SPECIAL(DEFAULT)=3
GPU_LIGHT_ITP_VARIANT: SINGLE_SEQUENCE=0, SHARED_EXECUTION(DEFAULT)=1
GPU_LIGHT_EXT_VARIANT: EXT_CACHE(DEFAULT)=0, EXT_UMA=1
```

We will focus on explaining GPU_LIGHT here as it is the most performant method and others are baselines. 

## Parameters

### QUERY
The details of each query is defined in [src/query/common/Query.h](../query/common/Query.h).

### MODE

- HYBRID_CPU_GPU: The execution that involves both CPUs and GPUs and the main storage is main memory.
- IN_MEMORY_GPU: The whole enumeration process is finished by GPUs only and the main storage is GPU memory. 
- EXTERNAL_GPU: The whole enumeration process is finished by GPUs and the main storage is main memory.

### VARIANT
The variant is only for HYBRID_CPU_GPU. 
- O0: GPUs and CPUs process separate work units and no shared execution.
- O1: GPUs and CPUs process separate work units and with shared execution.
- O2: GPUs and CPUs cooperatively work together in a way where CPUs load data subgraphs for inter-partition processing. In this case, CPUs will not enumerate instances. Shared execution is enabled.
- O3: Similar to O2 but CPUs will enumerate instances in this case.


### LAZY_TRAVERSAL_COMPRESS_LEVEL

Refer to [src/query/common/LazyTraversalCommon.h](../query/common/LazyTraversalCommon.h).

### GPU_LIGHT_EXT_VARIANT
Specifically for GPU_LIGHT and EXTERNAL_GPU mode. 
- EXT_CACHE: Load data subgraphs from main memory on demand. Only load the subgraphs needed for the current enuemrating inter-partition instances.
- EXT_UMA: Use UMA to load data graphs.


## Example runs
To run the PBF (SIGMOD 2020), run GPU_LIGHT with HYBRID_CPU_GPU and O2
```
./patternmatchgpu -f ~/datasets/com-dblp.ungraph.txt.bin -d 0 -a 6 -q 2 ~/datasets/com-dblp.ungraph.txt.bin.weight.uniform.part.4 -p 4 -v 1 -m 0 -o 2
```

To run PBF as in-memory mode
```
./patternmatchgpu -f ~/datasets/com-dblp.ungraph.txt.bin -d 0 -a 6 -q 2 -p 1 -t 1 -v 1 -m 1

```

To run PBF with external mode 
```
./patternmatchgpu -f ~/datasets/com-dblp.ungraph.txt.bin -d 0 -a 6 -q 2 -p 1 -t 1 -v 1 -m 2
```

Please refer to [scripts/exp_subgenum.sh](../../scripts/exp_subgenum.sh) for more examples on how to run PBF.