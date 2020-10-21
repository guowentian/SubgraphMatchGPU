# Parallel subgraph enumeration on Multi-core CPUs

## Compile 
Simply "make -j" in this folder.

## Prepare
The program accepts two formats of data graphs, i.e., SNAP and BIN.
For more descriptions on these formats, please refer to the [main folder](../../README.md).


## Run
We can run the program without any parameter to output a helping message, which can help you to input the parameters.

There are some parameters you need to specify to run the program.

- -f, the file name of the data graph, whose format should be either SNAP or BIN.
- -d, the direction of the graph. 0 for undirected graph and 1 for directed graph. For subgraph enumeration, wet set it to 0. If 1 is specified, we do not guarantee the correctness of the result.
- -a, the algorithm to run. The available algorithms are listed in the helping message.
- -q, the query to run. The supported queries are listed in the helping message. You can other queries, but take some development efforts.
- -t, the number of threads used.
- -b, the memory buffer limit (in GB). When we run BFS-style of algorithms, we need to ensure the generated partial instances would not exceed the limit of the memory buffer. For DFS-style of algorithms, this parameter is not needed. 


### Example
- Run the CPU_WCOJ (worst-case optimal join) algorithm, which is also the same as Ullman's approach, on the triangle query using 32 threads
```
./patternmatchcpu -f ~/datasets/com-dblp.ungraph.txt.bin -d 0 -a 0 -q 0 -t 32 
```
- Run the CPU_BFS (PGX) algorithm on the chordal square query using 32 threads with a memory buffer limit as 64 GB.
```
./patternmatchcpu -f ~/datasets/com-dblp.ungraph.txt.bin -d 0 -a 2 -q 2 -t 32 -b 64
```
- Run reusable BFS that is based on CPU_BFS on the similar setting
```
./patternmatchcpu -f ~/datasets/com-dblp.ungraph.txt.bin -d 0 -a 3 -q 2 -t 32 -b 64
```


## Script 
[scripts/exp_reuse_cpu.sh](../../scripts/exp_reuse_cpu.sh) shows some example runs.