This folder provides utilities to convert a data graph into a compacted format and generate a partition plan.

# Compile
```
make clean; make -j
```

# Convert to BIN format
Run `preprocess_graph` without arguments would show the helper message.
```
./preprocess_graph -f GRAPH_FILENAME(SNAP format) -d DIRECTED -o OUTPUT_FORMAT
OUTPUT_FORMAT: 0  for .snap, 1 (Default) for .bin, 3 for .adj, 
```
- GRAPH_FILENAME: the path of data file in SNAP format as the input
- DIRECTED: 0 for undirected graph and 1 for directed graph
- OUTPUT_FORMAT: *bin* is the binary file (recommended). *adj* means the CSR adjacent list, where one adjacent list is written per line. *adj* is the input format used for *FENNEL*.

For example, to transform the SNAP format to BIN format.
```
./preprocess_graph -f ~/data/com-dblp.ungraph.txt -d 0
```
A BIN file `~/data/com-dblp.graph.txt.bin` would be generated. 


# Generate partition plan
We rely on some existing tools for graph partitioning, e.g., METIS and FENNEL. To do that, need to provide the data formats that can be the inputs for these tools.

For METIS, we need to specify the edge weights as well as the graph structures. There are different ways to define/compute the edge weights. The simplist one is to set each edge weight as 1; other methods can be setting the number of motifs crossing this edge. The motif cut counting process provide different ways to compute the edge weights and generate the output file for METIS.

## Motif cut counting

Simply run `motif_cut` without arguments would show the helper message.
```
motifcut -a finder_type -f filename -d directed -p partition_type -t thread_num -o output_filename -z output_type
PARTITION_TYPE: Q0 (TRIANGLE) 0, Q1 (square) 1, Q2 (chordal square) 2, Q3 (4 clique) 3, Q4 (house) 4, Q5 (quad triangle) 5, Q6 (near5clique) 6, Q7 (5 clique) 7, Q8 (chordal roof) 8, Q9 (three triangle) 9, Q10 (solar square) 10, Q11 (6 clique) 11,Q15 (LINE, Default)
FINDER_TYPE: 0 for exact (Default), 1 for sample
OUTPUT_TYPE: 0 for MOTIF_CUT, 1 for METIS_WEIGHT, 2 for METIS_VE_WEIGHT
```
Some important options
- filename: The path of input file for data graph
- directed: 0 for undirected graph and 1 for directed graph
- partition_type: The motif types chosen to assign the edge weights. By default, LINE means setting 1 (equal weight) for each edge.

It supports the following output formats:
- METIS_WEIGHT: Each edge weight in this format means the number of motifs crossing that edge.
- METIS_VE_WEIGHT: Similar to METIS_WEIGHT, plus setting the vertex weight as the number of neighbors it has. 
- MOTIF_CUT: This format is only for experiment purpose and not for METIS. The first line is total number of motifs. Each subsequent line has the form 'u v w'. It means that there is an edge from u to v with the weight as w. Note that for undirected graph, when there is u-w->v, then there is also v-w->u.

For example, to set 1 for each edge weight and assign number of neighbors as the vertex weight, 
```
./motif_cut -a 0 -f ~/data/com-dblp.ungraph.txt.bin -d 0 -t 1 -z 2
```
A file `~/data/com-dblp.ungraph.txt.bin.veweight.uniform` would be generated.




## Use partitioner
You need to install METIS or MT-METIS to use partitioner.
Check out [the website](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) to download, install, and use.


# Script
[scripts/preprocess.sh](../../scripts/preprocess.sh) automate the above process. Check it for more examples.

