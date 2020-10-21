# This script is a simple demo to show how to use GPU solutions

# change the location for your METIS
metis_dir=~/programs/metis-5.1.0/bin

wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
gzip -d com-dblp.ungraph.txt.gz 

../src/preprocess/preprocess_graph -f com-dblp.ungraph.txt -d 0
../src/preprocess/motif_cut -a 0 -f com-dblp.ungraph.txt.bin -d 0 -t 1 -z 2
partition_num=4
${metis_dir}/gpmetis ./com-dblp.ungraph.txt.bin.veweight.uniform ${partition_num}

query_type=2
#run cpu solution
../src/cpu/patternmatchcpu -f ./com-dblp.ungraph.txt.bin -d 0 -a 3 -q ${query_type} -t 10 -b 64 

#run gpu solution
../src/gpu/patternmatchgpu -f ./com-dblp.ungraph.txt.bin -d 0 -a 6 -q ${query_type} -e ./com-dblp.ungraph.txt.bin.veweight.uniform.part.${partition_num} -p ${partition_num} -t 1 -v 1 -m 0 -o 2 