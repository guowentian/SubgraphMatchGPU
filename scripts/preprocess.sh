# variables that are seldomly changed
data_dir=~/datasets
metis_bin_dir=~/programs/metis-5.1.0/bin
mtmetis_bin_dir=~/programs/mt-metis-0.6.0/bin
# number of threads used for motif cut
thread_num=4

# frequently changed variables
# for each data set, specify the corresponding partition number
data_sets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
partitions=(4 4)
 
# Generate our home-made graph format.
# Meanwhile, we eliminate loops, compact vertex ids.
preprocess_bin_graph () {
    binfile=../src/preprocess/preprocess_graph
    prev_data_sets=${data_sets}
    #### specify the original graph data sets here
    data_sets=(com-dblp.ungraph.txt com-youtube.ungraph.txt)

    for ((idx=0;idx<${#data_sets[@]};idx++));do
        data_set=${data_sets[$idx]}
        direction=0
        echo "$binfile -f ${data_dir}/${data_set} -d $direction"
        $binfile -f ${data_dir}/${data_set} -d $direction
        sleep 1
    done
    echo "done"
    data_sets=${prev_data_sets}
}

# METIS partition requires an input file format with adjacent lists and edge weights. 
# This preprocessing step prepares the input file for METIS
assign_edge_weight () {
    # 2 for a file type that has both vertex and edge weights for metis
    motifcut_output_type=2
    binfile=../src/preprocess/motif_cut
    for ((idx=0;idx<${#data_sets[@]};idx++));do
        data_set=${data_sets[$idx]}
        direction=0

        echo "$binfile -a 0 -f ${data_dir}/${data_set} -d $direction -t ${thread_num} -z ${motifcut_output_type}"
        $binfile -a 0 -f ${data_dir}/${data_set} -d $direction -t ${thread_num} -z ${motifcut_output_type}
        sleep 1
    done
    echo "done motif cut!"
}

# sequential metis
metis_partition () {
    feature=veweight.uniform
    binmetis=${metis_bin_dir}/gpmetis
    for ((idx=0;idx<${#data_sets[@]};idx++));do
        data_set=${data_sets[$idx]}
        partition=${partitions[$idx]}
        input_file=${data_dir}/${data_set}.${feature}
        echo "$binmetis ${input_file} $partition"
        $binmetis ${input_file} $partition
    done
}

# parallel metis
parallel_metis_partition () {
    feature=veweight.uniform
    binmetis=${mtmetis_bin_dir}/mtmetis
    for ((idx=0;idx<${#data_sets[@]};idx++));do
        data_set=${data_sets[$idx]}
        partition=${partitions[$idx]}
        input_file=${data_dir}/${data_set}.${feature}
        output_file=${data_dir}/${data_set}.${feature}.part.${partition}

        # mtmetis recognizes the file type by the extension, so we have to set to .graph
        ln -s ${input_file} ${input_file}.graph
        echo "$binmetis ${input_file}.graph $partition ${output_file} -t -T${thread_num}"
        $binmetis ${input_file}.graph $partition ${output_file} -t -T${thread_num}
        rm ${input_file}.graph
    done
}

# Generates the input file that can be recognized by fennel
preprocess_adj_graph () {
    binfile=../src/preprocess/preprocess_graph
    # 3 for ADJ file type
    output_type=3

    for ((idx=0;idx<${#data_sets[@]};idx++));do
        data_set=${data_sets[$idx]}
        direction=0
        echo "$binfile -f ${data_dir}/${data_set} -d $direction -o $output_type"
        $binfile -f ${data_dir}/${data_set} -d $direction -o $output_type
        sleep 1
    done
    echo "done"
}


fennel_partition () {
    bin=../lib/fennel/fennel
    input_dir=$data_dir
    output_dir=$data_dir
    property=fennel.part

    for ((idx=0;idx<${#data_sets[@]};idx++));do
      orig_data_set=${data_sets[$idx]}
      partition=${partitions[$idx]}
      data_set=${orig_data_set}.adj.txt
      info_file=${input_dir}/${data_set}.info

      vcount=`awk '{print $1}' ${info_file}`
      ecount=`awk '{print $2}' ${info_file}`
      echo "$bin --in ${input_dir}/$data_set --out ${output_dir}/${data_set}.${property}.${partition} --part_num $partition --vertex_num $vcount --edge_num $ecount > ${output_dir}/fennel_partition_log.txt"
      $bin --in ${input_dir}/$data_set --out ${output_dir}/${data_set}.${property}.${partition} --part_num $partition --vertex_num $vcount --edge_num $ecount > ${output_dir}/fennel_partition_log.txt
    done
}


# normal graphs
preprocess_bin_graph
assign_edge_weight
#metis_partition
parallel_metis_partition


# for large graphs that metis cannot handle
#preprocess_adj_graph 
#fennel_partition
