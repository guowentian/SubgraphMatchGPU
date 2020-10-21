# variables that are seldomly changed
data_dir=~/datasets
# number of threads used for motif cut
thread_num=20

# frequently changed variables
#data_sets=(graph500_28_4.txt.bin graph500_29_4.txt.bin)
data_sets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
partitions=(4)
 
preprocess_adj_graph () {
    binfile=../src/preprocess/preprocess_graph
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

    for orig_data_set in ${data_sets[@]}; do
        for partition in ${partitions[@]}; do
            data_set=${orig_data_set}.adj.txt
            info_file=${input_dir}/${data_set}.info

            vcount=`awk '{print $1}' ${info_file}`
            ecount=`awk '{print $2}' ${info_file}`
            echo "$bin --in ${input_dir}/$data_set --out ${output_dir}/${data_set}.${property}.${partition} --part_num $partition --vertex_num $vcount --edge_num $ecount"
            $bin --in ${input_dir}/$data_set --out ${output_dir}/${data_set}.${property}.${partition} --part_num $partition --vertex_num $vcount --edge_num $ecount
        done
    done
}
