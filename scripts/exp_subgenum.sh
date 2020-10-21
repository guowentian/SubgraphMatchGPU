# This script is used to conduct experiments for subgEnum paper.

# global configuration variables
data_dir=/home/wentian/datasets
bin_file=../src/gpu/patternmatchgpu
output_dir=logs

###### Running variables used for each run 
###### set these variables before each experiments 
# The running variables that are frequently modified
datasets=(com-youtube.ungraph.txt.bin wiki-Talk.txt.bin wiki-topcats.txt.bin as-skitter.txt.bin soc-twitter-higgs.txt.bin soc-dogster.txt.bin com-orkut.ungraph.txt.bin)
methods=(15)
queries=(2 3 6 7 9 10 11 12)
feature=gpu_coprocess
# The running variables that are seldom modified
directions=(0 0 0 0 0 0 0 0)
threads_num=1
dev_num=1
partition_num=1
mode=1
variant=0
gpu_light_itp_variant=1
gpu_light_ext_variant=0
compress_level=3
enable_ordering=1
# for timeout run
time_limit=24h
enable_time_limit=0 # No need to change it
# partition property
partition_property=veweight.uniform.part


run_once () {
  for ((di=0;di<${#datasets[@]};di++)); do
      dataset=${datasets[$di]}
      direction=${directions[$di]}

      for query in ${queries[@]}; do
          for method in ${methods[@]}; do
              output_file=${dataset}_a${method}_q${query}_${feature}.txt
              command="${bin_file} -f ${data_dir}/${dataset} -d ${direction} -a ${method} -q ${query} -t ${threads_num} -v ${dev_num} -e ${data_dir}/${dataset}.${partition_property}.${partition_num} -p ${partition_num} -m $mode -o $variant -x ${gpu_light_itp_variant} -c ${compress_level} -r ${enable_ordering} -y ${gpu_light_ext_variant} >> ${output_dir}/${output_file} 2>&1"
              if [ ${enable_time_limit} -ne 0 ]
              then
                command="timeout ${time_limit} $command"
              fi
              echo "$command"
              date > ${output_dir}/${output_file}
              eval $command
              date >> ${output_dir}/${output_file}
              sleep 1
          done
      done
  done
}

timeout_run_once () {
  enable_time_limit=1
  run_once
  enable_time_limit=0 # unset it after the run
}

###################### general run #######################

coprocess_run () {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  methods=(6)
  queries=(2 3 6 7)

  feature=gpu_coprocess
  threads_num=1
  dev_num=2
  partition_num=4
  mode=0
  variant=2

  # (optional) For comparison, turn off the effects of fast counting techniques 
  #compress_level=0
  #compress_level=2
  # (optional) For comparison, turn off the ordering pruning
  #enable_ordering=0

  # (optional) Try out different partition plans 
  partition_property=veweight.uniform.part
  #partition_property=adj.txt.fennel.part
  #partition_property=weight.uniform.part

  run_once 
}


external_run () {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  methods=(6)
  queries=(2 3 6 7)

  feature=gpu_external
  threads_num=1
  dev_num=1
  partition_num=1
  mode=2
  variant=0
  gpu_light_itp_variant=1
  gpu_light_ext_variant=0
  
  run_once
}

inmemory_run () {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  methods=(6)
  queries=(2 3 6 7)

  feature=gpu_inmemory
  threads_num=1
  dev_num=1
  partition_num=1
  mode=1
  variant=0

  run_once
}

###################### comparison run #######################

light_uma_run () {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  methods=(6)
  queries=(2 3 6 7)

  threads_num=1
  dev_num=2
  partition_num=1 # for uma
  mode=2
  variant=0

  feature=gpu_external_uma
  gpu_light_ext_variant=1
  partition_property=veweight.uniform.part

  run_once
}

coprocess_compare_light_itp_variant () {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  methods=(6)
  queries=(2 3 6 7)

  threads_num=1
  dev_num=1
  partition_num=4
  mode=0
  variant=2

  # for comparison, turn off the effects of fast counting techniques and ordering
  compress_level=0
  enable_ordering=0

  # Should disable the intra-partition search and compile in order to proceed for this comparison
  feature=gpu_coprocess_inter_partition_ss
  gpu_light_itp_variant=0
  run_once 

  feature=gpu_coprocess_inter_partition_group
  gpu_light_itp_variant=1
  run_once 
}

coprocess_compare_vary_partition_num() {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  methods=(6)
  queries=(2 3 6 7)

  threads_num=1
  dev_num=1
  mode=0
  variant=2
  
  partition_nums=(1 4 8 12 16)
  
  for cur_partition_num in ${partition_nums[@]}; do
    partition_num=${cur_partition_num}
    feature=gpu_coprocess_partition_${partition_num}
    if [ ${partition_num} = 1 ] 
    then
      mode=1
    else 
      mode=0
    fi
    run_once
  done
}

multi_gpu() {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  methods=(6)
  queries=(2 3 6 7)

  threads_num=8
  partition_num=4
  mode=0
  variant=2

  dev_nums=(2 1)
  echo $dev_nums
  for dn in ${dev_nums[@]}; do
    dev_num=$dn
    feature=gpu_coprocess_${dn}
    run_once
  done
}

vary_graph_size () {
  methods=(6)
  queries=(2 3 6 7)

  feature=gpu_coprocess
  threads_num=1
  dev_num=2
  compress_level=2
  
  # 20% vertices
  datasets=(yahoog2_20.txt.bin)
  mode=1
  variant=0
  partition_num=1
  run_once

  # 40% vertices
  datasets=(yahoog2_40.txt.bin)
  mode=0
  variant=2
  partition_num=4
  partition_property=veweight.uniform.part
  run_once

  # 60% vertices
  datasets=(yahoog2_60.txt.bin)
  mode=0
  variant=2
  partition_num=8
  partition_property=adj.txt.fennel.part
  run_once

  # 80% vertices
  datasets=(yahoog2_80.txt.bin)
  mode=0
  variant=2
  partition_num=12
  partition_property=adj.txt.fennel.part
  run_once
}

# Run one function at a time as variables would be overriden between functions
echo "start a new test"
coprocess_run
#external_run
#inmemory_run

#light_uma_run
#coprocess_compare_light_itp_variant
#coprocess_compare_vary_partition_num
#multi_gpu
#vary_graph_size 
echo "done!"
