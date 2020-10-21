# This script is used to test and experiment the performance of reuse bfs on GPUs

# global configuration variables
data_dir=/home/wentian/datasets
bin_file=../src/gpu/patternmatchgpu
output_dir=reuse

###### Running variables used for each run 
###### set these variables before each experiments 
# The running variables that are frequently modified
datasets=(com-youtube.ungraph.txt.bin wiki-Talk.txt.bin wiki-topcats.txt.bin as-skitter.txt.bin soc-twitter-higgs.txt.bin soc-dogster.txt.bin com-orkut.ungraph.txt.bin)
methods=(15 16 17)
queries=(2 3 6 7 9 10 11 12)
feature=gpu_cnmem2_profile
# The running variables that are seldom modified
directions=(0 0 0 0 0 0 0 0)
threads_num=1
dev_num=1
partition_num=1
mode=1
variant=0
# for timeout run
time_limit=1s

# the basic format of output file is 
# dataset_method_query_feature
# 'feature' can be changed to identify different purposes of the experiments

## Run the program according to the specified running variables.
## This is a frequently used routine after configuring the variables.
run_once () {
  for ((di=0;di<${#datasets[@]};di++)); do
      dataset=${datasets[$di]}
      direction=${directions[$di]}

      for query in ${queries[@]}; do
          for method in ${methods[@]}; do
              output_file=${dataset}_a${method}_q${query}_${feature}.txt
              echo "${bin_file} -f ${data_dir}/${dataset} -d ${direction} -a ${method} -q ${query} -t ${threads_num} -v ${dev_num} -p ${partition_num} -m $mode -o $variant > ${output_dir}/${output_file} 2>&1"
              date > ${output_dir}/${output_file}
              ${bin_file} -f ${data_dir}/${dataset} -d ${direction} -a ${method} -q ${query} -t ${threads_num} -v ${dev_num} -p ${partition_num} -m $mode -o $variant >> ${output_dir}/${output_file} 2>&1
              date >> ${output_dir}/${output_file}
              sleep 1
          done
      done
  done
}

timeout_run_once () {
  for ((di=0;di<${#datasets[@]};di++)); do
      dataset=${datasets[$di]}
      direction=${directions[$di]}

      for query in ${queries[@]}; do
          for method in ${methods[@]}; do
              output_file=${dataset}_a${method}_q${query}_${feature}.txt
              echo "timeout ${time_limit} ${bin_file} -f ${data_dir}/${dataset} -d ${direction} -a ${method} -q ${query} -t ${threads_num} -v ${dev_num} -p ${partition_num} -m $mode -o $variant > ${output_dir}/${output_file} 2>&1"
              date > ${output_dir}/${output_file}
              timeout ${time_limit} ${bin_file} -f ${data_dir}/${dataset} -d ${direction} -a ${method} -q ${query} -t ${threads_num} -v ${dev_num} -p ${partition_num} -m $mode -o $variant >> ${output_dir}/${output_file} 2>&1
              date >> ${output_dir}/${output_file}
              sleep 1
          done
      done
  done
}


effect_of_reuse () {
  datasets=(com-youtube.ungraph.txt.bin wiki-Talk.txt.bin wiki-topcats.txt.bin soc-twitter-higgs.txt.bin soc-dogster.txt.bin com-orkut.ungraph.txt.bin)
  methods=(4 5)
  queries=(2 3 6 7 9 10 11 12)
  feature=gpu_cnmem2_profile
  run_once
}

reuse_profile_run () {
  datasets=(com-youtube.ungraph.txt.bin wiki-Talk.txt.bin as-skitter.txt.bin soc-twitter-higgs.txt.bin soc-dogster.txt.bin com-orkut.ungraph.txt.bin)
  methods=(4)
  queries=(2 3 6 7 9 10 11 12)
  
  feature=gpu_cnmem2_reuse_profile
  run_once
}

perf_run () {
  datasets=(wiki-topcats.txt.bin  soc-twitter-higgs.txt.bin soc-dogster.txt.bin com-orkut.ungraph.txt.bin)
  methods=(5)
  queries=(2 3 6 7 9 10 11 12)

  feature=gpu_cnmem2_profile
  run_once
}

timeout_perf_run () {
  datasets=(wiki-topcats.txt.bin soc-twitter-higgs.txt.bin soc-dogster.txt.bin com-orkut.ungraph.txt.bin)
  methods=(5)
  queries=(2 3 6 7 9 10 11 12)
  time_limit=3h

  feature=gpu_cnmem2_profile
  timeout_run_once
}


echo "start a new perf test"
#perf_run
#reuse_profile_run
#effect_of_reuse
timeout_perf_run
