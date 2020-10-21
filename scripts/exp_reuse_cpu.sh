# this script is to test the performance of reuse bfs on CPUs

# global configuration variables
data_dir=/home/wentian/datasets
bin_file=../src/cpu/patternmatchcpu
output_dir=reuse

###### Running variables used for each run 
###### set these variables before each experiments 
# The running variables that are frequently modified
datasets=(com-youtube.ungraph.txt.bin wiki-Talk.txt.bin wiki-topcats.txt.bin as-skitter.txt.bin soc-twitter-higgs.txt.bin soc-dogster.txt.bin com-orkut.ungraph.txt.bin)
methods=(15 16 17)
queries=(2 3 6 7 9 10 11 12)
feature=cpu
thread_num=32
# The running variables that are seldom modified
directions=(0 0 0 0 0 0 0 0)
memory=64
# for timeout run
time_limit=1s

run_once () {
  for ((di=0;di<${#datasets[@]};di++)); do
    dataset=${datasets[$di]}
    direction=${directions[$di]}

    for query in ${queries[@]}; do
      for method in ${methods[@]}; do 
        outfile=$output_dir/${dataset}_a${method}_q${query}_${feature}.txt
        echo "$bin_file -f ${data_dir}/$dataset -d $direction -a $method -q $query -t $thread_num -b $memory >> $outfile"
        date > $outfile 
        $bin_file -f ${data_dir}/$dataset -d $direction -a $method -q $query -t $thread_num -b $memory >> $outfile
        date >> $outfile
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
        outfile=$output_dir/${dataset}_a${method}_q${query}_${feature}.txt
        echo "timeout ${time_limit} $bin_file -f ${data_dir}/$dataset -d $direction -a $method -q $query -t $thread_num -b $memory >> $outfile"
        date > $outfile 
        timeout ${time_limit} $bin_file -f ${data_dir}/$dataset -d $direction -a $method -q $query -t $thread_num -b $memory >> $outfile
        date >> $outfile
      done
    done
  done
}


effect_of_reuse () {
  #datasets=("com-youtube.ungraph.txt.bin" "wiki-Talk.txt.bin" "as-skitter.txt.bin" "soc-twitter-higgs.txt.bin" "soc-dogster.txt.bin" "com-orkut.ungraph.txt.bin")
  datasets=("com-dblp.ungraph.txt.bin")
  queries=(2 3 6 7 9 10 11 12)
  methods=(3 2)
  feature=cpu
  thread_num=32

  run_once
}


profile_set_intersect_ratio () {
  datasets=(com-dblp.ungraph.txt.bin com-youtube.ungraph.txt.bin)
  queries=(2 3 6 7 9 10 11 12)
  methods=(3)
  feature=cpu_profile
  thread_num=32

  run_once 
}

effect_of_reuse
#profile_set_intersect_ratio
