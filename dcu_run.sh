#! /bin/bash

#SBATCH -p wzhdnormal
#SBATCH -N 32
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH -J glm
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.out
#SBATCH -x a04r3n01,g01r1n03,g01r1n05,g01r1n08

ulimit -u 200000
source /work/home/yuguo960516yuguo/llm/env.sh
rm -rf ./hostfile/*
echo "START TIME: $(date)"
hostfile=./hostfile/$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)
np=$(($np*4))
#module list
echo $np
nodename=$(cat $hostfile |sed -n "1p")
dist_url=`echo $nodename | awk '{print $1}'`
# mpirun -np $np --allow-run-as-root --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/dcu_single.sh $dist_url
mpirun -np $np -mca pml ob1 -mca btl self,vader,tcp --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/dcu_single.sh $dist_url
