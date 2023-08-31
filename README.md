# GLM-Pretrain-in-Megatron-DeepSpeed
GLM-Pretrain in Megatron-DeepSpeed for DCU based on [GLM-130B](https://github.com/THUDM/GLM-130B). 该工程去掉 lightop 后同样可以在 NV 平台运行。

## 模型介绍

GLM 模型采用了一种自回归的空白填充方法, 在 NLP 领域三种主要的任务（自然语言理解、无条件生成、有条件生成）上都取得了不错的结果。

## 数据集

根据用户环境指定数据集存放路径。

```
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d oscar-1GB.jsonl.xz
```

## GLM 预训练

要求 DCU 集群 Slurm ，MPI 环境正常，使用 [DTK22.10.1](https://developer.hpccube.com/tool/)。

推荐用户使用预编译好的 Python3.7 包来快速建立 Python3 虚拟环境，Pytorch、Apex、Torchvision、Deepspeed 需要在[光合开发者社区](https://cancon.hpccube.com:65024/4/main/)下载所需 DCU 版本安装包。可以参考以下流程：

```
export PYTHON3_LIB_PATH=/python_lib_path
virtualenv -p /python_bin_path/python3 --system-site-packages venv_torch3.7
source env.sh	# 进入 venv_glm 虚拟环境

pip3 install -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple
pip3 install icetk -i https://mirrors.aliyun.com/pypi/simple
```

处理数据参考：

```
bash creat-data.sh # 数据预处理
```

运行 GLM-11B 预训练。由于当前添加了 GQA，所以参数量会有减少。可以通过设置 NUM_KV_HEADS 等于 NHEADS 来使用 MHA。需要注意的是，在 Megatron 中只有 TP=1 时，NUM_KV_HEADS 才可以设置成1，等价于 MQA。 具体参数配置更改可以参考 dcu_single.sh。

```
sbatch dcu_run.sh
```

采用悟道数据集的GLM-130B配置参考wudao_dcu_single.sh，loss曲线如下：

![GLM-130B loss](https://github.com/yuguo-Jack/GLM-Pretrain-in-Megatron-DeepSpeed/blob/main/GLM-130B%20loss.png)

后续更新收敛情况。

## 参考

[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

[GLM-130B](https://github.com/THUDM/GLM-130B)
