# Muilti-GPU-Fine-Tuning:functin call
Muilti-GPU-Fine-Tuning by X-tunter
# 概述
本项目通过x-tuner微调LLama3-8B-Instruct模型，使用 XTuner 在 Agent-Flan 数据集上微调 Llama3-8B-Instruct，  
以让 Llama3-8B-Instruct 模型获得智能体能力，调用function tools能力提升。
## 目录
- [1. 微调 Llama3-8B-Instruct 模型](#1-微调-llama3-8b-instruct-模型)
- [2. 接入 FastGPT](#2-接入-fastgpt)
## 1. 微调 Llama3-8B-Instruct 模型
### 1.1 创建notebook
  - 在超算互联网汇视威AI训练平台，模型调试中创建一个notebook
  - 使用的算法为：function-call-04；
  - 使用的镜像为：function-call:v0.0.1；
  - 使用的数据为：huggingface-cache
### 1.2 设置环境变量
【在平台上】每次打开新的终端都需要执行如下操作
  ```python
  # 为了连接网络
  export http_proxy=http://10.10.9.50:3000
  export https_proxy=http://10.10.9.50:3000
  export no_proxy=localhost,127.0.0.1
# 配置本地下载huggingface文件的cache目录
  export HF_HOME=/code/huggingface-cache/
# 配置huggingface连接的镜像
  export HF_ENDPOINT=https://hf-mirror.com
  ```
### 1.3 下载模型
  由于huggingface上直接下载Meta的Llama-3-8B-Instruct需要填写申请，我们换个源下载模型
  ```python
  apt-get install git-lfs                 # git下载大文件需要安装
  git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
  ```
  平台上已经下载好了模型，在/dataset/Llama-3-8B-Instruct/路径下。需要在创建任务或者notebook时挂载huggingface-cache数据集
### 1.4 数据准备
  下载数据命令如下：
  ```python
huggingface-cli download internlm/Agent-FLAN --repo-type dataset --revision main --local-dir-use-symlinks False --local-dir /code/internlm_Agent-FLAN_data
  ```
数据来源：https://hf-mirror.com/datasets/internlm/Agent-FLAN/viewer/default/agent_instruct_react?row=0  
数据集介绍：https://developer.volcengine.com/articles/7389112107738644506  
由于 HuggingFace 上的 Agent-Flan 数据集暂时无法被 XTuner 直接加载，因此我们首先要下载到本地，然后转换成 XTuner 直接可用的格式。数据转换命令如下：
```python
python /code/convert_agentflan.py 
/dataset/datasets/internlm_Agent-FLAN/data/ 
/code/data_converted
```
**平台上转换后的数据位于：/code/data_converted**  
读取指定目录下的所有.jsonl文件，处理文件中的JSON数据，然后将处理后的数据保存为一个新的数据集文件。  
处理包括将'conversation'键重命名为'messages'，并移除'id'键。最终，数据被保存在指定的保存路径下，并附加_converted后缀。
```python
import json
import os
import sys
from datasets import Dataset


file_path = sys.argv[1]  # /xxx/internlm/Agent-Flan/data
save_path = sys.argv[2]  # /code/data
if file_path.endswith('/'):
    file_path = file_path[:-1]

ds = []
for file in os.listdir(file_path):
    if not file.endswith('.jsonl'):
        continue
    with open(os.path.join(file_path, file)) as f:
        dataset = f.readlines()
        for item in dataset:
            conv = json.loads(item)
            conv['messages'] = conv.pop('conversation')
            if 'id' in conv:
                conv.pop('id')
            ds.append(conv)

ds = Dataset.from_list(ds)
ds.save_to_disk(f'{save_path}_converted')
```
### 1.5 使用xtuner进行微调训练
https://github.com/InternLM/xtuner
XTuner是一个专为大语言模型（LLM）和多模态模型微调设计的高效、灵活、全能的轻量化工具库。  
1. 支持多种大语言模型：XTuner支持包括但不限于InternLM、Mixtral-8x7B、Llama2、ChatGLM、Qwen、Baichuan等多种大语言模型。  
2. 高效性能：XTuner能够在有限的硬件资源下微调大型模型，例如在8GB显存环境下微调7B参数模型。  
它集成了高性能算子（如FlashAttention、Triton kernels等）以加速训练过程，并兼容DeepSpeed框架，可应用各种ZeRO策略优化训练效率。  
3. 灵活适配：XTuner支持多种微调算法，包括QLoRA、LoRA、全量参数微调等多种微调算法，用户可以根据具体需求作出最优选择。
4. 全能功能：XTuner支持增量预训练、指令微调与Agent微调等多种训练方式。预定义众多开源对话模版，支持与开源或训练所得模型进行对话。  
训练所得模型可无缝接入部署工具库LMDeploy、大规模评测工具库OpenCompass及VLMEvalKit。
5. 易用性：XTuner以配置文件的形式封装了大部分微调场景，0基础的非专业人员也能一键开始微调，使得微调过程傻瓜化。  
6. 快速上手：XTuner提供了快速上手指南，包括安装、微调和对话等步骤，使得用户可以轻松开始使用XTuner进行模型微调。  
这里使用4块DCU进行分布式训练，上下文长度调整为1664，训练命令如下（单机训练时长约14~15天）：
```python
NPROC_PER_NODE=4 xtuner train \
/code/llama3_8b_instruct_qlora_agentflan_3e.py \
--work-dir /userhome/llama3-8b-ft/agent-flan \
--deepspeed deepspeed_zero3_offload
```
### 1.5.1 设置训练变量：
```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
# 指定预训练模型的路径
pretrained_model_name_or_path = '/dataset/Llama-3-8B-Instruct/'
# 是否使用可变长度的注意力机制
use_varlen_attn = False
# 是否使用激活检查点技术来减少GPU内存使用
use_activation_checkpointing=True

# Data
# 数据转换后的存储路径
agent_flan_path = '/code/data_converted'
# 提示模板
prompt_template = PROMPT_TEMPLATE.llama3_chat
# 输入的最大长度
# max_length = 2048     #  4096长度太大, 这里改成2048
# 可能会out of memory，修改为1536
max_length = 1664     #  4096长度太大, 这里改成1536, 1664
# 是否将数据打包到最大长度
pack_to_max_length = False
# 数据集中的最大长度
# max_dataset_length=2000 # max=77427

# parallel
# 序列并行的大小
sequence_parallel_size = 1
```
```python
# Scheduler & Optimizer
# 每个设备的批次大小
batch_size = 1  # per_device
# 累积计数，用于控制梯度累积，这里设置为16，并乘以序列并行大小。
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
# 数据加载器的工作线程数，这里设置为0。
dataloader_num_workers = 0
# 最大训练周期数，这里设置为3。
max_epochs = 3
# 优化器类型，这里设置为AdamW。
optim_type = AdamW
# 学习率
lr = 2e-4
# Adam优化器的beta参数
betas = (0.9, 0.999)
# 权重衰减
weight_decay = 0
# 梯度裁剪的最大范数
max_norm = 1  # grad clip
# 预热比例
warmup_ratio = 0.03
```
```python
# Save
# 保存模型的步数
save_steps = 500
# 最大保存的检查点数量
# save_steps = 100
save_total_limit = 50  # Maximum checkpoints to keep (-1 means unlimited)
# save_total_limit = -1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
# 评估频率
evaluation_freq = 500
# 系统描述，包括如何使用外部工具和格式化回复的说明。
SYSTEM = (
    'You are a assistant who can utilize external tools.\n'
    "[{{\'name\': \'ArxivSearch.get_arxiv_article_information\',"
    "\'description\': \'Run Arxiv search and get the article meta information.\',"
    "\'parameters\': [{{\'name': \'query', \'type\': \'STRING\', \'description\':"
    "\'the content of search query\'}}], \'required\': [\'query\'], \'return_data\':"
    "[{{\'name\': \'content\', \'description\': \'a list of 3 arxiv search papers\', \'type\': \'STRING\'}}],"
    "\'parameter_description\': \'If you call this tool, you must pass arguments in the JSON format"
    "{{key: value}}, where the key is the parameter name.\'}}]\n"
    'To use a tool, please use the following format:\n```\n'
    'Thought:Think what you need to solve, do you need to use tools?\n'
    "Action:the tool name, should be one of [[\'ArxivSearch\']]\n"
    'Action Input:the input to the action\n```\n'
    'The response after utilizing tools should using the following format:\n```\n'
    'Response:the results after call the tool.\n```\n'
    'If you already know the answer, or you do not need to use tools,\n'
    'please using the following format to reply:\n```\n'
    'Thought:the thought process to get the final answer\n'
    'Final Answer:final answer\n```\nBegin!'
)

# 评估输入，这里是一个列表，包含一个请求帮助搜索InternLM2技术报告的输入。
evaluation_inputs = [
    'Please help me search the InternLM2 Technical Report.'
]
```
1.5.2模型设置
```python
#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
# tokenizer: 配置分词器（Tokenizer）。
# AutoTokenizer.from_pretrained: 使用AutoTokenizer类从预训练模型自动加载分词器。from_pretrained方法会自动下载并加载与预训练模型匹配的分词器。
# pretrained_model_name_or_path: 预训练模型的名称或路径。
# trust_remote_code=True: 允许加载远程代码，这是为了确保能够正确加载分词器。
# padding_side='right': 指定填充（Padding）的方向为右侧。这意味着在处理序列时，如果需要填充以达到固定长度，填充将被添加到序列的右侧。
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')
```
```python
# model: 用于配置模型。
# SupervisedFinetune: 指定模型类型为监督式微调。
# use_varlen_attn: 是否使用可变长度的注意力机制。
# use_activation_checkpointing: 是否使用激活检查点技术。
model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    use_activation_checkpointing=use_activation_checkpointing,
    
    # LLM（Large Language Model）配置
    # 使用AutoModelForCausalLM类从预训练模型自动加载因果语言模型。
    # pretrained_model_name_or_path: 预训练模型的名称或路径。
    # trust_remote_code=True: 允许加载远程代码。
    # torch_dtype=torch.float16: 指定模型使用的数据类型为半精度浮点数（float16），这有助于减少内存使用和加速计算。
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16),
    # LoRA（Low-Rank Adaptation）配置
    # type=LoraConfig: 指定使用LoRA配置。
    # r=64: LoRA的rank参数，控制参数矩阵的秩。
    # lora_alpha=16: LoRA的alpha参数，控制参数矩阵的缩放。
    # lora_dropout=0.1: LoRA的dropout率，用于正则化以防止过拟合。
    # bias='none': 指定LoRA不使用偏置项。
    # task_type='CAUSAL_LM': 指定任务类型为因果语言模型。
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))
```
### 1.5.3 数据集设置
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
# 配置处理特定数据集。
agent_flan = dict(
    type=process_hf_dataset, # 指定处理数据集的函数或方法
    dataset=dict(type=load_from_disk, dataset_path=agent_flan_path), # 指定从磁盘加载数据集的方式和路径。
    tokenizer=tokenizer, # 分词器配置
    max_length=max_length, # 设置处理后数据的最大长度
    dataset_map_fn=openai_map_fn, # 指定用于数据集的映射函数
    template_map_fn=dict( # 指定用于模板的映射函数工厂和模板。
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True, # 移除数据集中未使用的列
    shuffle_before_pack=True, # 在打包前对数据进行随机打乱
    pack_to_max_length=pack_to_max_length, # 是否将数据打包到最大长度
    use_varlen_attn=use_varlen_attn) # 是否使用可变长度的注意力机制
    # max_dataset_length=max_dataset_length) #

# 根据sequence_parallel_size的值选择采样器类型。
# 如果sequence_parallel_size大于1，则使用序列并行采样器。
# 如果sequence_parallel_size不大于1，则使用默认采样器。
sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler

# 配置训练数据加载器。
train_dataloader = dict(
    batch_size=batch_size, # 设置每个批次的大小
    num_workers=dataloader_num_workers, # 设置数据加载器的工作线程数
    dataset=agent_flan, # 使用之前配置的agent_flan数据集
    sampler=dict(type=sampler, shuffle=True), # 使用之前选择的采样器，并设置为随机打乱
    # 使用默认的批处理函数，并设置是否使用可变长度的注意力机制。
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn)) 
### 1.5.4 调度器和优化器
```python
#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer 配置优化器。
optim_wrapper = dict(
    # 指定使用AmpOptimWrapper，这是一个用于混合精度训练的优化器包装器。
    type=AmpOptimWrapper,
    # 配置具体的优化器
    # type=optim_type: 优化器的类型
    # lr=lr: 学习率
    # betas=betas: Adam优化器的beta参数
    # weight_decay=weight_decay: 权重衰减
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    # 配置梯度裁剪，max_norm是梯度的最大范数，error_if_nonfinite表示如果梯度不是有限值则抛出错误。
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts, # 累积计数，用于梯度累积
    loss_scale='dynamic', # 损失尺度设置为动态
    dtype='float16') # 数据类型设置为半精度浮点数（float16）

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
# param_scheduler: 用于配置多个学习率调度器。
# 第一个调度器是LinearLR，用于学习率预热（warm-up），
# 在训练的前warmup_ratio * max_epochs周期内，学习率从start_factor线性增加到正常值。
# 第二个调度器是CosineAnnealingLR，
# 用于在预热后使用余弦退火策略调整学习率，直到训练结束。
# by_epoch=True: 表示学习率调整是基于周期（epoch）的。
# begin和end参数定义了每个调度器的有效区间。
# convert_to_iter_based=True: 表示将基于周期的学习率调度器转换为基于迭代次数的调度器。
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
# 用于配置训练。
# type=TrainLoop: 指定使用的训练循环类型。
# max_epochs=max_epochs: 设置最大训练周期数
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
```
### 1.5.5运行时hook函数设置
在深度学习训练中，hook 是在训练过程中的特定点自动调用的函数或类，用于执行如日志记录、评估、调整学习率等辅助任务。
```python
#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # 用于在训练开始前或周期性地打印或记录数据集的信息。
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    # 在训练过程中定期评估聊天模型的性能。
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template),
    # 用于监控和记录训练过程中的吞吐量，即模型处理数据的速度。
    dict(type=ThroughputHook)
]
# 用于处理可变长度注意力（variable length attention）的参数，
# 并将它们传递给消息中心（MessageHub）。
if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    # 配置一个计时器钩子，用于记录每次迭代的时间。
    timer=dict(type=IterTimerHook),
    # 配置一个日志记录钩子，用于每10次迭代打印一次日志。
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    # 配置一个参数调度钩子，用于启用学习率调度器等。
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    # 配置一个检查点钩子，用于保存模型的检查点。
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    # 配置一个分布式环境中的采样器种子钩子，
    # 用于确保在分布式训练中采样器的种子是一致的。
    sampler_seed=dict(type=DistSamplerSeedHook),
)
```
### 1.5.6 其他配置
```python
# configure environment
# 环境配置
env_cfg = dict(
    # whether to enable cudnn benchmark
    # 设置是否启用 NVIDIA 的 cuDNN 基准测试。
    # 这个选项可以加速训练，但需要在固定的模型和数据集上进行预热。
    cudnn_benchmark=False,
    # set multi process parameters
    # 多进程配置。
    # 指定多进程启动方法为 fork，这是 Linux 系统中常用的进程创建方式。
    # 设置 OpenCV 的线程数为 0，意味着 OpenCV 将使用默认的线程数。
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    # 分布式训练配置。
    # 指定使用 NCCL 作为分布式通信后端，
    # NCCL 是 NVIDIA 收集通信库，适用于多 GPU 训练。
    dist_cfg=dict(backend='nccl'),
)
# set visualizer
# 设置可视化器为 None，不使用可视化工具
visualizer = None

# set log level
# 设置日志级别为 INFO
# 表示只记录信息级别以上的日志，不记录调试信息。
log_level = 'INFO'

# load from which checkpoint
# 设置从哪个检查点加载模型
# 这里设置为 None，意味着不从任何检查点加载
load_from = None

# whether to resume training from the loaded checkpoint
# 设置是否从加载的检查点恢复训练
# 这里设置为 False，意味着不恢复训练。
resume = False

# Defaults to use random seed and disable `deterministic`
# 设置随机种子为 None
# 即不固定随机种子，随机生成。
# deterministic=False: 设置不使用确定性算法
# 这可能会加快训练速度，但牺牲了结果的可重复性
randomness = dict(seed=None, deterministic=False)

# set log processor
# 日志处理是基于迭代而不是基于周期
log_processor = dict(by_epoch=False)
```
若模型进入正常迭代，则表明调试无误
### 1.6使用【训练管理】进行训练
创建一个训练任务，该任务使用6个节点（6台带IB资源的机器），预计需要训练1天19小时
运行命令为
```python
bash run.sh llama3_8b_instruct_qlora_agentflan_3e.py 6
```
出现以下类似代码表明训练顺利进行  
11/19 12:50:41 -mmengine -lNFO -Saving checkpoint at 2000 iterations11/19 12:57:36 - mmengine - lN0 - lter(trailn) [20104296] r: 1.1524e-04 eta: 23:14:06 time: 81.4554 data time:45.8042 memory: 6910 los:0.3550 tlops:3.2676 tokens per sec: 48.6446
[2024-11-19 13:01:10.055! WARNING! stage3.py.1998:stepl4 pvtorch alocator cache fushes since last step.this happens when there is high memory presure and is detimen  
平台上已经训练好模型，放在huggingface-cache数据下，挂载该数据后目录位于：/dataset/llama3-8b-functioncall-ft/agent-flan

### 1.7模型转换
创建一个新的notebook，进行到notebook，打开终端执行命令
这里训练到4296步取中间结果查看：
```python
# 模型转成hf格式
xtuner convert pth_to_hf \
/code/llama3_8b_instruct_qlora_agentflan_3e.py \
/dataset/llama3-8b-functioncall-ft/agent-flan/iter_4296.pth \
/code/llama3-8b-ft/agent-flan/iter_4296_hf
```
```python
# lora hf格式模型合并
xtuner convert merge /dataset/Llama-3-8B-Instruct/ \
/code/llama3-8b-ft/agent-flan/iter_4296_hf/ \
/code/llama3-8b-ft/agent-flan/merged --device cpu
```
执行final_test.py，调用模型测试模型微调前后的区别：  
`python /code/final_test.py `
```python
# 文件位于平台/code/final_test.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model_path = "/dataset/Llama-3-8B-Instruct/"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
```
```python
eval_prompt = """    
<|start_header_id|>system<|end_header_id|>

You are a assistant who can utilize external tools.
[{'name': 'ArxivSearch.get_arxiv_article_information','description': 'Run Arxiv search and get the article meta information.','parameters': [{'name': 'query', 'type': 'STRING', 'description':'the content of search query'}],'required': ['query'],'return_data':[{'name': 'content', 'description': 'a list of 3 arxiv search papers', 'type': 'STRING'}],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format{key: value}, where the key is the parameter name.'}]
To use a tool, please use the following format:
```
Thought:Think what you need to solve, do you need to use tools?
Action:the tool name, should be one of [['ArxivSearch']]
Action Input:the input to the action
```
The response after utilizing tools should using the following format:
```
Response:the results after call the tool.
```
If you already know the answer, or you do not need to use tools,
please using the following format to reply:
```
Thought:the thought process to get the final answer
Final Answer:final answer
```
Begin!<|eot_id|><|start_header_id|>user<|end_header_id|>

Please help me search the InternLM2 Technical Report.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
```
```python

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="cuda", use_cache=False)  # don't quantize here

base_model.eval()
with torch.no_grad():
    print(tokenizer.decode(base_model.generate(**model_input, max_new_tokens=256)[0], skip_special_tokens=True))

print("=========下面是微调后的模型=========")

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda:1")
merged_model_path = "/code/llama3-8b-ft/agent-flan/merged"

merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16, device_map="cuda:1", use_cache=False)  # don't quantize here

merged_model.eval()
with torch.no_grad():
    print(tokenizer.decode(merged_model.generate(**model_input, max_new_tokens=256)[0], skip_special_tokens=True))
```
微调前，没有准确给出function需要的json格式参数
![微调前](.images/微调前，没有准确给出function所需的Json格式参数.png)
微调后，可以正确给出json格式function参数 
![微调后](.images/微调后，可以正确给出json格式function参数：.png)
## 2.微调后模型下载并接入FastGPT
### 2.1下载模型
在平台上获取文件的下载链接，接下来就可以使用wget或者直接使用浏览器下载
```python
# 在自己的机器上建立一个文件夹，用于放置模型
wget http://hsw.csidc.cn/notebook_xd44276265ab4f84ba0a4146845631ea_task0/files/llama3-8b-ft/agent-flan/merged.zip?_xsrf=2%7C21ce93ab%7Cb08b04297a2d3192b14649aa2ef83391%7C1732277244 -O ./llama3-8b-ft.zip
```
### 2.2 使用xinference部署下载的模型
查看容器的配置信息： 使用 docker inspect 命令可以查看容器的详细配置信息，包括创建时使用的参数。  
docker inspect <container_id_or_name>
```python
# fastgpt 
sudo docker logs -f ab5afd3acf26
# oneapi
sudo docker logs --tail 100 25c995b90a1a
```
将模型放置xinference目录下，
### 2.3 
在该目录下启动xinference
```python
# 使用docker部署xinference并使用9997端口
sudo docker run -d \
    -e XINFERENCE_MODEL_SRC=modelscope \
    -v $(pwd)/.xinference:/root/.xinference \
    -v $(pwd):/home \
    -v $(pwd)/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/.cache/modelscope:/root/.cache/modelscope \
    -p 9997:9990 \
    --gpus all \
    registry.cn-hangzhou.aliyuncs.com/xprobe_xinference/xinference:latest   xinference-local -H 0.0.0.0
```
创建一个 xinference 容器，设置必要的环境变量和挂载卷，开放 9997 端口，并允许容器访问宿主机的所有 GPU。容器启动后，会运行 xinference-local 命令，以监听所有网络接口上的请求。
1. docker run：Docker 命令，用于创建并启动一个新的容器。
2. -d：以分离模式运行容器，即在后台运行。
3. -e XINFERENCE_MODEL_SRC=modelscope：设置环境变量 XINFERENCE_MODEL_SRC，值为 modelscope。这个环境变量可能被 xinference 服务用来确定模型的来源路径。
4. -v $(pwd)/.xinference:/root/.xinference：将宿主机当前目录下的 .xinference 目录挂载到容器的 /root/.xinference 目录。这样，容器可以访问宿主机上的配置文件。
5. -v $(pwd):/home：将宿主机当前目录挂载到容器的 /home 目录。这允许容器访问宿主机上的文件。
6. -v $(pwd)/.cache/huggingface:/root/.cache/huggingface：将宿主机当前目录下的 .cache/huggingface 目录挂载到容器的 /root/.cache/huggingface 目录。这通常用于存储 Hugging Face 模型的缓存。
7. -v $(pwd)/.cache/modelscope:/root/.cache/modelscope：将宿主机当前目录下的 .cache/modelscope 目录挂载到容器的 /root/.cache/modelscope 目录。这可能用于存储 modelscope 模型的缓存。
8. -p 9997:9997：将容器的 9997 端口映射到宿主机的 9997 端口。这样，可以通过宿主机的 9997 端口访问 xinference 服务。
9. --gpus all：允许容器访问宿主机上的所有 GPU 设备。
10. registry.cn-hangzhou.aliyuncs.com/xprobe_xinference/xinference：指定要拉取的 Docker 镜像地址。
11. xinference-local -H 0.0.0.0：在容器启动后执行的命令，xinference-local 是 xinference 的本地运行模式，-H 0.0.0.0 指定服务监听所有网络接口。

打开xinference界面，找到“llama-3-instruct”模型，如下所示：
![1](.images/11.png)
![2](.images/12.png)
![3](.images/11.png)
### 2.3接入FastGPT
安装fastgpt
1. 启动 docker
2. 下载 https://raw.githubusercontent.com/labring/FastGPT/main/projects/app/data/config.json
3. 下载 https://raw.githubusercontent.com/labring/FastGPT/main/files/docker/docker-compose-pgvector.yml，修改名字为docker-compose.yml
4. 新建fastgpt文件夹，将文件移动到文件夹中。
5. 启动容器
```python
# 启动容器
docker-compose up -d
# 等待10s，OneAPI第一次总是要重启几次才能连上Mysql
sleep 10
# 重启一次oneapi(由于OneAPI的默认Key有点问题，不重启的话会提示找不到渠道，临时手动重启一次解决，等待作者修复)
docker restart oneapi
```
http://localhost:3001/  
配置one api  
初始化账号密码：root，123456  
baseurl为本地ip，类型选择自定义渠道，名称为xinference部署的模型名称。  
docker inspect <container id> 查看fastgpt配置文件位置，具体位置跟自己创建的文件位置有关。  
写入LLM配置  
```python
{   "model": "llama-3-instruct",
    "name": "llama-3-instruct",
    "maxContext": 8000,
    "maxResponse": 8000,
    "quoteMaxToken": 8000,
    "maxTemperature": 1.2,
    "charsPointsPrice": 0,
    "censor": false,
    "vision": false,
    "datasetProcess": false,
    "usedInClassify": true,
    "usedInExtractFields": true,
    "usedInToolCall": true,
    "usedInQueryExtension": true,
    "toolChoice": false,
    "functionCall": false,
    "customCQPrompt": "",
    "customExtractPrompt": "",
    "defaultSystemChatPrompt": "",
    "defaultConfig": {}
},
```
```python
# 配置文件准备好后， 需要重启fastgpt
docker restart fastgpt
```
http://localhost:3000/  
打开FastGPT测试模型