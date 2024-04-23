
# LLAMA3

Llama1 Content Length: 2K
Llama2 Content Length: 4K
Llama3 Content Length: 8K

## Debian环境部署

设置python虚拟环境
```bash
> sudo apt install python3-venv  python3-pip
> cd /opt/Data/PythonVenv
> python3 -m venv llama3
> source /opt/Data/PythonVenv/llama3/bin/activate
```

部署推理环境

```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> 
```

使用官方模型推理
```bash
> torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir /opt/Data/ModelWeight/meta/llama3/Meta-Llama-3-8B \
    --tokenizer_path /opt/Data/ModelWeight/meta/llama3/Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
>
> torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir /opt/Data/ModelWeight/meta/llama3/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path /opt/Data/ModelWeight/meta/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
>
```

启动服务
```bash
> python WebGradio/WebGradioAutoModelForCausalLM.py
> jupyter notebook --no-browser --port 7001 --ip=192.168.2.198
> jupyter notebook --no-browser --port 7000 --ip=192.168.2.200
```