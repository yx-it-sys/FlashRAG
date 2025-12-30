# 安装FlashRAG
使用源码安装：
```bash
git clone https://github.com/yx-iy-sys/FlashRAG.git
cd FlashRAG
pip install -e .
pip install toml
conda install -c pytorch -c nvidia faiss-gpu
```
运行期间会遇到一些库依赖报错，但是都容易解决，具体我忘了:(

# 运行OmniSearch Pipeline：
- 项目运行的一切脚本都位于``examples/scripts``中，结果会保存在``examples/scripts/result``里
```bash
result/
├── okvqa_dummy_100_2025_10_13_13_40_experiment # 每次结果都会按照结束时间唯一标识，防止覆盖
│   ├── config.yaml # 当前实验配置
│   ├── intermediate_data.json  # 中间结果
│   ├── metric_score.txt    # 得分
│   └── output.jsonl    # 更详细的结果，包括上下文，Token隐藏状态，Confidence Score等
└── okvqa_dummy_100_2025_10_13_14_39_experiment
    └── config.yaml
    ├── ...
```
- 目前仅支持显式地给出模型存储路径，而非自动到HF缓存中寻找，下载模型到本地之后符号链接即可，我在``examples/scripts/models/Qwen2.5-VL-7B-Instruct``链接了模型文件。
- Prompt让我单独拿出来了，方便查看与模块化修改，位于``examples/scripts/omni_prompt.toml``
- 按照原项目，各种运行参数在`my_config.yaml`与`run_omni_pipeline.py`中的`config_dict`混合配置，虽然不太方便，也不太懂为啥，但还是照做了。
```python
    config_dict = {
        "data_dir": "data/",    # 数据集annotations存放位置
        "image_path": "data/images/val2014",    # 图片存放位置
        "index_path": "indexes/bm25",   # 索引地址
        "corpus_path": "indexes/bm25/corpus.jsonl", # 语料库位置
        "generator_model": "Qwen2.5-VL-7B-Instruct",
        "generator_model_path": args.model_path,    # 模型存放位置
        "retrieval_method": "bm25", # Sparse Retrieval
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 1,    # 返回1个检索结果
        "save_intermediate_data": True, #保存中间结果
    }
```
``my_config.yaml``只需要修改一处：
```yaml
# ----Environment Settings----
gpu_id: "0" # gpu id
dataset_name: "okvqa_dummy_100" # 数据集名称（.jsonl所在文件夹）
split: ["dev",'test']   # 数据集划分(.jsonl文件名)
```
- 运行：
```bash
python run_omni_pipeline.py --model_path models/Qwen2.5-VL-7B-Instruct/
```