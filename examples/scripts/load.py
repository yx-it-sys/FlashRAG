from sentence_transformers import SentenceTransformer
import os

# --- 定义模型名称和本地保存路径 ---
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
# 你可以自定义这个文件夹名称
local_model_path = '/mnt/data/models' 

# --- 步骤 1: 下载并保存模型 (如果文件夹不存在) ---

# 检查模型是否已经保存在本地
if not os.path.exists(local_model_path):
    print(f"本地模型未找到，正在从 Hugging Face 下载并保存到 '{local_model_path}'...")
    
    # 1. 从网络加载模型
    # 这一步会自动从 Hugging Face Hub 下载模型
    model = SentenceTransformer(model_name)
    
    # 2. 将模型保存到你指定的路径
    model.save(local_model_path)
    
    print("模型已成功保存。")
else:
    print(f"在 '{local_model_path}' 中找到本地模型。")


# --- 步骤 2: 从指定文件夹加载模型 ---

print("\n正在从本地文件夹加载模型...")
# 直接将本地文件夹路径传给 SentenceTransformer 构造函数
try:
    bert_model = SentenceTransformer(local_model_path)
    print("模型已成功从本地文件夹加载！")

    # --- 验证模型是否工作 ---
    sentences = ["这是一个测试。", "This is a test."]
    embeddings = bert_model.encode(sentences)
    print("\n模型工作正常，已生成向量。向量维度:", embeddings.shape)

except Exception as e:
    print(f"从本地加载模型失败: {e}")
    print("请检查路径是否正确，或尝试删除文件夹后重新运行脚本以下载模型。")