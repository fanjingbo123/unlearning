import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import math
import copy
from tqdm import tqdm
from datasets import load_dataset,load_from_disk
def get_token_probabilities(model, tokenizer, text):
    # Tokenize 输入文本
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]  # Shape: (sequence_length,)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    probabilities = []
    with torch.no_grad():
        # 获取模型的 logits
        outputs = model(input_ids=input_ids.unsqueeze(0))
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
        # 计算 softmax 概率
        probs = F.softmax(logits, dim=-1)
        
        # 遍历每个 token，从第二个 token 开始
        for i in range(1, len(input_ids)):
            # 获取当前位置 token 的 ID
            current_token_id = input_ids[i].item()
            
            # 获取前一个位置的概率分布
            token_probs = probs[0, i-1]
            
            # 获取当前 token 的概率
            token_prob = token_probs[current_token_id].item()
            
            # 解码当前 token
            current_token = tokenizer.decode([current_token_id])
            
            # 获取前缀
            prefix = tokenizer.decode(input_ids[:i])
            
            probabilities.append({
                "prefix": prefix,
                "token": current_token,
                "probability": token_prob
            })
    
    return probabilities

def compute_P(probs):
    result = []
    adder = 0
    for item in probs:
        probability = item['probability']
        adder = math.log(probability)
        result.append(adder)
    return result

def perturb_model_parameters(model, mean=0.0, std=1e-5):
    with torch.no_grad(): 
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 生成与参数相同形状的正态分布扰动
                noise = torch.randn_like(param) * std + mean
                # 将扰动添加到参数中
                param.add_(noise)

def compute_MRD(Pt,Pt_perturb):
    # result = [abs(a - b) / abs(a) for a, b in zip(Pt, Pt_perturb)]
    assert len(Pt)==len(Pt_perturb)
    # result = [(a - b) / a for a, b in zip(Pt, Pt_perturb)]
    result = []
    for a, b in zip(Pt, Pt_perturb):
        if a != 0:
            result.append((a - b) / a)

    return sum(result)

def main(K):
    # 设置模型名称
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False  # 根据需要设置，通常中文使用慢速 tokenizer 可能更合适
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 根据硬件支持情况选择 dtype，如不支持 bfloat16，可改为 torch.float16
        cache_dir="./.cache",  # 可根据需要设置缓存目录
        low_cpu_mem_usage=True,
        device_map="auto",  # 自动选择设备（GPU/CPU）
        # use_auth_token='YOUR_HUGGINGFACE_TOKEN'  # 如果模型需要认证，取消注释并填写您的 token
    )
    # 设置模型为评估模式
    model.eval()
    
    for i in range(0,25):
        if i<4:
            continue
        texts = []
        MRDs = []
        train_dataset = load_from_disk("/home/yz979/project/chengye/test/wmdp/chunk_"+str(i))
        # for q,a in zip(train_dataset['question'],train_dataset['answer']):
        #     texts.append(q+a)
        for q in zip(train_dataset['text']):
            texts.append(q)
        original_state = copy.deepcopy(model.state_dict())
        for text in texts[0:3]:
            MRD = 0
            seed = 42
            for i in tqdm(range(0,K)):
                torch.manual_seed(seed+i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed+i)
                model.load_state_dict(original_state)
                probs = get_token_probabilities(model, tokenizer, text)
                Pt = compute_P(probs)
                perturb_model_parameters(model)
                probs_perturb = get_token_probabilities(model, tokenizer, text)
                Pt_perturb = compute_P(probs_perturb)
                temp = compute_MRD(Pt,Pt_perturb)
                MRD += temp
                
            MRD = abs(MRD/K)
            # with open('chunk99_100.txt', 'a') as f:
            #     f.write(f"{text}\n") 
            #     f.write(f"MRD: {abs(MRD)}\n\n")
            # print("MRD:", abs(MRD), "\n\n")
            print(text)
            print(MRD)
            MRDs.append(MRD)
        res = 0
        for i in MRDs:
            res += i
        res = res/len(MRDs)
        with open('/home/yz979/project/chengye/WAGLE/wmdp_mrd.txt', 'a') as f:
            f.write(f"{res}\n") 
        


if __name__ == "__main__":
    main(5)

