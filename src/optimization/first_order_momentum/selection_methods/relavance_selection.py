import os
import sys
import numpy as np

from sentence_transformers import SentenceTransformer
import torch

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.insert(0, GPO_ROOT_PATH)


def calculate_sentence_similarity(x, string_list, k):
    # 修改：使用作者確認的 BGE-base 模型
    # v1.5 是目前的標準版本，效果優於舊版
    model = SentenceTransformer('BAAI/bge-base-en-v1.5') 
    
    # 如果您的機器有 GPU，保留這行；如果是 Mac M1/M2 或純 CPU，請註解掉或改成 'cpu' / 'mps'
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu') 

    with torch.no_grad():
        # 注意：BGE v1.5 可以直接 encode，不需要額外的指令前綴 (instruction) 用於相似度計算
        embedding_x = model.encode([x], normalize_embeddings=True)
        embeddings_string_list = model.encode(string_list, normalize_embeddings=True)

    similarity = embedding_x @ embeddings_string_list.T
    
    top_k_similar_indices = np.argsort(similarity[0])[::-1][:k]

    return top_k_similar_indices

class Relavance_Selection:
    def select(self, history_list, select_num, momentum_para_name):
        if momentum_para_name in {'feedback'}:
            new = history_list[-1][0]
            selected_list = history_list[:-1][::-1]
            selected_list = [selected[0] for selected in selected_list]
            if len(selected_list) != 0:
                selected_idx = calculate_sentence_similarity(
                    new, selected_list, select_num - 1
                )
                selected_list = [selected_list[i] for i in selected_idx]
            selected_list.insert(0, new)
        else:
            assert momentum_para_name == "para"
            new = (history_list[-1][0], history_list[-1][1])
            selected_list = history_list[:-1][::-1]
            selected_sentence_list = [selected[0] for selected in selected_list]
            selected_list = [(selected[0], selected[1]) for selected in selected_list]
            if len(selected_sentence_list) != 0:
                selected_idx = calculate_sentence_similarity(
                    new[0], selected_sentence_list, select_num - 1
                )
                selected_list = [(selected_list[i][0], selected_list[i][1]) for i in selected_idx]
            selected_list.insert(0, new)

        return selected_list