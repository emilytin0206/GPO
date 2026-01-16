import os
import json
import random

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

mmlu_tasks = [
    'abstract_algebra', 
    'anatomy', 
    'astronomy', 
    'business_ethics', 
    'clinical_knowledge', 
    'college_biology', 
    'college_chemistry', 
    'college_computer_science', 
    'college_mathematics', 
    'college_medicine', 
    'college_physics', 
    'computer_security', 
    'conceptual_physics', 
    'econometrics', 
    'electrical_engineering', 
    'elementary_mathematics', 
    'formal_logic', 
    'global_facts', 
    'high_school_biology', 
    'high_school_chemistry', 
    'high_school_computer_science', 
    'high_school_european_history', 
    'high_school_geography', 
    'high_school_government_and_politics', 
    'high_school_macroeconomics', 
    'high_school_mathematics', 
    'high_school_microeconomics', 
    'high_school_physics', 
    'high_school_psychology', 
    'high_school_statistics', 
    'high_school_us_history', 
    'high_school_world_history', 
    'human_aging', 
    'human_sexuality', 
    'international_law', 
    'jurisprudence', 
    'logical_fallacies', 
    'machine_learning', 
    'management', 
    'marketing', 
    'medical_genetics', 
    'miscellaneous', 
    'moral_disputes', 
    'moral_scenarios', 
    'nutrition', 
    'philosophy', 
    'prehistory', 
    'professional_accounting', 
    'professional_law', 
    'professional_medicine', 
    'professional_psychology', 
    'public_relations', 
    'security_studies', 
    'sociology', 
    'us_foreign_policy', 
    'virology', 
    'world_religions'
    ]




def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            json_object = json.loads(line)
            data.append(json_object)
    return data

class MMLU_Dataset:

    def get_ratio(self):
        train_ratio = 1  
        eval_ratio = 1  
        test_ratio = 1  
        print(
            f"[Train ratio]: {train_ratio}, [Eval ratio]: {eval_ratio}, [Test ratio]: {test_ratio}"
        )
        return train_ratio, eval_ratio, test_ratio
    
    def read_data(self, task, mmlu_subsets=['all'], mmlu_train_num=-1, shuffle_train_data=True):

        datas = []
        root_data_folder_path = os.path.join(GPO_ROOT_PATH, f"data/MMLU/")
        
        # 1. 篩選要跑的子集 (Subsets)
        if isinstance(task, list):
            requested_tasks = task
        elif task == "all":
            requested_tasks = mmlu_tasks
        else:
            requested_tasks = [task]

        if 'all' in mmlu_subsets:
            config_subsets = mmlu_tasks
        else:
            config_subsets = mmlu_subsets
            
        tasks_to_run = sorted(list(set(requested_tasks) & set(config_subsets)))

        if not tasks_to_run:
            print("Warning: No tasks to run after filtering.")
            return []

        # === 準備合併後的容器 ===
        merged_train_data = []
        merged_eval_data = []
        merged_test_data = []
        merged_few_shot_data = []
        merged_format_data = []

        print(f"Loading MMLU subsets: {tasks_to_run}")

        for t in tasks_to_run:
            task_data_folder_path = os.path.join(GPO_ROOT_PATH, f"data/MMLU/{t}")
            
            f_train = os.path.join(task_data_folder_path, f"train.jsonl")
            f_eval = os.path.join(task_data_folder_path, f"eval.jsonl")
            f_test = os.path.join(task_data_folder_path, f"test.jsonl")
            f_few_shot_examples = os.path.join(task_data_folder_path, f"few_shot_examples.jsonl")
            f_format = os.path.join(task_data_folder_path, f"format.jsonl")
            
            # 讀取資料
            train_data = read_jsonl(f_train)
            eval_data = read_jsonl(f_eval)
            test_data = read_jsonl(f_test)
            few_shot_examples = read_jsonl(f_few_shot_examples)
            format_data = read_jsonl(f_format)

            # === 2. 實作：每個子集各取 n 個 (train_num) ===
            if mmlu_train_num > 0:
                # 如果資料不夠 n 筆，slice 會自動取到最大長度 (即全部)，不會報錯
                train_data = train_data[:mmlu_train_num]

            # 將處理後的資料加入合併清單
            merged_train_data.extend(train_data)
            merged_eval_data.extend(eval_data)
            merged_test_data.extend(test_data)
            merged_few_shot_data.extend(few_shot_examples)
            merged_format_data.extend(format_data)

        # === 3. 實作：打散 (Shuffle) ===
        if shuffle_train_data:
            print(f"Shuffling merged training data (Total: {len(merged_train_data)})...")
            random.seed(42) # 設定 seed 確保實驗可重現
            random.shuffle(merged_train_data)
            
            # 也可以選擇性地打散 eval/test，通常保持順序或打散皆可，這裡主要針對 train
            # random.shuffle(merged_eval_data) 

        # 回傳單一合併後的 Task 物件
        # Task 名稱設為 "mmlu_merged" 或是子集名稱的組合，這裡用 generic 名稱避免檔名過長
        task_name_str = "mmlu_merged" 
        
        datas.append({
            "task": task_name_str,
            "train_data": merged_train_data,
            "eval_data": merged_eval_data,
            "test_data": merged_test_data,
            "few_shot_data": merged_few_shot_data,
            "format_data": merged_format_data,
            "train_num_examples": len(merged_train_data),
            "eval_num_examples": len(merged_eval_data),
            "test_num_examples": len(merged_test_data),
            "few_shot_num_examples": len(merged_few_shot_data),
            "format_num_examples": len(merged_format_data),
        })

        return datas
    
    def get_single_question(self, data, idx):
        return data[idx]["input"]
    
    def get_single_answer(self, data, idx):
        return data[idx]["output"]
    
    def get_single_solution(self, data, idx):
        pass
    
    
    def get_q(self, data):
        return data["input"]
    
    def get_a(self, data):
        return data["output"]

    def get_task_setting(self, task):
        return {
            "is_multiple_choice": True,
            "prediction_treat_as_number": False,
            "prediction_treat_as_bool": False,
            "prediction_treat_as_rouge": False,
            "extract_final_answer_by_prompting_again": True,
        }