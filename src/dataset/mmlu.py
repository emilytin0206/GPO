import os
import json
import random

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

mmlu_tasks = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
    'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 
    'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
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
        print(f"[Train ratio]: {train_ratio}, [Eval ratio]: {eval_ratio}, [Test ratio]: {test_ratio}")
        return train_ratio, eval_ratio, test_ratio
    
    def read_data(self, task, mmlu_subsets=['all'], mmlu_train_num=-1, shuffle_train_data=True):
        datas = []
        
        # 1. 處理任務列表
        if isinstance(task, list):
            requested_tasks = mmlu_tasks if 'all' in task else task
        elif task == "all":
            requested_tasks = mmlu_tasks
        else:
            requested_tasks = [task] # 即使是單一任務字串，也轉為列表處理

        # 2. 處理子集過濾
        if isinstance(mmlu_subsets, list) and 'all' in mmlu_subsets:
            config_subsets = mmlu_tasks
        elif isinstance(mmlu_subsets, list):
            config_subsets = mmlu_subsets
        else:
            # 如果傳入的是逗號分隔字串 (從 sh 傳入時常發生)
            config_subsets = mmlu_subsets.split(',') if isinstance(mmlu_subsets, str) else [mmlu_subsets]
            
        tasks_to_run = sorted(list(set(requested_tasks) & set(config_subsets)))

        if not tasks_to_run:
            print("Warning: No tasks to run after filtering.")
            return []

        # 3. 合併數據容器
        merged_train_data = []
        merged_eval_data = []
        merged_test_data = []
        merged_few_shot_data = []
        merged_format_data = []

        print(f"Merging data from tasks: {tasks_to_run}")

        for t in tasks_to_run:
            task_data_folder_path = os.path.join(GPO_ROOT_PATH, f"data/MMLU/{t}")
            
            # 讀取各個檔案
            train_data = read_jsonl(os.path.join(task_data_folder_path, "train.jsonl"))
            eval_data = read_jsonl(os.path.join(task_data_folder_path, "eval.jsonl"))
            test_data = read_jsonl(os.path.join(task_data_folder_path, "test.jsonl"))
            few_shot_data = read_jsonl(os.path.join(task_data_folder_path, "few_shot_examples.jsonl"))
            format_data = read_jsonl(os.path.join(task_data_folder_path, "format.jsonl"))

            # 限制每個子任務的訓練筆數
            if mmlu_train_num > 0:
                train_data = train_data[:mmlu_train_num]

            # 合併數據
            merged_train_data.extend(train_data)
            merged_eval_data.extend(eval_data)
            merged_test_data.extend(test_data)
            # Few-shot 只要取一部分即可，避免過大，這裡示範只取第一個或全部合併
            merged_few_shot_data.extend(few_shot_data)
            merged_format_data.extend(format_data)

        # 4. 打散訓練資料 (混合不同科目的題目)
        if shuffle_train_data:
            print(f"Shuffling merged training data (Total: {len(merged_train_data)})...")
            random.seed(42)
            random.shuffle(merged_train_data)

        # 5. 回傳單一合併後的任務物件
        # 我們將 task 名稱設為 "mmlu_merged_tasks"，這樣 optimize.py 只會跑一次
        datas.append({
            "task": "mmlu_merged_tasks",
            "train_data": merged_train_data,
            "train_num_examples": len(merged_train_data),
            "eval_data": merged_eval_data,
            "eval_num_examples": len(merged_eval_data),
            "test_data": merged_test_data,
            "test_num_examples": len(merged_test_data),
            "few_shot_data": merged_few_shot_data,
            "few_shot_num_examples": len(merged_few_shot_data),
            "format_data": merged_format_data,
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