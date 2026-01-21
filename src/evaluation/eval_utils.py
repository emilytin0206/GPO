"""The utility functions for evaluation."""
import re
import os
import sys
import pdb
import json
import string
import openai
import hashlib
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from concurrent.futures import ProcessPoolExecutor

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, GPO_ROOT_PATH)

from src.evaluation import metrics
from src.dataset.base import Base_Dataset

# ======================================================
# [NEW] User Provided Utilities
# ======================================================

def extract_tags(text: str, tag_name: str) -> list:
    """Extract content wrapped in specific tags."""
    if not text: return []
    
    # 1. 標準格式 <TAG_BEGIN>...</TAG_END> (忽略大小寫)
    pattern = f"<{tag_name}_BEGIN>(.*?)</{tag_name}_END>"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    # 2. 容錯格式
    if not matches:
        pattern_loose = f"<{tag_name}[ _]BEGIN>(.*?)</{tag_name}[ _]END>"
        matches = re.findall(pattern_loose, text, re.DOTALL | re.IGNORECASE)
        
    return [m.strip() for m in matches]

def insert_prompts_template(correct, wrong):
    """Format the prompt for feedback generation."""
    c = "\n".join(correct) if correct else "None"
    w = "\n".join(wrong) if wrong else "None"
    return f"Correct:\n{c}\n---\nWrong:\n{w}"

def file_has_content(filepath: str) -> bool:
    if not os.path.exists(filepath):
        return False
    return os.path.getsize(filepath) > 0

# ======================================================
# Existing Utils (Polished)
# ======================================================

def read_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def polish_sentence(sentence, add_ending_punc=False):
    sentence = sentence.strip()
    if sentence:
        sentence = sentence.replace("**", "")
        if len(sentence) > 1:
            sentence = sentence[0].upper() + sentence[1:]
        if add_ending_punc and not (sentence.endswith(".") or sentence.endswith("?") or sentence.endswith("!")):
            sentence += "."
    return sentence

def remove_punctuation_from_string(input_string, is_filename=True):
    if is_filename:
        punctuation_subset_str = string.punctuation.replace("!", "").replace("?", "").replace(".", "")
        output_string = input_string.translate(str.maketrans("", "", punctuation_subset_str))
        output_string = output_string.replace("!", "<EXCLAMATION>").replace("?", "<QUESTION>").replace(".", "<PERIOD>")
    else:
        output_string = input_string.translate(str.maketrans("", "", string.punctuation))
    return output_string

def instruction_to_filename(instruction, md5_hashing=True):
    if md5_hashing:
        m = hashlib.md5()
        m.update(instruction.encode("utf-8"))
        filename = m.hexdigest()
    else:
        filename = instruction.replace("\n", "")
        filename = remove_punctuation_from_string(repr(filename))
        filename = filename if filename else "<NO INSTRUCTION>"
    return filename

def _get_accuracy(
    dataset_name, task_name, true_answer, pred_answer, treat_include_as_correct=False
):
    """
    Revised accuracy check relying on metrics.extract_choice.
    """
    true_answer = str(true_answer).lower().strip()
    pred_answer = str(pred_answer).lower().strip() # extract_choice returns "A", so "a" here
    
    # 處理 True Answer 可能包含括號的情況 (A) -> a
    bracketed_set = set([f"({l})" for l in string.ascii_lowercase])
    if true_answer in bracketed_set:
        true_answer = re.findall(r"\(.*?\)", true_answer)[0][1]

    # 如果是數字比較 (BBH specific logic kept)
    if dataset_name == "bbh" and task_name == "word_sorting":
        return int(pred_answer == true_answer)

    # 核心比較: 精確匹配
    # 因為我們現在有了 robust parsing，pred_answer 應該是乾淨的 "a" 或數字
    is_exact = (pred_answer == true_answer)
    
    return int(is_exact)
    
# the Boolean symbols appeared in BBH tasks
BOOLEAN_SYMBOLS = [["false", "true"], ["no", "yes"], ["invalid", "valid"]]

all_lowercase_letters = string.ascii_lowercase  # "abcd...xyz"
bracketed_lowercase_letters_set = set(
    [f"({l})" for l in all_lowercase_letters]
)  # {"(a)", ...}
bracketed_uppercase_letters_set = set(
    [f"({l.upper()})" for l in all_lowercase_letters]
)  # {"(A)", ...}


def read_jsonl(filepath):
    """Read the jsonl file (AQuA raw data)."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def polish_sentence(sentence, add_ending_punc=False):
    """Standardize the sentence to English syntax."""
    sentence = sentence.strip()
    if sentence:
        sentence = sentence.replace("**", "")
        if len(sentence) > 1:
            sentence = sentence[0].upper() + sentence[1:]  # capitalize the first letter
        if add_ending_punc and not (
            sentence.endswith(".") or sentence.endswith("?") or sentence.endswith("!")
        ):
            sentence += "."
    return sentence


def remove_punctuation_from_string(input_string, is_filename=True):
    """Remove punctuations from string to comply with filename requirements."""
    # remove punctuations other than "!", "?", "."
    if is_filename:
        punctuation_subset_str = (
            string.punctuation.replace("!", "").replace("?", "").replace(".", "")
        )
        output_string = input_string.translate(
            str.maketrans("", "", punctuation_subset_str)
        )
        # replace punctuations "!", "?", "." with indicating letters
        output_string = (
            output_string.replace("!", "<EXCLAMATION>")
            .replace("?", "<QUESTION>")
            .replace(".", "<PERIOD>")
        )
    else:
        output_string = input_string.translate(
            str.maketrans("", "", string.punctuation)
        )
    return output_string


def instruction_to_filename(instruction, md5_hashing=True):
    """Convert an instruction string to filename."""
    if md5_hashing:
        m = hashlib.md5()
        m.update(instruction.encode("utf-8"))
        filename = m.hexdigest()
    else:
        # remove punctuations and line break, and give a name to the empty string
        filename = instruction.replace("\n", "")
        filename = remove_punctuation_from_string(repr(filename))
        filename = filename if filename else "<NO INSTRUCTION>"
    return filename


def get_second_round_prompt(dataset_name, task_name, end_choice=None):

    numeric_list = ["gsm8k"]

    if dataset_name == "bbh":
        # bool type
        if task_name == "boolean_expressions":
            second_round_prompt = (
                """Directly tell me your answer using only "True" or "False"."""
            )
        elif task_name in {
            "causal_judgement",
            "navigate",
            "sports_understanding",
            "web_of_lies",
        }:
            second_round_prompt = (
                """Directly tell me your answer using only "Yes" or "No"."""
            )
        elif task_name == "formal_fallacies":
            second_round_prompt = (
                """Directly tell me your answer using only "valid" or "invalid"."""
            )

        # multiple choice
        elif task_name in {
            "date_understanding",
            "disambiguation_qa",
            "geometric_shapes",
            "hyperbaton",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
            "logical_deduction_three_objects",
            "movie_recommendation",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "ruin_names",
            "salient_translation_error_detection",
            "snarks",
            "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
        }:
            second_round_prompt = f"Therefore, summarize your answer with an option among (A) and {end_choice}."

        # numeric
        elif task_name in {"multistep_arithmetic_two", "object_counting"}:
            second_round_prompt = """Therefore, summarize your final answer with just one number in your response."""

        # generative
        elif task_name == "dyck_languages":
            second_round_prompt = """Directly output the newly added parentheses that make the parentheses close correctly."""
        elif task_name == "word_sorting":
            second_round_prompt = """Directly output the result of sorting the words alphabetically, separating each word with a space."""

    elif dataset_name == "mmlu":
        second_round_prompt = f"Therefore, summarize your answer with an option among (A) and (D)."

    elif dataset_name == "wsc":
        second_round_prompt = f"Therefore, summarize your answer with an option between (A) and (B)."

    elif dataset_name in numeric_list:
        second_round_prompt = """Therefore, summarize your final answer with just one number in your response."""
    
    return second_round_prompt


def _split_by_Q(sentence):
    """Split the response and only keep the part before the first "Q:"."""
    return sentence.split("Q:")[0].strip()


def gen_prompt(
    data,
    instruction,
    idx,
    evaluate_generated_ins_on_few_shot=False,
    few_shot_data=None,
    few_shot_num=3,
    include_qa=True,
    instruction_pos="A_begin",
    dataset_name="mmlu",
    is_chat_model=False,
):
    """Generate a prompt from the available exemplars and the given instruction."""
    dataset_name = dataset_name.lower()
    Dataset_class = Base_Dataset(dataset_name)
    question = Dataset_class.get_single_question(data, idx)

    prompt = ""
    if not evaluate_generated_ins_on_few_shot:
        if include_qa:
            if instruction_pos == "before_Q":
                if instruction:
                    prompt += instruction + "\n"
                prompt += "Q: " + question
                prompt += "\nA: "
            elif instruction_pos == "Q_begin":
                if instruction:
                    prompt += "Q: " + instruction + " "
                else:
                    prompt += "Q: "
                prompt += question
                prompt += "\nA: "
            elif instruction_pos == "Q_end":
                prompt += "Q: " + question
                if instruction:
                    prompt += " " + instruction + "\nA: "
                else:
                    prompt += "\nA: "
            else:
                prompt += f"Q: {question}\n"
                prompt += "A:"
                if instruction:
                    prompt += f" {instruction} "
        else: 
            if instruction_pos == "Q_begin":
                if instruction:
                    prompt += instruction + "\n"
                prompt += question
            else:  # instruction_pos == "Q_end"
                prompt += question
                if instruction:
                    prompt += "\n" + instruction
    else:
        if not include_qa:
            if instruction_pos == "Q_begin":
                if not is_chat_model:
                    for data in few_shot_data[:few_shot_num]:
                        prompt += instruction + "\n" + Dataset_class.get_q(data) + "\n" + Dataset_class.get_a(data) + "\n\n"
                    if instruction:
                        prompt += instruction + "\n"
                    prompt += question
            else:  # instruction_pos == "Q_end"
                if not is_chat_model:
                    for data in few_shot_data[:few_shot_num]:
                        if instruction:
                            prompt +=  Dataset_class.get_q(data) + "\n" + instruction + "\n" + Dataset_class.get_a(data) + "\n\n"
                        else:
                            prompt +=  Dataset_class.get_q(data) + "\n" + Dataset_class.get_a(data) + "\n\n"
                    prompt += question
                    if instruction:
                        prompt += "\n" + instruction
                else:
                    prompt = []
                    if instruction:
                        for data in few_shot_data[:few_shot_num]:
                            prompt.append({"role": "user", "content": Dataset_class.get_q(data) + "\n" + instruction})
                            prompt.append({"role": "assistant", "content": Dataset_class.get_a(data)})
                        prompt.append({"role": "user", "content": question})
                    else:
                        for data in few_shot_data[:few_shot_num]:
                            prompt.append({"role": "user", "content": Dataset_class.get_q(data)})
                            prompt.append({"role": "assistant", "content": Dataset_class.get_a(data)})
                        prompt.append({"role": "user", "content": question})
        else:
            if instruction_pos == "A_begin":
                for data in few_shot_data[:few_shot_num]:
                    if instruction:
                        prompt += "Q: " + Dataset_class.get_q(data) + "\n" + "A: " + instruction + " " + Dataset_class.get_a(data) + "\n\n"
                    else:
                        prompt += "Q: " + Dataset_class.get_q(data) + "\n" + "A: " + Dataset_class.get_a(data) + "\n\n"
                prompt += f"Q: {question}\n"
                prompt += "A:"
                if instruction:
                    prompt += f" {instruction} "
    
    return prompt


def fetch_true_answer(data, idx, dataset_name):
    """Fetch the true answer of the dataset at the idx'th position."""
    dataset_name = dataset_name.lower()
    Dataset_class = Base_Dataset(dataset_name)
    answer = Dataset_class.get_single_answer(data, idx)
    return answer


def _get_index_from_symbol(answer):
    """Get the index from the letter symbols A, B, C, D."""
    answer = str(answer).lower()
    if answer in bracketed_lowercase_letters_set:
        answer = re.findall(r"\(.*?\)", answer)[0][1]
    index = ord(answer) - ord("a")
    return index


def _get_accuracy(
    dataset_name, task_name, true_answer, pred_answer, treat_include_as_correct=False
):
    """Get the accuracy of a prediction."""
    true_answer = str(true_answer).lower().strip()
    pred_answer = str(pred_answer).lower().strip()
    true_answer_included_in_pred_answer = true_answer in pred_answer
    
    # 完整保留特殊任務邏輯 (Dyck Languages, Word Sorting 等)
    if dataset_name == "bbh" and task_name == "dyck_languages":
        pred_pattern = r"[>\)\]}]+[ >\)\]}]*"
        pred_matches = re.findall(pred_pattern, pred_answer)
        if len(pred_matches) > 0:
            pred_matches = [item.strip() for item in pred_matches]
            return int(pred_matches[-1] == true_answer)
        else:
            return 0
    
    elif dataset_name == "bbh" and task_name == "word_sorting":
        pred_answer_list = [item.strip() for item in pred_answer.split("\n") if len(item.split(' ')) == 1]
        true_answer_list = [item.strip() for item in true_answer.split(" ")]
        if pred_answer_list != true_answer_list:
            pred_answer_list = [item.strip() for item in pred_answer.split(" ") if len(item.split(' ')) == 1]
        if pred_answer_list != true_answer_list:
            pred_answer_list = pred_answer.split("\n")
            for pred_ans in pred_answer_list:
                pred_ans.strip()
                pred_ans_parse = pred_ans.split(' ')
                if pred_ans_parse == true_answer_list:
                    return 1
        return int(pred_answer_list == true_answer_list)

    if true_answer in bracketed_lowercase_letters_set:
        true_answer = re.findall(r"\(.*?\)", true_answer)[0][1]
    if pred_answer in bracketed_lowercase_letters_set:
        pred_answer = re.findall(r"\(.*?\)", pred_answer)[0][1]
        
    result_exact_match = (pred_answer == true_answer) or (
        remove_punctuation_from_string(pred_answer, is_filename=False).strip()
        == remove_punctuation_from_string(true_answer, is_filename=False).strip()
    )
    
    # 完整保留 Boolean 邏輯
    is_boolean_match = False
    if any([true_answer in item for item in BOOLEAN_SYMBOLS]):
        boolean_type_index = np.where(
            [true_answer in item for item in BOOLEAN_SYMBOLS]
        )[0][0]
        true_answer_as_true_or_false_str = str(
            bool(
                np.where(np.array(BOOLEAN_SYMBOLS[boolean_type_index]) == true_answer)[
                    0
                ][0]
            )
        ).lower()
        if pred_answer in {"0", "1"}:
            pred_answer = str(bool(int(pred_answer))).lower()
        is_boolean_match = (
            pred_answer == true_answer_as_true_or_false_str
            or pred_answer.strip() == true_answer_as_true_or_false_str.strip()
        )
        
    accuracy = int(result_exact_match or is_boolean_match)
    if treat_include_as_correct:
        accuracy = int(bool(accuracy) or true_answer_included_in_pred_answer)
    return accuracy


def evaluate_with_api_key(task, *args, **kwargs):
    api_key, scorer_llm_name, instruction = task
    openai.api_key = api_key
    return instruction, evaluate_single_instruction(
        scorer_llm_name=scorer_llm_name, instruction=instruction, *args, **kwargs
    )


def evaluate_parallel_instructions(
    api_keys, scorer_llm_name, instructions, *args, **kwargs
):
    """
    Evaluate instructions. 
    Modified to run serial evaluation for ANY model name (to support Qwen/Ollama).
    """
    results = {}
    
    # 這裡移除了原本嚴格的模型名稱檢查
    # 只要呼叫此函式，就對所有 instruction 進行評估
    print(f"Evaluating {len(instructions)} instructions using {scorer_llm_name}...")
    
    for instruction in tqdm(instructions, desc="Evaluating Instructions"):
        try:
            results[instruction] = evaluate_single_instruction(
                scorer_llm_name=scorer_llm_name,
                instruction=instruction,
                *args,
                **kwargs,
            )
        except Exception as e:
            print(f"Error evaluating instruction '{instruction}': {e}")
            # 發生錯誤時回傳 0 分或空 dataframe 以避免中斷
            # 這裡回傳一個空的 result 以便後續偵錯，或者 raise e
            pass

    return results


def evaluate_single_instruction(
    data,
    scorer_llm_name,
    instruction,
    eval_index_all,
    tokenizer,
    call_server_func,
    dataset_name,
    task_name,
    extract_final_answer_by_prompting_again,
    instruction_pos,
    is_multiple_choice,
    evaluate_generated_ins_on_few_shot,
    few_shot_data,
    few_shot_num,
    include_qa,
    prediction_treat_as_number=False,
    prediction_treat_as_bool=False,
    prediction_treat_as_rouge=False,
    prediction_num_decimals=0,
):
    assert instruction_pos in {
        "before_Q",
        "Q_begin",
        "Q_end",
        "A_begin",
    }, (
        "The instruction position should be either before the question, or at the"
        " beginning of the question, at the end of the question, or at the"
        " beginning of the answer."
    )

    num_eval_examples = len(eval_index_all)

    true_answers = [
        fetch_true_answer(data, idx=idx, dataset_name=dataset_name)
        for idx in eval_index_all
    ]

    # ====================== generate raw prompts ======================
    
    if scorer_llm_name in {
        "llama2-chat-7b",
        "llama2-chat-13b",
    }:
        is_chat_model = True
    else:
        is_chat_model = False
    
    raw_prompts_flattened = []
    for i in trange(num_eval_examples, desc="Generating First round Prompts"):
        raw_prompt = gen_prompt(
            data=data,
            instruction=instruction,
            idx=eval_index_all[i],
            evaluate_generated_ins_on_few_shot=evaluate_generated_ins_on_few_shot,
            few_shot_data=few_shot_data,
            few_shot_num=few_shot_num,
            include_qa=include_qa,
            instruction_pos=instruction_pos,
            dataset_name=dataset_name,
            is_chat_model=is_chat_model
        )
        raw_prompts_flattened.append(raw_prompt)

    if is_chat_model and evaluate_generated_ins_on_few_shot:
        if scorer_llm_name in {
            "llama2-chat-7b",
            "llama2-chat-13b",
        }:
            raw_prompts_flattened_idx = []
            for raw_prompt in raw_prompts_flattened:
                prompt_id = tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True)
                raw_prompts_flattened_idx.append(prompt_id)
    else:
        if scorer_llm_name in {
            "llama2-chat-7b",
            "llama2-chat-13b",
        }:
            conv_flattened = []
            raw_prompts_flattened_idx = []
            for raw_prompt in raw_prompts_flattened:
                conv = [{"role": "user", "content": raw_prompt}]
                conv_flattened.append(conv)
                prompt_id = tokenizer.apply_chat_template(conv, add_generation_prompt=True)
                raw_prompts_flattened_idx.append(prompt_id)
        elif scorer_llm_name in {"llama2-7b"}:
            pass
        else:
            # 修正點：移除對 gpt-3.5/gpt-4 的強制檢查，允許 Qwen 通過
            conv_flattened = []
            for raw_prompt in raw_prompts_flattened:
                conv = [{"role": "user", "content": raw_prompt}]
                conv_flattened.append(conv)

    # ====================== get first round answers ======================
    # 修正點：放寬條件，非 Llama 模型都走通用介面 (call_server_func)
    if scorer_llm_name not in {"llama2-7b", "llama2-chat-7b", "llama2-chat-13b"}:
        # 適用於 GPT, Qwen (via API/Ollama) 等接受字串輸入的模型
        raw_answers = []
        for prompt in tqdm(raw_prompts_flattened, desc="Get First Round Answers"):
            # 防呆處理：某些 API 回傳 list，某些回傳 str
            response = call_server_func(prompt)
            if isinstance(response, list) or isinstance(response, tuple):
                raw_answers.append(response[0])
            else:
                raw_answers.append(response)

    elif scorer_llm_name in {"llama2-7b"}:
        raw_answers = call_server_func(raw_prompts_flattened)
    else:
        # Llama chat
        raw_answers = call_server_func(raw_prompts_flattened_idx)
    
    # ====================== generate second round prompt ======================
    if extract_final_answer_by_prompting_again and is_chat_model and not evaluate_generated_ins_on_few_shot:
        prompts_second_round_flattened = []
        prompts_second_round_flattened_idx = []
        for conv, raw_answer in tqdm(
            zip(conv_flattened, raw_answers), desc="Generating Second round Prompts"
        ):
            if is_multiple_choice:
                if instruction_pos == "Q_end":
                    end_choice = metrics._extract_bracketed_choice_from_string(
                        conv[0]["content"].split("\n")[-2], is_lower=False
                    )
                elif instruction_pos == "Q_begin":
                    end_choice = metrics._extract_bracketed_choice_from_string(
                        conv[0]["content"].split("\n")[-1], is_lower=False
                    )
                second_round_prompt = get_second_round_prompt(
                    dataset_name, task_name, end_choice
                )
            else:
                second_round_prompt = get_second_round_prompt(dataset_name, task_name)
            if include_qa:
                raw_answer = _split_by_Q(raw_answer)
            conv.append({"role": "assistant", "content": raw_answer})
            conv.append({"role": "user", "content": second_round_prompt})
            prompts_second_round_flattened.append(conv)
            if scorer_llm_name in {
                "llama2-chat-7b",
                "llama2-chat-13b",
            }:
                second_round_prompt_idx = tokenizer.apply_chat_template(
                    conv, add_generation_prompt=True
                )
                prompts_second_round_flattened_idx.append(second_round_prompt_idx)

        # ====================== get second round answers ======================
        # 修正點：允許通用模型進入此區塊 (雖然 Qwen 預設 is_chat_model=False 可能不會進來，但為了相容性修改)
        if scorer_llm_name not in {"llama2-chat-7b", "llama2-chat-13b"}:
            raw_answers_second_round = []
            for prompt in tqdm(prompts_second_round_flattened, desc="Get Second Round Answers"):
                 # 注意：這裡原本傳入的是 conversation list，需確認 call_server_func 是否支援
                 # 若使用 Qwen API，可能需要轉成 string 或保持 list
                 # 這裡假設您的 call_server_func 能處理 list (如 openai style)
                 response = call_server_func(prompt, max_decode_steps=100)
                 if isinstance(response, list) or isinstance(response, tuple):
                    raw_answers_second_round.append(response[0])
                 else:
                    raw_answers_second_round.append(response)
        else:
            raw_answers_second_round = call_server_func(
                prompts_second_round_flattened_idx, max_decode_steps=100
            )

    # ====================== extract answers ======================
    def _extract_second_round_answer_for_parsing(ans):
        if evaluate_generated_ins_on_few_shot and not is_chat_model:
            return ans.strip(":").strip().split("\n")[0]
        else:
            return ans.strip()

    raw_answers_to_parse = (
        list(map(_extract_second_round_answer_for_parsing, raw_answers_second_round))
        if extract_final_answer_by_prompting_again and is_chat_model and not evaluate_generated_ins_on_few_shot
        else list(map(_extract_second_round_answer_for_parsing, raw_answers))
    )
    
    # ====================== calc rouge ======================
    if prediction_treat_as_rouge and not extract_final_answer_by_prompting_again:
        accuracies = [
            metrics.calc_rouge(
                raw_answer_to_parse, true_answer
            )
            for raw_answer_to_parse, true_answer in zip(raw_answers_to_parse, true_answers)
        ]

    # ====================== parse answers ======================
    else:
        raw_answers_to_parse = [
            metrics.get_normalized_prediction(
                ans,
                is_multiple_choice,
                prediction_treat_as_number,
                prediction_num_decimals,
                prediction_treat_as_bool,
            )
            for ans in raw_answers_to_parse
        ]

        # ====================== calculate accuracies ======================
        accuracies = []
        for i, _ in tqdm(
            enumerate(eval_index_all), desc="Extract Answers from Predictions"
        ):
            accuracy = _get_accuracy(
                dataset_name=dataset_name,
                task_name=task_name,
                true_answer=true_answers[i],
                pred_answer=raw_answers_to_parse[i],
                treat_include_as_correct=False,
            )
            accuracies.append(accuracy)

    # ====================== save results ======================
    detailed_results_df = pd.DataFrame(
        list(
            zip(
                eval_index_all,
                raw_prompts_flattened,
                raw_answers,
                raw_answers_to_parse,
                true_answers,
                accuracies,
            )
        ),
        columns=[
            "index_in_raw_dataset",
            "raw_prompt",
            "raw_answer",
            "parsed_answer",
            "true_answer",
            "accuracy",
        ],
    )
    if extract_final_answer_by_prompting_again and is_chat_model and not evaluate_generated_ins_on_few_shot:
        detailed_results_df.insert(
            3, "raw_prompt_second_round", prompts_second_round_flattened
        )
        detailed_results_df.insert(
            4, "raw_answer_second_round", raw_answers_second_round
        )

    detailed_results_df.set_index("index_in_raw_dataset", inplace=True)
    return detailed_results_df