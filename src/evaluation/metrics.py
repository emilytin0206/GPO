"""Final answer parser for reasoning tasks."""

import dataclasses
import re
import string
from typing import Dict, List, Sequence
from rouge import Rouge
import pdb

# ======================================================
# [NEW] Engineering Extraction Logic (您的新引擎)
# ======================================================

def to_float_maybe(s: str) -> float:
    """Try to convert string to float, handling commas."""
    if not s: 
        raise ValueError
    # 移除非數字字符，但保留 . 和 -
    matches = re.findall(r'-?\d+\.?\d*', s.replace(',', ''))
    if matches: 
        return float(matches[-1])
    raise ValueError

def extract_choice(s: str) -> str:
    """
    Robust extraction of multiple-choice answers (A/B/C/D/E).
    """
    if not s: 
        raise ValueError("Empty input string")
    
    # 1. 預處理
    text = s.strip()

    # 2. [最強優先級] LaTeX Boxed 格式: \boxed{A}
    match_boxed = re.search(r'\\boxed\{\s*([A-E])\s*\}', text, re.IGNORECASE)
    if match_boxed:
        return match_boxed.group(1).upper()

    # 3. [關鍵邏輯] 標準化與切割 (定位結論區)
    text_lower = text.lower()
    keywords = ['answer is', 'answer:', 'the answer is', 'correct answer is', 'option:', 'choice:']
    
    found_keyword = False
    for pat in keywords:
        if pat in text_lower:
            # 使用 rsplit 確保我們抓的是最後一次出現的關鍵字
            text_lower = text_lower.rsplit(pat, 1)[-1].strip()
            found_keyword = True 
            break
            
    # 4. [提取選項] 根據是否鎖定結論區，決定抓頭還是抓尾
    
    # 4.1 尋找括號格式: (A), (B)
    matches_paren = re.findall(r'\(([A-E])\)', text_lower, re.IGNORECASE)
    if matches_paren:
        # 如果有鎖定結論區 -> 答案通常在開頭 -> 取第一個
        # 如果沒鎖定 (全文) -> 答案通常在結尾 -> 取最後一個
        return matches_paren[0].upper() if found_keyword else matches_paren[-1].upper()
        
    # 4.2 尋找單獨字母: A, B (需有邊界 \b)
    matches_word = re.findall(r'\b([A-E])\b', text_lower, re.IGNORECASE)
    if matches_word:
        return matches_word[0].upper() if found_keyword else matches_word[-1].upper()

    # 5. [保底策略] 極簡字串處理 (針對直接輸出 "A." 的情況)
    if len(s.strip()) < 10:
        match_simple = re.search(r'([A-E])', s, re.IGNORECASE)
        if match_simple:
            return match_simple.group(1).upper()

    # 若真的什麼都沒抓到
    raise ValueError(f"No choice found in: {s}")

# ======================================================
# Existing GPO Helpers
# ======================================================

word2num = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12"
}
BOOLEAN_SYMBOLS = [["false", "true"], ["no", "yes"], ["invalid", "valid"]]

def calc_rouge(pred, ans):
    rouge = Rouge()
    if isinstance(ans, str):
        scores = rouge.get_scores(pred, ans)[0]['rouge-l']['f']
    elif isinstance(ans, list):
        scores_list = [rouge.get_scores(pred, a)[0]['rouge-l']['f'] for a in ans]
        scores = sum(scores_list) / len(scores_list)
    return scores

def remove_punctuation_from_string(input_string):
    output_string = input_string.translate(str.maketrans("", "", string.punctuation))
    return output_string

# ======================================================
# Main Parsing Function (已修正：呼叫 extract_choice)
# ======================================================

def get_normalized_prediction(
    prediction: str,
    is_multiple_choice: bool,
    treat_as_number: bool,
    num_decimals: int = 0,
    treat_as_bool: bool = False,
) -> str:
    """
    Returns a normalized prediction using robust extraction logic.
    """
    prediction = prediction.strip()
    
    # 1. [修正點] 處理選擇題 -> 強制使用 extract_choice
    if is_multiple_choice:
        try:
            return extract_choice(prediction)
        except ValueError:
            # 如果抓不到，回傳空字串，避免舊邏輯誤判
            return ""

    # 2. [修正點] 處理數字 -> 強制使用 to_float_maybe
    if treat_as_number:
        try:
            val = to_float_maybe(prediction)
            # 格式化小數位數
            if num_decimals > 0:
                return "{:.{prec}f}".format(val, prec=num_decimals)
            else:
                # 如果是整數，去掉 .0
                if val.is_integer():
                    return str(int(val))
                return str(val)
        except ValueError:
            # 抓不到數字就回傳原始字串
            pass

    # 3. 處理布林值 (Boolean) -> 維持原有邏輯
    if treat_as_bool:
        prediction_lower = prediction.lower()
        for idx, pair in enumerate(BOOLEAN_SYMBOLS):
            if pair[0] in prediction_lower: return "false" 
            if pair[1] in prediction_lower: return "true"
        prediction_parsed = remove_punctuation_from_string(prediction).lower().strip()
        return prediction_parsed

    # 4. 預設處理 (文字)
    prediction_parsed = prediction.lower()
    keywords = ["answer is", "answer:", "is:"]
    for k in keywords:
        if k in prediction_parsed:
            prediction_parsed = prediction_parsed.split(k)[-1]
    
    return prediction_parsed.strip()


@dataclasses.dataclass
class NormalizationResult:
    target: str
    prediction: str
    treat_as_number: bool
    num_decimals: int

def get_normalized_target_and_prediction(
    target: str, prediction: str
) -> NormalizationResult:
    # Target 處理
    target = target.lower().strip()
    if "answer is" in target:
        target = target.split("answer is")[-1].strip()
    
    treat_as_number = False
    try:
        float(target)
        treat_as_number = True
    except:
        pass

    num_decimals = 0
    if treat_as_number and "." in target:
        num_decimals = len(target.split(".")[-1])

    # 這裡呼叫修正後的 get_normalized_prediction
    normalized_prediction = get_normalized_prediction(
        prediction, 
        is_multiple_choice=False, 
        treat_as_number=treat_as_number, 
        num_decimals=num_decimals
    )

    return NormalizationResult(
        target=target,
        prediction=normalized_prediction,
        treat_as_number=treat_as_number,
        num_decimals=num_decimals,
    )

def number_included_accuracy_list(
    targets: Sequence[str],
    predictions: Sequence[str],
) -> List[bool]:
    correct_list = []
    for prediction, target in zip(predictions, targets):
        res = get_normalized_target_and_prediction(target, prediction)
        if res.treat_as_number:
            try:
                correct = abs(float(res.prediction) - float(res.target)) <= 1e-5
            except:
                correct = False
        else:
            correct = res.target == res.prediction
        correct_list.append(correct)
    return correct_list

def number_included_accuracy(
    targets: Sequence[str], predictions: Sequence[str]
) -> Dict[str, float]:
    lst = number_included_accuracy_list(targets, predictions)
    acc = sum(lst) / len(lst) * 100 if lst else 0.0
    return {"accuracy": acc, "accuracy_with_calc": acc}

if __name__ == "__main__":
    # 測試用
    ans = get_normalized_prediction(prediction="Based on the above, the answer is (C).", is_multiple_choice=True, treat_as_number=False)
    print(f"Test Result: {ans}") # 預期輸出: C