"""
LongMemEval Converter

将 LongMemEval 数据集转换为 Locomo 格式。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from eval.converters.base import BaseConverter
from eval.converters.registry import register_converter


def convert_time_format(input_str: str) -> str:
    """
    将格式为 "YYYY/MM/DD (Day) HH:MM" 的时间字符串
    转换为 "H:MM am/pm on D Month, YYYY" 的格式。
    """
    # 输入格式: %Y: 年份, %m: 月份, %d: 日期, %a: 星期缩写, %H: 24小时制小时, %M: 分钟
    input_format = "%Y/%m/%d (%a) %H:%M"
    
    # 解析输入字符串为 datetime 对象
    dt_object = datetime.strptime(input_str, input_format)
    
    # 输出格式: %-I: 12小时制小时(无前导零), %M: 分钟, %p: AM/PM, 
    #          %-d: 日期(无前导零), %B: 月份全称, %Y: 年份
    output_format = "%-I:%M %p on %-d %B, %Y"
    
    # 格式化为目标字符串，并将 AM/PM 转为小写
    formatted_string = dt_object.strftime(output_format).lower()
    
    # 确保月份首字母大写
    parts = formatted_string.split(' ')
    parts[4] = parts[4].capitalize()
    
    return ' '.join(parts)


def convert_lmeval_s_to_locomo_style(lmeval_data: list) -> list:
    """
    将 LongMemEval-S 格式转换为 Locomo 格式
    
    Args:
        lmeval_data: LongMemEval-S 原始数据
        
    Returns:
        Locomo 格式数据
    """
    locomo_style_data = []
    
    for data in lmeval_data:
        data_dict = {
            "qa": [],
            "conversation": {}
        }
        
        # 找出包含答案的 session 索引
        evidence_session_idx = []
        for idx, session_id in enumerate(data["haystack_session_ids"]):
            if session_id in data["answer_session_ids"]:
                evidence_session_idx.append(idx)
        
        # 标记包含答案的消息
        for idx, session in enumerate(data["haystack_sessions"]):
            for i, msg in enumerate(session):
                data["haystack_sessions"][idx][i]["has_answer"] = idx in evidence_session_idx
        
        # 收集 evidence
        evidence = []
        for idx, session in enumerate(data["haystack_sessions"]):
            for i, msg in enumerate(session):
                if msg["has_answer"]:
                    evidence.append(f"D{idx}:{i}")
        
        # 构建 QA
        data_dict["qa"].append({
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"],
            "evidence": evidence,
            "category": data["question_type"]
        })
        
        # 构建对话
        data_dict["conversation"]["speaker_a"] = f"user_{data['question_id']}"
        data_dict["conversation"]["speaker_b"] = f"assistant_{data['question_id']}"
        
        for idx, session in enumerate(data["haystack_sessions"]):
            data_dict["conversation"][f"session_{idx}_date_time"] = convert_time_format(
                data["haystack_dates"][idx]
            )
            data_dict["conversation"][f"session_{idx}"] = []
            
            for i, msg in enumerate(session):
                data_dict["conversation"][f"session_{idx}"].append({
                    "speaker": msg["role"] + f"_{data['question_id']}",
                    "text": msg["content"],
                    "dia_id": f"D{idx}:{i}"
                })
        
        locomo_style_data.append(data_dict)
    
    return locomo_style_data


@register_converter("longmemeval")
class LongMemEvalConverter(BaseConverter):
    """LongMemEval 数据集转换器"""
    
    def get_input_files(self) -> Dict[str, str]:
        """返回需要的输入文件"""
        return {
            "raw": "longmemeval_s_cleaned.json"
        }
    
    def get_output_filename(self) -> str:
        """返回输出文件名"""
        return "longmemeval_s_locomo_style.json"
    
    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        """
        执行转换
        
        Args:
            input_paths: {"raw": "path/to/longmemeval_s_cleaned.json"}
            output_path: 输出文件路径
        """
        print(f"🔄 Converting LongMemEval to Locomo format...")
        
        # 读取原始数据
        with open(input_paths["raw"], "r", encoding="utf-8") as f:
            lmeval_data = json.load(f)
        
        print(f"   Loaded {len(lmeval_data)} items")
        
        # 转换格式
        locomo_style_data = convert_lmeval_s_to_locomo_style(lmeval_data)
        
        # 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_style_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved {len(locomo_style_data)} entries to {output_path}")

