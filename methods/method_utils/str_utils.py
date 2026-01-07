import re

def extract_last_question(cot_prompt):
    # 查找所有的 "question:" 或 "claim:"（大小写不敏感）
    question_or_claim_matches = re.findall(r"(question:|claim:)", cot_prompt, re.IGNORECASE)
    
    if len(question_or_claim_matches) > 0:
        # 如果有 "question:" 或 "claim:"，提取最后一个 "question:" 或 "claim:" 后的内容
        last_question_or_claim_index = cot_prompt.lower().rfind("question:") if "question:" in cot_prompt.lower() else cot_prompt.lower().rfind("claim:")  # 查找 "question:" 或 "claim:"
        # 截取最后一个 "question:" 或 "claim:" 后的内容
        last_question_or_claim_content = cot_prompt[last_question_or_claim_index + len(question_or_claim_matches[-1]):].strip()
        return last_question_or_claim_content
    else:
        # 如果没有 "question:" 或 "claim:"，返回整个内容
        return cot_prompt.strip()


def extract_last_answer(cot_prompt):
    # 查找所有的 "answer:"（大小写不敏感）
    answer_matches = re.findall(r"answer:", cot_prompt, re.IGNORECASE)
    
    if len(answer_matches) > 0:
        # 如果有 "answer:"，提取最后一个 "answer:" 后的内容
        last_answer_index = cot_prompt.lower().rfind("answer:")  # 查找最后一个 "answer:"
        # 截取最后一个 "answer:" 后的内容
        last_answer_content = cot_prompt[last_answer_index + len("answer:"):].strip()
        return last_answer_content
    else:
        # 如果没有 "answer:"，返回整个内容
        return cot_prompt.strip()

# 测试
cot_prompt = '''Task: Answer the given question step-by-step, and conclude with the phrase 'so the final answer is: .
question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins? Answer: Step 1, when did Theodor Haecker die? Theodor Haecker was 65 years old when he died. Step 2, when did  Harry Vaughan Watkins die? Harry Vaughan Watkins was 69 years old when he died. Step 3, so the final answer is: Harry Vaughan Watkins. End of answer.
Claim: Why did the founder of Versus die? Answer: Step 1, who is the funder of Versus? The founder of Versus was Gianni Versace. Step 2, why did Gianni Versace die? Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997. Step 3, so the final answer is: Shot. End of answer.
question: Who is the grandchild of Dambar Shah? Answer: Step 1, who is the son of  Dambar Shah? Dambar Shah (? - 1645) was the father of Krishna Shah. Step 2, who is the son of Krishna Shah? Krishna Shah (? - 1661) was the father of Rudra Shah. Step 3, so the final answer is: Rudra Shah. End of answer.
claim: Who invented the telephone? Answer: Step 1, who is credited with inventing the telephone? The telephone was invented by Alexander Graham Bell. Step 2, when did Alexander Graham Bell invent the telephone? Alexander Graham Bell patented the invention on March 7, 1876. Step 3, so the final answer is: Alexander Graham Bell. End of answer.
'''

# 调用函数并打印结果
last_question_or_claim = extract_last_question(cot_prompt)
print(last_question_or_claim)  # 输出: Who invented the telephone?

def add_step_tags(text: str, step_tag: str = "ки") -> str:
        """
        为推理文本的每个步骤后添加 step_tag 标记 (Math-Shepherd PRM 格式要求)
        
        支持的步骤格式:
        1. "Step 1: ... Step 2: ..." 或 "Step 1: ...\nStep 2: ..."
        2. "1. ... 2. ..." 或 "1) ... 2) ..."
        3. 以换行符分隔的多行文本
        
        Args:
            text: 原始推理文本
            step_tag: 步骤标记，默认为 "ки"
            
        Returns:
            添加了步骤标记的文本
        """
        # 如果文本已经包含 step_tag，直接返回
        if step_tag in text:
            return text
        
        # 模式1: 匹配 "Step N:" 格式 (不区分大小写)
        step_pattern = re.compile(r'(Step\s+\d+\s*[:.：])', re.IGNORECASE)
        if step_pattern.search(text):
            # 在每个 "Step N:" 之前插入 step_tag（除了第一个）
            parts = step_pattern.split(text)
            result = parts[0]  # 第一部分（可能是空或问题文本）
            for i in range(1, len(parts), 2):
                step_marker = parts[i]  # "Step N:"
                step_content = parts[i + 1] if i + 1 < len(parts) else ""
                if i > 1:  # 不是第一个 step
                    result = result.rstrip() + f" {step_tag}\n"
                result += step_marker + step_content
            # 在最后一个步骤后添加 step_tag
            result = result.rstrip() + f" {step_tag}"
            return result
        
        # 模式2: 匹配 "N." 或 "N)" 格式的编号
        numbered_pattern = re.compile(r'(\n\s*\d+[.)]\s+)')
        if numbered_pattern.search(text):
            parts = numbered_pattern.split(text)
            result = parts[0]
            for i in range(1, len(parts), 2):
                number_marker = parts[i]
                step_content = parts[i + 1] if i + 1 < len(parts) else ""
                if result.strip():  # 前面有内容
                    result = result.rstrip() + f" {step_tag}"
                result += number_marker + step_content
            result = result.rstrip() + f" {step_tag}"
            return result
        
        # 模式3: 按换行符分隔（每行作为一个步骤）
        lines = text.strip().split('\n')
        if len(lines) > 1:
            tagged_lines = []
            for line in lines:
                line = line.strip()
                if line:  # 非空行
                    tagged_lines.append(f"{line} {step_tag}")
            return '\n'.join(tagged_lines)
        
        # 单行文本，直接在末尾添加 step_tag
        return f"{text.strip()} {step_tag}"