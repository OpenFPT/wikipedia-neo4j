"""Prompt templates for agent and decomposition."""

from __future__ import annotations

COMPLEXITY_DETECTION_PROMPT = """Analyze this Vietnamese question and determine its complexity level (1-5):
- Level 1: Simple factual question about one entity (e.g., "Ai là tác giả của X?")
- Level 2: Question about relationships between 2 entities (e.g., "X làm việc ở đâu?")
- Level 3: Multi-hop question requiring 2-3 steps (e.g., "Ai là bạn của tác giả của X?")
- Level 4: Complex multi-hop with multiple branches (e.g., "Những công ty nào được thành lập bởi những người từng làm việc tại Y?")
- Level 5: Very complex with aggregation or comparison (e.g., "So sánh các thành tích của những người từng làm việc tại X và Y")

Question: {question}

Respond with ONLY a JSON object:
{{"complexity": <1-5>, "reasoning": "brief explanation"}}"""

DECOMPOSE_QUESTION_PROMPT = """Break down this complex Vietnamese question into 2-4 simpler sub-questions that can be answered sequentially.

Original question: {question}

Each sub-question should:
1. Be answerable independently using the knowledge graph
2. Build on previous answers
3. Progress toward answering the original question

Respond with ONLY a JSON object:
{{
  "sub_questions": [
    {{"order": 1, "question": "first sub-question"}},
    {{"order": 2, "question": "second sub-question"}},
    ...
  ],
  "synthesis_instruction": "how to combine answers to get final answer"
}}"""

SYNTHESIS_PROMPT = """Given the original question and answers to sub-questions, synthesize a final answer.

Original question: {question}

Sub-questions and answers:
{sub_qa}

Provide a comprehensive answer in Vietnamese that addresses the original question, citing the sources from the sub-question answers."""
