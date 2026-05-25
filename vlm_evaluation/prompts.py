import json
SYSTEM_PROMPT = """You are an expert evaluator assessing student artifacts for The Apprentice Project.
You must output your evaluation STRICTLY as a valid JSON object. Do not include any other conversational text."""

def generate_evaluation_prompt(student_id: str, artifact_type: str, rubric: str) -> str:
    return f"""USER: 
{SYSTEM_PROMPT}

Artifact ID: {student_id}
Category: {artifact_type}
Rubric Schema:
{rubric}

Please evaluate the artifact based on the rubric.
Output strictly in this JSON format:
{{"score": <int>, "feedback": "<brief reasoning>"}}
ASSISTANT:"""
