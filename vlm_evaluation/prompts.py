import json

SYSTEM_PROMPT = """You are an expert evaluator assessing student artifacts for The Apprentice Project.
Your goal is to evaluate the provided image of a student's work based on the provided rubric.
You must be objective and provide a score.
"""


def generate_evaluation_prompt(student_id: str, artifact_type: str, rubric: dict) -> str:
    rubric_str = json.dumps(rubric, indent=2)
    return f"""USER: <image>
{SYSTEM_PROMPT}

Here is a student artifact (ID: {student_id}) for the category: {artifact_type}.

Rubric for Evaluation:
{rubric_str}

Please evaluate the artifact based on the rubric.
Provide your response as a JSON object matching the following schema:
{{
  "skill": "{rubric.get('skill', 'skill')}",
  "dimension": "{rubric.get('dimension', 'dimension')}",
  "score": <Your Score 1-{rubric.get('max', 5)}>,
  "max": {rubric.get('max', 5)}
}}

ASSISTANT:"""
