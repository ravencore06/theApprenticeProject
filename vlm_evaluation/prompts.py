SYSTEM_PROMPT = """You are an expert evaluator assessing student artifacts for The Apprentice Project.
Your goal is to evaluate the provided image of a student's work based on the provided rubric.
You must be objective and provide a score along with a brief explanation.
"""


def generate_evaluation_prompt(student_id: str, artifact_type: str, rubric: str) -> str:
    return f"""USER: <image>
{SYSTEM_PROMPT}

Here is a student artifact (ID: {student_id}) for the category: {artifact_type}.

Rubric for Evaluation:
{rubric}

Please evaluate the artifact based on the rubric.
Provide your response in the following format:
SCORE: [Your Score 1-5]
FEEDBACK: [Your reasoning here]

ASSISTANT:"""
