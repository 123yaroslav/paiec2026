"""Difficulty-judge prompt shared by offline rating and the submission runtime."""

INSTRUCTION = (
    "You estimate how hard a task is for current frontier AI systems. "
    "Read the task and rate its difficulty on a scale from 1 to 9, where 1 means almost every AI system "
    "solves it correctly and 9 means almost none does. Answer with a single digit only."
)


def _bounded(tok, text, max_tokens):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    return tok.decode(ids[:half]) + "\n...\n" + tok.decode(ids[-half:])


def build(tok, text, max_tokens=1024):
    messages = [{"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": "Task:\n" + _bounded(tok, text, max_tokens) + "\n\nDifficulty (1-9):"}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt
