import gc
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed


def make_run_dir(base_dir, resume=None):
    """Return the folder for this run: `base_dir/resume` if given, else a new `run_<i>_<datetime>`."""
    base_dir = Path(base_dir)
    if resume:
        run_dir = base_dir / resume
        assert run_dir.exists(), f"{run_dir} does not exist, cannot resume it"
        return run_dir
    indices = [int(m.group(1)) for d in base_dir.glob("run_*") if (m := re.match(r"run_(\d+)_", d.name))]
    run_dir = base_dir / f"run_{max(indices, default=0) + 1}_{datetime.now():%Y-%m-%d_%H-%M}"
    run_dir.mkdir(parents=True)
    return run_dir


def write_run_summary(run_dir, **params):
    """Write the experiment parameters to `run_dir/parameters.txt` (one `name: value` per line)."""
    lines = [f"run: {Path(run_dir).name}", f"written: {datetime.now():%Y-%m-%d %H:%M:%S}"]
    lines += [f"{k}: {v}" for k, v in params.items()]
    (Path(run_dir) / "parameters.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prompt(tokenizer, question, chat_template, instruction):
    """Question + instruction, wrapped in the chat template if the model uses one."""
    text = question + "\n" + instruction
    if not chat_template:
        return text + "\n"
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def boxed_answers(text):
    """Contents of all non-empty \\boxed{...} in the text, in order (handles nested braces like \\boxed{\\frac{1}{2}})."""
    answers = []
    for match in re.finditer(r"\\boxed\{", text):
        depth = 1
        for j in range(match.end(), len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                if text[match.end():j].strip():
                    answers.append(text[match.end():j].strip())
                break
    return answers


def extract_answer(text, thinks):
    """Last box after </think> for thinking models, first box otherwise. None if there is no answer."""
    if thinks:
        if "</think>" not in text:
            return None                       # still thinking when the token limit was reached
        boxes = boxed_answers(text.split("</think>")[-1])
        return boxes[-1] if boxes else None
    boxes = boxed_answers(text)
    return boxes[0] if boxes else None


def to_number(answer):
    """'$1,234' -> 1234.0, '20\\%' -> 20.0, '\\frac{3}{4}' -> 0.75, None if no number."""
    if answer is None:
        return None
    a = re.sub(r"\\[dt]?frac\{(\d+)\}\{(\d+)\}", r"\1/\2", answer).replace(",", "")
    if m := re.fullmatch(r"\s*(-?\d+)/(\d+)\s*", a):
        return int(m.group(1)) / int(m.group(2))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", a)
    return float(numbers[-1]) if numbers else None


def is_correct(answer, gold):
    """True if the extracted answer is numerically equal to the gold answer."""
    value = to_number(answer)
    return value is not None and math.isclose(value, float(gold), abs_tol=1e-6)


def answer_cell(r, thinks):
    """Short text for the comparison table: '18 ✓', '180 ✗', 'cut off' or 'no answer'."""
    answer = extract_answer(r["output"], thinks)
    if answer is None:
        return "no answer" if r["finished"] else "cut off"
    return f"{answer}  {'✓' if is_correct(answer, r['gold']) else '✗'}"


def load_results(path):
    """Read a .jsonl results file; returns [] if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass    # a half-written line from a disconnect is ignored (that problem is simply redone)
    return results


def save_results(path, results):
    """Append results to a .jsonl file and force them to disk."""
    with open(path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())    # make sure it really reaches Google Drive


def load_model(model_name):
    """Load model (float16, on GPU) and tokenizer set up for batched generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"           # needed for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to("cuda").eval()
    print(f"Loaded {model_name} | GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer


def free_gpu():
    """Release GPU memory of objects that are no longer referenced."""
    gc.collect()
    torch.cuda.empty_cache()


class StopAfterAnswer(StoppingCriteria):
    """Stops each sequence as soon as it contains a complete, non-empty \\boxed{...} answer."""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer, self.prompt_length = tokenizer, prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_length:], skip_special_tokens=True)
        return torch.tensor([bool(boxed_answers(t)) for t in texts], device=input_ids.device)


@torch.no_grad()
def generate_answers(model, tokenizer, questions, cfg, instruction, temperature, top_p):
    """Generate one answer per question. Returns a list of (text, n_tokens, finished)."""
    prompts = [build_prompt(tokenizer, q, cfg["chat_template"], instruction) for q in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    prompt_length = inputs["input_ids"].shape[1]

    # Stop at the normal end token, and also at <|im_end|> (end of a chat turn for Qwen models)
    stop_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        stop_ids.append(im_end)

    if cfg["sampling"]:
        decoding = dict(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        decoding = dict(do_sample=False, temperature=None, top_p=None, top_k=None)   # greedy
    outputs = model.generate(
        **inputs, **decoding,
        max_new_tokens=cfg["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
        stopping_criteria=StoppingCriteriaList([StopAfterAnswer(tokenizer, prompt_length)]) if cfg["stop_after_answer"] else None,
    )

    results = []
    for tokens in outputs[:, prompt_length:].tolist():
        # Everything after the first stop token is padding.
        end = next((k for k, t in enumerate(tokens) if t in stop_ids), None)
        n_tokens = end if end is not None else len(tokens)
        text = tokenizer.decode(tokens[:n_tokens], skip_special_tokens=True)
        stopped_after_answer = cfg["stop_after_answer"] and bool(boxed_answers(text))
        finished = end is not None or stopped_after_answer or n_tokens < cfg["max_new_tokens"]
        results.append((text, n_tokens, finished))
    return results


def run_model(model_key, cfg, problems, results_dir, instruction, batch_size, seed, temperature, top_p):
    """Run one model on all problems not yet in `results_dir/<model_key>.jsonl`, saving after every batch."""
    path = Path(results_dir) / f"{model_key}.jsonl"
    done_ids = {r["id"] for r in load_results(path)}
    todo = [p for p in problems if p["id"] not in done_ids]
    print(f"{model_key}: {len(done_ids)} problems already done, {len(todo)} to go. Saving to {path}")
    if not todo:
        return

    model, tokenizer = load_model(cfg["name"])
    set_seed(seed)
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        t0 = time.time()
        outputs = generate_answers(model, tokenizer, [p["question"] for p in batch], cfg, instruction, temperature, top_p)
        batch_time = time.time() - t0         # one generate call for the whole batch, so this is per batch
        results = []
        for p, (text, n_tokens, finished) in zip(batch, outputs):
            answer = extract_answer(text, cfg["thinks"])
            results.append({**p, "model": cfg["name"], "output": text, "answer": answer,
                            "correct": is_correct(answer, p["gold"]),
                            "n_tokens": n_tokens, "finished": finished, "time": batch_time,
                            "chain_of_thought_length": text.count("\n\n")})
        save_results(path, results)
        print(f"  saved {len(done_ids) + start + len(batch)}/{len(problems)}  ({batch_time:.0f} s)")
    del model, tokenizer
    free_gpu()

    # Short sanity check, so problems with the setup are noticed right away.
    results = load_results(path)
    print(f"Done: {sum(r['correct'] for r in results)}/{len(results)} correct | "
          f"{sum(not r['finished'] for r in results)} cut off at the token limit | "
          f"{sum(r['answer'] is None for r in results)} without a boxed answer")
