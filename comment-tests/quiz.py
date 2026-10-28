"""Small graded quality check: problems with known answers, greedy decoding.

Usage: .venv/bin/python quiz.py --model models/<dir> --tag name
"""
import argparse, json, re, pathlib, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT = pathlib.Path(__file__).parent
Q = [  # (question, expected answer substring)
    ("A train leaves at 9:40, travels 210 km at 84 km/h, waits 25 minutes, then travels 150 km at "
     "100 km/h. What 24-hour clock time does it arrive?", "14:05"),
    ("What is 37 * 43 - 128?", "1463"),
    ("A shirt costs $80. It is discounted 25%, then the discounted price is raised 10%. Final price in dollars?", "66"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines take to make 100 widgets?", "5"),
    ("How many prime numbers are there between 50 and 80?", "7"),
    ("Sum of the interior angles of a heptagon, in degrees?", "900"),
    ("A bat and ball cost $1.10 total; the bat costs $1.00 more than the ball. Ball price in cents?", "5"),
    ("What is the remainder when 2^20 is divided by 7?", "4"),
    ("Alice is 3 times as old as Bob. In 12 years she will be twice as old as him. How old is Alice now?", "36"),
    ("How many distinct ways can the letters of the word LEVEL be arranged?", "30"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    model, tok = load(a.model)
    greedy = make_sampler(temp=0.0)
    score, log = 0, []
    t0 = time.time()
    for q, ans in Q:
        msgs = [{"role": "user", "content": q + " Think briefly, then end with 'ANSWER: <value>'."}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        out = generate(model, tok, prompt, max_tokens=700, sampler=greedy)
        m = re.findall(r"ANSWER:\s*\**\s*\$?([^\n*]+)", out)
        got = m[-1].strip() if m else "(none)"
        ok = re.search(rf"(?<![\d.]){re.escape(ans)}(?![\d])", got) is not None
        score += ok
        log.append(dict(q=q, expected=ans, got=got, ok=ok))
        print(f"{'OK ' if ok else 'BAD'} expected {ans:>6}  got {got[:40]}")
    print(f"== {a.tag}: {score}/{len(Q)} in {time.time() - t0:.0f}s")
    (ROOT / "results" / f"quiz-{a.tag}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
