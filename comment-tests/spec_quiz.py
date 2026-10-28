"""Speculative decoding check: the same quiz, same model, same greedy settings, with and without the MTP drafter.

Usage: .venv/bin/python spec_quiz.py --model models/<dir> --tag NAME [--draft mlx-community/Qwen3.8-27B-MTP-bf16]
Writes results/spec_<tag>.json with every full answer, so two runs can be compared word for word.
"""
import argparse, json, re, pathlib, time
import mlx.core as mx
from mlx_vlm import load, generate
from quiz import Q

ROOT = pathlib.Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--draft")
    p.add_argument("--block", type=int, help="draft block size (default: the drafter's own)")
    a = p.parse_args()
    model, proc = load(a.model)
    tok = getattr(proc, "tokenizer", proc)
    extra = {}
    if a.draft:
        from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
        draft, kind = load_drafter(a.draft)
        validate_drafter_compatibility(model, draft, kind)   # raises rather than silently falling back
        extra = {"draft_model": draft, "draft_kind": kind}
        if a.block: extra["draft_block_size"] = a.block
        print(f"drafter: {a.draft} ({kind})", flush=True)
    warm = tok.apply_chat_template([{"role": "user", "content": "Say hello."}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
    generate(model, proc, warm, max_tokens=32, temperature=0.0, **extra)          # compile kernels before timing
    print("warm-up done, starting quiz", flush=True)
    score, rows, toks, secs, peak = 0, [], 0, 0.0, 0.0
    for q, ans in Q:
        msgs = [{"role": "user", "content": q + " Think briefly, then end with 'ANSWER: <value>'."}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t0 = time.time()
        r = generate(model, proc, prompt, max_tokens=700, temperature=0.0, **extra)
        dt = time.time() - t0
        m = re.findall(r"ANSWER:\s*\**\s*\$?([^\n*]+)", r.text)
        got = m[-1].strip() if m else "(none)"
        ok = ans in got
        score += ok; toks += r.generation_tokens; secs += r.generation_tokens / max(r.generation_tps, 1e-9)
        peak = max(peak, r.peak_memory)
        rows.append({"q": q, "expected": ans, "got": got, "ok": ok, "tokens": r.generation_tokens,
                     "tps": round(r.generation_tps, 2), "wall_s": round(dt, 1), "text": r.text})
        print(f"{'ok ' if ok else 'MISS'} {r.generation_tps:5.1f} tok/s  {got[:40]:40s} (want {ans})", flush=True)
    out = {"tag": a.tag, "model": a.model, "draft": a.draft, "score": score, "of": len(Q),
           "avg_tps": round(toks / secs, 2), "peak_gb": round(peak, 2), "rows": rows}
    if a.draft:
        from mlx_vlm.speculative.utils import format_speculative_stats
        out["draft_stats"] = format_speculative_stats(extra["draft_model"])
    (ROOT / "results" / f"spec_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"SCORE {score}/{len(Q)}  avg {out['avg_tps']} tok/s  peak {out['peak_gb']} GB  {out.get('draft_stats') or ''}")


if __name__ == "__main__":
    main()
