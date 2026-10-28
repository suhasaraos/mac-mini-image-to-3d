"""The spec_quiz.py quiz against any OpenAI-compatible server (MTPLX, oMLX), so other runtimes get the same test.

Usage: python api_quiz.py --url http://127.0.0.1:8000/v1 --model NAME --tag NAME
Greedy (temperature 0), thinking off, 700-token limit. Decode speed = tokens after the first one / time after the
first one, so prompt reading is excluded. Writes results/spec_<tag>.json in the same shape as spec_quiz.py.
"""
import argparse, json, pathlib, re, time, urllib.request

ROOT = pathlib.Path(__file__).parent
from quiz import Q          # run with the local-llm .venv python


def ask(url, model, q):
    body = {"model": model, "stream": True, "max_tokens": 700, "temperature": 0.0, "top_p": 1.0,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": q + " Think briefly, then end with 'ANSWER: <value>'."}]}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text, t_first, chunks, usage = "", None, 0, None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    if t_first is None: t_first = time.time()
                    text += piece; chunks += 1
    t_end = time.time()
    n = (usage or {}).get("completion_tokens") or chunks
    return text, n, (n - 1) / max(t_end - (t_first or t0), 1e-9), t_end - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    ask(a.url, a.model, "Say hello.")                                      # warm-up, untimed
    print("warm-up done, starting quiz", flush=True)
    score, rows, toks, secs = 0, [], 0, 0.0
    for q, ans in Q:
        text, n, tps, wall = ask(a.url, a.model, q)
        m = re.findall(r"ANSWER:\s*\**\s*\$?([^\n*]+)", text)
        got = m[-1].strip() if m else "(none)"
        ok = ans in got
        score += ok; toks += n; secs += n / max(tps, 1e-9)
        rows.append({"q": q, "expected": ans, "got": got, "ok": ok, "tokens": n, "tps": round(tps, 2),
                     "wall_s": round(wall, 1), "text": text})
        print(f"{'ok ' if ok else 'MISS'} {tps:5.1f} tok/s  {got[:40]:40s} (want {ans})", flush=True)
    out = {"tag": a.tag, "model": a.model, "url": a.url, "score": score, "of": len(Q),
           "avg_tps": round(toks / secs, 2), "rows": rows}
    (ROOT / "results" / f"spec_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"SCORE {score}/{len(Q)}  avg {out['avg_tps']} tok/s")


if __name__ == "__main__":
    main()
