"""Memory check: a long, scripted coding session against the local server, logging system memory as context grows.

Usage: .venv/bin/python coding_session.py [--tag NAME]
Starts serve.sh, feeds real source files turn by turn (each with a coding task), and after every turn prints the
context size alongside memory used / free, swap and macOS memory pressure. Samples every 2 s to results/mem_<tag>.csv.
"""
import argparse, json, os, pathlib, re, subprocess, threading, time, urllib.request

ROOT = pathlib.Path(__file__).parent
TM = pathlib.Path(os.environ.get("TRELLIS_MAC", pathlib.Path.home() / "trellis-mac"))   # source files to feed in
FILES = [TM / "generate.py", TM / "lowpoly.py", TM / "turntable.py", TM / "strip_far_side.py",
         TM / "TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py", TM / "TRELLIS.2/trellis2/pipelines/trellis2_texturing.py",
         TM / "TRELLIS.2/trellis2/renderers/mesh_renderer.py", TM / "TRELLIS.2/app.py",
         TM / "TRELLIS.2/trellis2/renderers/pbr_mesh_renderer.py", TM / "TRELLIS.2/trellis2/modules/sparse/basic.py"]
TASKS = ["Explain what this file does in five short bullet points.",
         "Find one likely bug or fragile spot and show a fix as a unified diff.",
         "Add a --verbose command-line flag that prints timings. Show only the changed lines.",
         "Write two pytest tests for the most important function here.",
         "How does this file connect to the files I showed you earlier? Be specific."]
GB = 1024 ** 3
URL = "http://127.0.0.1:8080/v1"


def mem():
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pg = int(re.search(r"page size of (\d+)", vm)[1])
    v = {k: int(n) * pg for k, n in re.findall(r"^(.+?):\s+(\d+)\.", vm, re.M)}
    app = v["Anonymous pages"] - v["Pages purgeable"]                    # Activity Monitor's "App Memory"
    used = app + v["Pages wired down"] + v["Pages occupied by compressor"]
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    swap = float(re.search(r"used = ([\d.]+)M", sw)[1]) / 1024
    mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    free_pct = int(re.search(r"(\d+)%", mp)[1])
    return used / GB, swap, free_pct


def chat(msgs):
    body = json.dumps({"messages": msgs, "max_tokens": 600, "temperature": 0.7}).encode()
    req = urllib.request.Request(f"{URL}/chat/completions", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tag", default="coding")
    p.add_argument("--server-args", default="", help='extra serve.sh flags, e.g. "--prompt-cache-size 1"')
    a = p.parse_args()
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / GB
    u0, s0, f0 = mem()
    print(f"Before the model loads: {u0:.1f} of {total:.0f} GB used, swap {s0:.1f} GB, pressure {f0}% free", flush=True)
    srv = subprocess.Popen([str(ROOT / "serve.sh"), *a.server_args.split()], stdout=subprocess.DEVNULL,
                           stderr=open(ROOT / "results" / f"server_{a.tag}.log", "w"))
    if a.server_args: print(f"Server flags: {a.server_args}", flush=True)
    log, stop, peak = [], threading.Event(), [0.0, 0.0, 100]
    t0 = time.time()
    def sample():
        while not stop.is_set():
            u, s, f = mem(); log.append((round(time.time() - t0, 1), round(u, 2), round(s, 2), f))
            peak[0], peak[1], peak[2] = max(peak[0], u), max(peak[1], s), min(peak[2], f)
            time.sleep(2)
    threading.Thread(target=sample, daemon=True).start()
    try:
        for _ in range(300):
            try: urllib.request.urlopen(f"{URL}/models", timeout=2); break
            except Exception: time.sleep(1)
        msgs = [{"role": "system", "content": "You are a concise senior Python engineer helping me in my editor."}]
        for i, f in enumerate(FILES, 1):
            task = TASKS[(i - 1) % len(TASKS)]
            msgs.append({"role": "user", "content": f"Here is `{f.name}`:\n```python\n{f.read_text()}\n```\n{task}"})
            t = time.time(); r = chat(msgs); dt = time.time() - t
            msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
            ctx = r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
            u, s, fr = mem()
            print(f"Turn {i:2d}  {f.name:28.28s} context {ctx:6,d} tok  {dt:5.0f}s | "
                  f"used {u:4.1f} GB  free {total - u:4.1f} GB  swap {s:4.1f} GB  pressure {fr}% free", flush=True)
    except Exception as e:
        print(f"Session stopped: {type(e).__name__}: {e} (server exit code {srv.poll()})", flush=True)
    finally:
        stop.set(); srv.terminate(); srv.wait()
    print(f"PEAK: {peak[0]:.1f} of {total:.0f} GB used ({total - peak[0]:.1f} GB left), "
          f"swap {s0:.1f} -> {peak[1]:.1f} GB, lowest pressure reading {peak[2]}% free", flush=True)
    out = ROOT / "results" / f"mem_{a.tag}.csv"
    out.write_text("t_s,used_gb,swap_gb,free_pct\n" + "\n".join(",".join(map(str, row)) for row in log) + "\n")


if __name__ == "__main__":
    main()
