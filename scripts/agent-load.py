#!/usr/bin/env python3
"""Cline 형태의 부하 재현: 컨텍스트를 키워가는 다중 턴 루프.

각 턴 = (대화 전체 전송) -> (짧은 응답 수신) -> (툴 결과 청크 추가).
컨텍스트가 CTX_CAP 에 닿으면 Cline 의 condense 처럼 앞부분을 버리고 다시 쌓는다.
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 900   # 초
CTX_CAP = 55000            # 토큰, -c 65536 아래로 여유를 둔다
CHARS_PER_TOK = 3.24       # bench-prompt.txt 69,693자 / 21,547토큰

corpus = open("/home/jeano/8G_Qwen3-Coder-30B-A3B/scripts/bench-prompt.txt").read()

def est(msgs):
    return int(sum(len(m["content"]) for m in msgs) / CHARS_PER_TOK)

def post(msgs):
    body = json.dumps({
        "messages": msgs, "max_tokens": 120, "temperature": 1.0,
        "top_p": 0.95, "top_k": 20, "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    u = d.get("usage", {})
    return (d["choices"][0]["message"]["content"], time.time() - t0,
            u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

SYSTEM = ("You are a coding agent working in a large repository. "
          "Answer in at most two sentences. Be concrete.")

start = time.time()
turn = condense = 0
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Read this file and summarize its purpose:\n\n" + corpus}]

while time.time() - start < DURATION:
    turn += 1
    try:
        txt, dt, ptok, ctok = post(msgs)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] turn {turn} 실패: {e}", flush=True)
        time.sleep(5)
        continue
    print(f"[{time.strftime('%H:%M:%S')}] turn {turn:3d} ctx~{est(msgs):6d}tok "
          f"prompt={ptok:6d} gen={ctok:3d} {dt:6.1f}s "
          f"({ctok/dt if dt else 0:5.1f} tok/s) condense={condense}", flush=True)

    msgs.append({"role": "assistant", "content": txt})
    # 툴 결과 주입 (~3k 토큰): 에이전트가 파일을 더 읽는 상황
    off = (turn * 9721) % max(1, len(corpus) - 10000)
    msgs.append({"role": "user",
                 "content": f"[tool_result read_file chunk {turn}]\n" + corpus[off:off + 9700]})

    if est(msgs) > CTX_CAP:
        condense += 1
        msgs = [msgs[0], {"role": "user", "content": "Continue. Prior context condensed.\n\n"
                          + corpus[:20000]}]
        print(f"[{time.strftime('%H:%M:%S')}] --- condense #{condense}, 컨텍스트 재구축", flush=True)

print(f"완료: {turn}턴, condense {condense}회, {time.time()-start:.0f}초", flush=True)
