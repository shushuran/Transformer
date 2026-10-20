import urllib.request, ssl, time

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
t0 = time.time()
try:
    req = urllib.request.Request(URL, headers={"User-Agent": "tlab/0.1"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
    txt = raw.decode("utf-8")
    print("OK", len(raw), "bytes", f"{time.time()-t0:.1f}s")
    print("chars:", len(txt), "| vocab:", len(set(txt)))
    print("head:", repr(txt[:80]))
except Exception as e:
    print("FAIL", type(e).__name__, e)
