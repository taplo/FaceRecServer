import requests, numpy as np, sys, time
from pathlib import Path

SERVER = "http://127.0.0.1:8012"
URL = SERVER + "/api/v1/embedding"

for i in range(30):
    try:
        r = requests.get(SERVER + "/openapi.json", timeout=2)
        if r.status_code == 200:
            print("Server ready")
            break
    except:
        pass
    time.sleep(1)
else:
    print("Server not ready")
    sys.exit(1)

test_dir = Path("test/01")
all_jpg = sorted(test_dir.glob("*.jpg"))
files = [str(f) for f in all_jpg[:2]]
print(f"Testing: {[Path(f).name for f in files]}")

embs = []
for f in files:
    with open(f, "rb") as img:
        r = requests.post(URL, files={"file": img}, timeout=120)
        data = r.json()
        name = Path(f).name
        print(f"{name}: code={data['code']}, msg={data['message']}")
        if data["code"] == 0:
            embs.append(np.array(data["data"]["embedding"]))

if len(embs) == 2:
    sim = np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
    print(f"\nDimension: {len(embs[0])}")
    print(f"Cosine similarity: {sim:.6f}")
else:
    print(f"Got {len(embs)} valid embeddings")
