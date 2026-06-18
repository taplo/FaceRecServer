import os, torch, zipfile, io, numpy as np
os.environ['HTTP_PROXY'] = 'http://192.168.3.200:8787'
os.environ['HTTPS_PROXY'] = 'http://192.168.3.200:8787'
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from facerecserver.face_detection.detector import FaceDetector
from facerecserver.face_detection.aligner import align_face
from facerecserver.face_recognition.model import SwinTransformer

# Pre-load all images from zip
det = FaceDetector(confidence_threshold=0.9)

def load_face(raw_bytes):
    img = np.array(Image.open(io.BytesIO(raw_bytes)).convert('RGB'))
    bbox, lm, _ = det.detect(img)
    if lm is None: return None
    face = align_face(img, lm, 120)
    t = torch.from_numpy(face).permute(2,0,1).float().unsqueeze(0) / 255.0
    return (t - 0.5) / 0.5

# Get image data from zip
with zipfile.ZipFile(r'D:\faces.zip', 'r', metadata_encoding='gbk') as zf:
    names = sorted([n for n in zf.namelist() if n.lower().endswith('.jpg')])
    groups = {}
    for n in names:
        pid = os.path.splitext(os.path.basename(n))[0].rsplit('-', 1)[0]
        groups.setdefault(pid, []).append((n, zf.read(n)))

multi = {k:v for k,v in groups.items() if len(v) >= 2}
face_cache = {}
for pid, items in list(multi.items())[:6]:
    for name, raw in items[:2]:
        face_cache[(pid, name)] = load_face(raw)
    # one cross person
    cp = [k for k in groups if k != pid][0]
    cp_name, cp_raw = groups[cp][0]
    face_cache[('cross', cp_name)] = load_face(cp_raw)

def test_config(label, path, use_lora, alpha_val):
    model = SwinTransformer(lora_rank=8, lora_scale=1.0, img_size=120, patch_size=6,
                            in_chans=3, num_classes=512, embed_dim=384,
                            depths=(2,18,2), num_heads=(8,16,16),
                            window_size=5, use_lora=use_lora, reso=120)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    alpha_t = torch.tensor([alpha_val]).float()
    
    def get_emb(t):
        with torch.no_grad():
            emb = model(t, alpha_t).numpy().flatten()
            return emb / np.linalg.norm(emb)
    
    same_sims, cross_sims = [], []
    tested = 0
    for pid, items in multi.items():
        if tested >= 5: break
        t1 = face_cache.get((pid, items[0][0]))
        t2 = face_cache.get((pid, items[1][0]))
        cp_name = groups[[k for k in groups if k != pid][0]][0][0]
        t3 = face_cache.get(('cross', cp_name))
        if any(x is None for x in [t1, t2, t3]): continue
        e1, e2, e3 = get_emb(t1), get_emb(t2), get_emb(t3)
        same_sims.append(float(np.dot(e1, e2)))
        cross_sims.append(float(np.dot(e1, e3)))
        tested += 1
    
    if same_sims:
        print(f'{label}: same={np.mean(same_sims):.4f} cross={np.mean(cross_sims):.4f} margin={np.mean(same_sims)-np.mean(cross_sims):.4f}')

test_config('TinyFace LoRA, alpha=0.7', 'models/swin_arcface_webface4m_tinyface/model.pt', True, 0.7)
test_config('TinyFace LoRA, alpha=1.0', 'models/swin_arcface_webface4m_tinyface/model.pt', True, 1.0)
test_config('TinyFace LoRA, alpha=0.0', 'models/swin_arcface_webface4m_tinyface/model.pt', True, 0.0)
test_config('Base no LoRA, alpha=0.7', 'models/swin_arcface_webface4m/model.pt', False, 0.7)
test_config('Base no LoRA, alpha=1.0', 'models/swin_arcface_webface4m/model.pt', False, 1.0)
test_config('Base no LoRA, alpha=0.0', 'models/swin_arcface_webface4m/model.pt', False, 0.0)
