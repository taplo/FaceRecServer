import os, torch, zipfile, io, numpy as np
os.environ['HTTP_PROXY'] = 'http://192.168.3.200:8787'
os.environ['HTTPS_PROXY'] = 'http://192.168.3.200:8787'
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from facerecserver.face_detection.detector import FaceDetector
from facerecserver.face_detection.aligner import align_face
from facerecserver.face_recognition.model import SwinTransformer

det = FaceDetector(confidence_threshold=0.9)

def load_face(raw_bytes):
    img = np.array(Image.open(io.BytesIO(raw_bytes)).convert('RGB'))
    bbox, lm, _ = det.detect(img)
    if lm is None: return None
    face = align_face(img, lm, 120)
    t = torch.from_numpy(face).permute(2,0,1).float().unsqueeze(0) / 255.0
    return (t - 0.5) / 0.5

# Cache test images
with zipfile.ZipFile(r'D:\faces.zip', 'r', metadata_encoding='gbk') as zf:
    names = sorted([n for n in zf.namelist() if n.lower().endswith('.jpg')])
    groups = {}
    for n in names:
        pid = os.path.splitext(os.path.basename(n))[0].rsplit('-', 1)[0]
        groups.setdefault(pid, []).append(n)
    
    # Collect test pairs
    test_pairs = []
    multi = {k:v for k,v in groups.items() if len(v) >= 2}
    for pid, imgs in list(multi.items())[:5]:
        cp = [k for k in groups if k != pid][0]
        test_pairs.append((zf.read(imgs[0]), zf.read(imgs[1]), zf.read(groups[cp][0])))

# Load faces
face_tensors = []
for raw1, raw2, raw3 in test_pairs:
    t1, t2, t3 = load_face(raw1), load_face(raw2), load_face(raw3)
    if all(x is not None for x in [t1, t2, t3]):
        face_tensors.append((t1, t2, t3))

def test_model(label, path, use_lora, alpha_val):
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
    for t1, t2, t3 in face_tensors:
        e1, e2, e3 = get_emb(t1), get_emb(t2), get_emb(t3)
        same_sims.append(float(np.dot(e1, e2)))
        cross_sims.append(float(np.dot(e1, e3)))
    
    if same_sims:
        print(f'{label}: same={np.mean(same_sims):.4f} cross={np.mean(cross_sims):.4f} margin={np.mean(same_sims)-np.mean(cross_sims):.4f}')

configs = [
    ('WF4M LoRA+alpha=0.7', 'models/swin_arcface_webface4m_tinyface/model.pt', True, 0.7),
    ('WF4M LoRA+alpha=1.0', 'models/swin_arcface_webface4m_tinyface/model.pt', True, 1.0),
    ('WF12M noLoRA+alpha=0.7', 'models/swin_arcface_webface12m/model.pt', False, 0.7),
    ('WF12M noLoRA+alpha=1.0', 'models/swin_arcface_webface12m/model.pt', False, 1.0),
    ('CosFace-WF4M-BRIAR LoRA+0.7', 'models/swin_cosface_webface4m_briar/model.pt', True, 0.7),
    ('CosFace-WF4M-BRIAR LoRA+1.0', 'models/swin_cosface_webface4m_briar/model.pt', True, 1.0),
    ('CosFace-WF12M-BRIAR LoRA+0.7', 'models/swin_cosface_webface12m_briar/model.pt', True, 0.7),
    ('CosFace-WF12M-BRIAR LoRA+1.0', 'models/swin_cosface_webface12m_briar/model.pt', True, 1.0),
]

for label, path, use_lora, alpha in configs:
    test_model(label, path, use_lora, alpha)
