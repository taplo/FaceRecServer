import os, torch, zipfile, io, numpy as np
os.environ['HTTP_PROXY'] = 'http://192.168.3.200:8787'
os.environ['HTTPS_PROXY'] = 'http://192.168.3.200:8787'
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from facerecserver.face_detection.detector import FaceDetector
from facerecserver.face_detection.aligner import align_face
from facerecserver.face_recognition.model import SwinTransformer

def test_model(model_path, use_lora, label, fix_norm=False):
    print(f'\n{"="*60}')
    print(f'Testing: {label}')
    print(f'Use LoRA: {use_lora}, Fix norm: {fix_norm}')
    
    model = SwinTransformer(lora_rank=8, lora_scale=1.0, img_size=120, patch_size=6,
                            in_chans=3, num_classes=512, embed_dim=384,
                            depths=(2,18,2), num_heads=(8,16,16),
                            window_size=5, use_lora=use_lora, reso=120)
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    
    # Remove LoRA keys if use_lora=False (they don't exist in base model anyway)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    if fix_norm:
        model.norm.weight.data.fill_(1.0)
        model.norm.bias.data.fill_(0.0)
    
    det = FaceDetector(confidence_threshold=0.9)
    
    def get_face_tensor(path):
        with zipfile.ZipFile(r'D:\faces.zip', 'r', metadata_encoding='gbk') as zf:
            raw = zf.read(path)
        img = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
        bbox, lm, _ = det.detect(img)
        if lm is None: return None
        face = align_face(img, lm, 120)
        t = torch.from_numpy(face).permute(2,0,1).float().unsqueeze(0) / 255.0
        return (t - 0.5) / 0.5
    
    def get_full_emb(t):
        with torch.no_grad():
            emb = model(t, torch.tensor([0.7]).float()).numpy().flatten()
            return emb / np.linalg.norm(emb)
    
    def get_backbone_emb(t):
        with torch.no_grad():
            x = model.patch_embed(t)
            if model.ape:
                x = x + model.absolute_pos_embed
            x = model.pos_drop(x)
            for layer in model.layers:
                x = layer(x, torch.tensor([0.7]).float())
            x = model.norm(x)
            x = x.mean(dim=1)
            return (x / x.norm(dim=1, keepdim=True)).numpy().flatten()
    
    with zipfile.ZipFile(r'D:\faces.zip', 'r', metadata_encoding='gbk') as zf:
        names = sorted([n for n in zf.namelist() if n.lower().endswith('.jpg')])
        groups = {}
        for n in names:
            pid = os.path.splitext(os.path.basename(n))[0].rsplit('-', 1)[0]
            groups.setdefault(pid, []).append(n)
        multi = {k:v for k,v in groups.items() if len(v) >= 2}
        
    results = {'bb': {'same': [], 'cross': []}, 'full': {'same': [], 'cross': []}}
    
    # Test 5 pairs
    tested = 0
    for pid, imgs in multi.items():
        if tested >= 5: break
        if len(imgs) < 2: continue
        
        # Find a cross person
        cross_pid = [k for k in groups if k != pid][0]
        
        t1 = get_face_tensor(imgs[0])
        t2 = get_face_tensor(imgs[1])
        t3 = get_face_tensor(groups[cross_pid][0])
        
        if any(x is None for x in [t1, t2, t3]):
            continue
        
        # Backbone
        e1, e2, e3 = [get_backbone_emb(t) for t in [t1, t2, t3]]
        results['bb']['same'].append(float(np.dot(e1, e2)))
        results['bb']['cross'].append(float(np.dot(e1, e3)))
        
        # Full pipeline
        p1, p2, p3 = [get_full_emb(t) for t in [t1, t2, t3]]
        results['full']['same'].append(float(np.dot(p1, p2)))
        results['full']['cross'].append(float(np.dot(p1, p3)))
        
        tested += 1
    
    if results['bb']['same']:
        bb_same = np.mean(results['bb']['same'])
        bb_cross = np.mean(results['bb']['cross'])
        print(f'Backbone avg pool: same={bb_same:.4f}, cross={bb_cross:.4f}, margin={bb_same-bb_cross:.4f}')
    
    if results['full']['same']:
        full_same = np.mean(results['full']['same'])
        full_cross = np.mean(results['full']['cross'])
        print(f'Full pipeline: same={full_same:.4f}, cross={full_cross:.4f}, margin={full_same-full_cross:.4f}')

# Test 1: Base model, use_lora=False
test_model('models/swin_arcface_webface4m/model.pt', use_lora=False, 
           label='swin_arcface_webface4m (base, no LoRA)')

# Test 2: Base model, use_lora=False, fix norm
test_model('models/swin_arcface_webface4m/model.pt', use_lora=False, 
           label='swin_arcface_webface4m (base, no LoRA, fix norm)', fix_norm=True)

# Test 3: TinyFace model, use_lora=True
test_model('models/swin_arcface_webface4m_tinyface/model.pt', use_lora=True, 
           label='swin_arcface_webface4m_tinyface (LoRA)')

# Test 4: TinyFace model, use_lora=True, fix norm
test_model('models/swin_arcface_webface4m_tinyface/model.pt', use_lora=True, 
           label='swin_arcface_webface4m_tinyface (LoRA, fix norm)', fix_norm=True)
