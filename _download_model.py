import os
os.environ['HTTP_PROXY'] = 'http://192.168.3.200:8787'
os.environ['HTTPS_PROXY'] = 'http://192.168.3.200:8787'

from huggingface_hub import hf_hub_download
import torch

print('Downloading swin_arcface_webface4m/model.pt...')
path = hf_hub_download(repo_id='kartiknarayan/PETALface', filename='swin_arcface_webface4m/model.pt', local_dir='./models')
print(f'Downloaded to: {path}')

ckpt = torch.load(path, map_location='cpu', weights_only=True)
if 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
print(f'Keys ({len(ckpt)}):')
for k in sorted(ckpt.keys()):
    v = ckpt[k]
    print(f'  {k}: {list(v.shape)}')

if 'norm.weight' in ckpt:
    w = ckpt['norm.weight']
    print(f'\nnorm.weight: mean={w.mean().item():.6f}, std={w.std().item():.6f}')
if 'norm.bias' in ckpt:
    b = ckpt['norm.bias']
    print(f'norm.bias: mean={b.mean().item():.6f}, std={b.std().item():.6f}')
