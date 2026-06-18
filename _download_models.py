import os
os.environ['HTTP_PROXY'] = 'http://192.168.3.200:8787'
os.environ['HTTPS_PROXY'] = 'http://192.168.3.200:8787'

from huggingface_hub import hf_hub_download

models_to_download = [
    'swin_arcface_webface12m/model.pt',
    'swin_cosface_webface4m_briar/model.pt',
    'swin_cosface_webface12m_briar/model.pt',
]

for model_file in models_to_download:
    path = hf_hub_download(
        repo_id='kartiknarayan/PETALface',
        filename=model_file,
        local_dir='./models'
    )
    print(f'Downloaded: {path}')
    # Quick check
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    if 'norm.weight' in ckpt:
        w = ckpt['norm.weight']
        print(f'  norm.weight: mean={w.mean():.6f}')
    has_lora = any('trainable_lora' in k for k in ckpt)
    print(f'  Has LoRA: {has_lora}, Total keys: {len(ckpt)}')
