import sys; sys.path.insert(0,"/mnt/workspace/yama/OpenStereo")
import time, numpy as np, torch
from PIL import Image
import torchvision.transforms.functional as TF
from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder
CKPT="/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"
def im(p):
    t=TF.to_tensor(p); m=torch.tensor([0.485,0.456,0.406]).view(3,1,1); s=torch.tensor([0.229,0.224,0.225]).view(3,1,1); return (t-m)/s
enc=StereoEncoder(ckpt_path=CKPT,update_iters=4,hidden_dim=512,N_stereo_tokens=64,input_size=(256,256)).cuda().eval()
li=im(Image.fromarray(np.random.randint(0,255,(256,256,3),dtype=np.uint8))).unsqueeze(0).cuda()
ri=li.clone()
def bench(fn,n=20,w=3):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
# current wrapper (forces fp32 internally)
t_fp32=bench(lambda: enc(li,ri))
# bf16 autocast (let conv run bf16, BN stays fp32) — patch the wrapper's internal autocast-disable
import starVLA.model.modules.action_model.StereoEncoder as SE
_orig=SE.StereoEncoder.forward
def fwd_amp(self,l,r):
    # bypass the internal 'autocast(enabled=False)' by running the model forward under bf16 autocast
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        return _orig(self,l,r)
SE.StereoEncoder.forward=fwd_amp
t_bf16=bench(lambda: enc(li,ri))
print(f"StereoEncoder @256x256, 4 iters:")
print(f"  fp32 (current wrapper): {t_fp32:.1f} ms")
print(f"  bf16 autocast (BN fp32): {t_bf16:.1f} ms  -> {t_fp32/t_bf16:.2f}x faster")
