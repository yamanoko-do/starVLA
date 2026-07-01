"""Profile StereoEncoder to find why a 'lightweight' stereo net is as slow as the 4B VLM.
Breaks the forward into sections with CUDA-sync timing, then runs torch.profiler for op-level detail.
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/mnt/workspace/yama/starVLA")
os.chdir("/mnt/workspace/yama/starVLA")
import sys
sys.path.insert(0, "/mnt/workspace/yama/OpenStereo")
import torch
import sys
sys.path.insert(0, "/mnt/workspace/yama/OpenStereo")
import torch.nn.functional as F
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.models.wavestereo.geometry import Geo_Encoding_Volume
from stereo.modeling.models.wavestereo.utils import InputPadder
from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder

CKPT = "/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"
enc = StereoEncoder(ckpt_path=CKPT, hidden_dim=512, N_stereo_tokens=64, input_size=(256,256)).cuda()
n_params = sum(p.numel() for p in enc.parameters())/1e6
print(f"StereoEncoder params: {n_params:.1f}M total", flush=True)

B = 4  # match B*T in training (train_seq_len=4)
left = torch.randn(B, 3, 256, 256, device="cuda")
right = torch.randn(B, 3, 256, 256, device="cuda")

# ---- full forward timing ----
for _ in range(3):  # warmup
    with torch.no_grad():
        _ = enc(left, right)
torch.cuda.synchronize()
t0=time.time()
N=10
for _ in range(N):
    with torch.no_grad():
        _ = enc(left, right)
torch.cuda.synchronize()
print(f"\n[full forward] B={B}: {(time.time()-t0)/N*1000:.1f} ms/call", flush=True)

# ---- section-by-section timing (replicate forward with sync points) ----
model = enc._get_model()
def section_forward():
    with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
        l = left.float(); r = right.float()
        padder = InputPadder(l.shape, divis_by=32); l, r = padder.pad(l, r)
        torch.cuda.synchronize(); t=time.time()
        fl = model.backbone(l); fr = model.backbone(r)
        torch.cuda.synchronize(); t_back=time.time()-t
        t=time.time()
        cv = correlation_volume(fl[0], fr[0], model.max_disp//4)
        torch.cuda.synchronize(); t_corr=time.time()-t
        t=time.time()
        enc_vol = model.cost_agg(cv, fl)
        torch.cuda.synchronize(); t_agg=time.time()-t
        hidden = model.hnet(fl[0]); net = torch.tanh(hidden)
        context = list(model.context_zqr_conv(fl[0]).split(split_size=model.hidden_dim, dim=1))
        unsq = enc_vol[0].reshape(B, -1, enc_vol[0].size(1), enc_vol[0].size(2), enc_vol[0].size(3))
        prob = F.softmax(enc_vol[0], dim=1); init_disp = disparity_regression(prob, model.max_disp//4)
        geo_fn = Geo_Encoding_Volume(unsq.float(), radius=model.corr_radius, num_levels=model.corr_levels)
        disp = init_disp
        t=time.time()
        for itr in range(enc.update_iters):
            disp = disp.detach(); corr = geo_fn(disp)
            net, delta_disp, _mf = model.update_block(net, context, feat_left=fl[0], feat_right=fr[0], disp=disp, corr=corr, itr=itr)
            disp = disp + delta_disp
        torch.cuda.synchronize(); t_upd=time.time()-t
        t=time.time()
        fused = enc.fuse(torch.cat([enc_vol[0], net], dim=1))
        pooled = F.adaptive_avg_pool2d(fused, (enc.grid, enc.grid))
        torch.cuda.synchronize(); t_fuse=time.time()-t
    return t_back, t_corr, t_agg, t_upd, t_fuse

# warmup
for _ in range(3): section_forward()
secs={"backbone(×2)":0,"correlation_volume":0,"cost_agg":0,"update(4 iters)":0,"fuse":0}
M=10
for _ in range(M):
    tb,tc,ta,tu,tf = section_forward()
    secs["backbone(×2)"]+=tb; secs["correlation_volume"]+=tc; secs["cost_agg"]+=ta; secs["update(4 iters)"]+=tu; secs["fuse"]+=tf
print(f"\n[section breakdown] B={B}, avg over {M}:")
tot=sum(secs.values())
for k,v in secs.items():
    print(f"  {k:20s}: {v/M*1000:7.2f} ms  ({v/tot*100:5.1f}%)")
print(f"  {'TOTAL':20s}: {tot/M*1000:7.2f} ms")

# ---- op-level profiler (top CUDA ops) ----
print("\n[op-level profiler top 12 by CUDA time] B=4:", flush=True)
from torch.profiler import profile, ProfilerActivity
for _ in range(3):
    with torch.no_grad(): _ = enc(left, right)
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    with torch.no_grad():
        for _ in range(3): _ = enc(left, right)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=40))
