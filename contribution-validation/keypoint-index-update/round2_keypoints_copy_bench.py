# Run as stdin from the base and patched checkout roots, not as an external module.
import hashlib
import json
import platform
import sys
from pathlib import Path
import torch
import kornia
from benchmarks.common import time_us, warm_up_cpu
from kornia.geometry.keypoints import Keypoints, VideoKeypoints

torch.set_num_threads(1)
print('python',sys.executable,'kornia',kornia.__file__,flush=True)
warm_up_cpu()
rows=[]
for name,shape,strided in [('pose',(17,2),False),('pose_batch',(32,17,2),False),('features',(32,2048,2),False),('video',(2,16,2048,2),False),('strided',(32,2048,2),True),('empty',(0,2),False)]:
    for inplace in (False,True):
        data=(torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32)%1024).reshape(shape)
        if strided: data=data.transpose(-1,-2).contiguous().transpose(-1,-2)
        keypoints=VideoKeypoints.from_tensor(data) if name=='video' else Keypoints(data)
        original=keypoints.data.clone()
        indices=(torch.tensor([0],dtype=torch.long) if name!='empty' else torch.empty(0,dtype=torch.long),)
        values=keypoints.data[:1].clone().fill_(-1)
        def fn(): return keypoints.index_put(indices,values,inplace=inplace)
        fn()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as prof:
            result=fn()
        allocs=[e.self_cpu_memory_usage for e in prof.events() if e.self_cpu_memory_usage>0]
        median,iqr=time_us(fn,min_run_time=.15)
        expected=original.clone();expected[:1]=-1
        assert torch.equal(result.data,expected)
        assert torch.equal(keypoints.data,expected if inplace else original)
        assert result is keypoints if inplace else result is not keypoints
        if name=='video': assert result.to_tensor().shape==shape
        rows.append(dict(case=name,shape=shape,inplace=inplace,median_us=median,iqr_us=iqr,allocation_bytes=sum(allocs),allocation_events=len(allocs),sha256=hashlib.sha256(result.data.contiguous().numpy().tobytes()).hexdigest()))
result={'python':sys.executable,'kornia_path':kornia.__file__,'torch':torch.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),'allocation_metric':'sum of positive self_cpu_memory_usage from torch.profiler; cumulative bytes, not peak','rows':rows}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows,indent=2),flush=True)
