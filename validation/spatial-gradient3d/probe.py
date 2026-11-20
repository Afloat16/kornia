import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import kornia
from benchmarks.common import time_us, warm_up_cpu

root = Path.cwd().resolve()
assert Path(kornia.__file__).resolve() == root / 'kornia/__init__.py'
torch.set_num_threads(1)
mode, output_dir = sys.argv[1:]
outdir = Path(output_dir)
outdir.mkdir(parents=True, exist_ok=True)
fn = kornia.filters.spatial_gradient3d
metadata = {'source': str(root), 'sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), 'sobel_sha256': hashlib.sha256((root/'kornia/filters/sobel.py').read_bytes()).hexdigest(), 'interpreter':sys.executable,'torch':torch.__version__,'threads':torch.get_num_threads()}
rows = []
if mode == 'checks':
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for shape in ((1,1,1,3,4),(2,3,4,5,6),(0,2,3,4,5),(1,1,2,1,3)):
            for layout in ('contiguous','transposed','expanded'):
                torch.manual_seed(17)
                image = torch.rand(shape,dtype=dtype)
                if layout == 'transposed': image=image.transpose(-1,-2)
                if layout == 'expanded': image=image[:, :1].expand(-1,3,-1,-1,-1)
                image.requires_grad_()
                old=image.detach().clone()
                version=image._version
                result=fn(image)
                weights=torch.arange(result.numel(),dtype=torch.float64).reshape(result.shape).remainder(7).to(dtype)
                grad=torch.autograd.grad((result*weights).sum(),image)[0]
                assert torch.equal(image,old) and image._version==version
                torch.save({'output':result.detach(), 'gradient':grad, 'input':image.detach()},outdir/f'{dtype}-{shape}-{layout}.pt')
                rows.append({'dtype':str(dtype),'shape':shape,'layout':layout})
    x=torch.rand(2,3,4,5,6,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(fn,(x,),fast_mode=True)
    try:
        scripted = torch.jit.script(fn)
        torch.testing.assert_close(scripted(x),fn(x),rtol=0,atol=0)
        rows.append({'transform':'script'})
    except Exception as error:
        rows.append({'transform':'script', 'error':str(error).splitlines()[0:8]})
    for name, transformed in [('functionalize', torch.func.functionalize(fn))]:
        torch.testing.assert_close(transformed(x),fn(x),rtol=0,atol=0)
        rows.append({'transform':name})
    batched=torch.rand(2,1,2,3,4,5)
    try:
        torch.testing.assert_close(torch.vmap(fn)(batched),torch.stack([fn(v) for v in batched]),rtol=0,atol=0)
        rows.append({'transform':'vmap'})
    except Exception as error:
        rows.append({'transform':'vmap','error':str(error).splitlines()[0:8]})
    primal=torch.rand(1,2,3,4,5,dtype=torch.float64)
    tangent=torch.rand_like(primal)
    jvp=torch.func.jvp(fn,(primal,),(tangent,))[1]
    torch.testing.assert_close(jvp,fn(tangent),rtol=0,atol=0)
    rows.append({'transform':'jvp'})
elif mode == 'memory':
    for shape in ((1,1,64,64,64),(1,1,128,128,128),(2,3,64,64,64)):
        x=torch.rand(shape)
        fn(x)
        gc.collect()
        trace=outdir/f'trace-{shape}.json'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True) as profile:
            result=fn(x)
        profile.export_chrome_trace(str(trace))
        events=json.loads(trace.read_text())['traceEvents']
        events=[x for x in events if x.get('name')=='[memory]']
        net=peak=positive=count=0
        for e in sorted(events,key=lambda e:e['ts']):
            n=e['args']['Bytes'];net+=n;peak=max(peak,net)
            if n>0:positive+=n;count+=1
        rows.append({'shape':shape,'positive_bytes':positive,'peak_net_bytes':peak,'positive_count':count,'output_bytes':result.numel()*result.element_size()})
        del result
elif mode=='timing':
    warm_up_cpu()
    for shape in ((1,1,8,8,8),(1,1,64,64,64),(1,1,128,128,128),(2,3,64,64,64)):
        torch.manual_seed(17)
        x=torch.rand(shape)
        median,iqr=time_us(lambda:fn(x),min_run_time=0.5)
        rows.append({'shape':shape,'median_us':median,'iqr_us':iqr})
else:
    raise ValueError(mode)
(outdir/f'{mode}.json').write_text(json.dumps({'metadata':metadata,'results':rows},indent=2))
print(json.dumps({'metadata':metadata,'results':rows}))
