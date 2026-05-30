"""Check progress of E1-E7 5000 steps experiments."""
from pathlib import Path

experiments = [
    'E1-Transformer', 'E1-SwiGLU', 'E1-Longer', 'E1-Longer-Causal',
    'E2-RankMixer-1-2', 'E2-RankMixer-5-2', 'E2-RankMixer-5-10',
    'E3-Full', 'E3-FFNOnly', 'E3-None',
    'E4-NoRoPE', 'E4-RoPE-10k',
    'E5-d32', 'E5-d64', 'E5-d128', 'E5-d256',
    'E6-b1', 'E6-b2', 'E6-b4', 'E6-b8',
    'E7-q1'
]

completed = []
for exp in experiments:
    run_dir = Path(f'outputs/e1_e7_5000/{exp}')
    ckpt_dirs = list(run_dir.glob('global_step*'))
    if ckpt_dirs:
        ckpt_dirs.sort(key=lambda p: int(p.name.split('.')[0].replace('global_step', '')))
        last = ckpt_dirs[-1]
        auc = None
        if 'AUC=' in last.name:
            try:
                auc = float(last.name.split('AUC=')[1])
            except:
                pass
        last_step = int(last.name.split('.')[0].replace('global_step', ''))
        completed.append({'name': exp, 'auc': auc, 'step': last_step})
        print(f'{exp}: Step={last_step}, AUC={auc}')
    else:
        print(f'{exp}: NOT STARTED')

print(f'\nCompleted (step>=5000): {sum(1 for c in completed if c["step"] >= 5000)}/{len(experiments)}')
