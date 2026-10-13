import json
import os
from pathlib import Path
import subprocess
import time

project = Path('/home/justiee/hse/ai360-transformer')
run = project / 'work/test_inference_20261013_7cef162'
results_dir = run / 'results'
results_dir.mkdir(exist_ok=True)
manifest = [item for item in json.loads((run / 'manifest.json').read_text()) if Path(item['checkpoint']).name == 'model_best.pth']
(results_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
results = []
for item in manifest:
    name = Path(item['checkpoint']).parent.name
    model = 'gpt' + name.removeprefix('tiny_stories')
    runtime = run / ('alibi' if name == 'tiny_stories_alibi' else 'standard')
    command = [
        str(project / '.venv/bin/python'), '-u', str(run / 'capture_results.py'),
        f'model={model}', f'inferencer.from_pretrained={item["remote_checkpoint"]}',
        'dataloader.batch_size=128', 'inferencer.device=cuda',
        f'tokenizer.vocab_path={run / "tokenizer.json"}',
        f'hydra.run.dir={results_dir / name / "hydra"}',
    ]
    if name == 'tiny_stories_alibi':
        command.append('model.decoder.use_flash_attn=false')
    case_dir = results_dir / name
    case_dir.mkdir(exist_ok=True)
    (case_dir / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    environment = os.environ.copy()
    environment['INFERENCE_RESULT_PATH'] = str(case_dir / 'metrics.json')
    environment['HYDRA_FULL_ERROR'] = '1'
    started = time.monotonic()
    print('START', name, flush=True)
    with (case_dir / 'inference.log').open('w') as output:
        completed = subprocess.run(command, cwd=runtime, env=environment, stdout=output, stderr=subprocess.STDOUT)
    result = {
        **item, 'model': model, 'batch_size': 128, 'device': 'cuda',
        'seconds': time.monotonic() - started,
        'status': 'ok' if completed.returncode == 0 else 'failed',
        'exit_code': completed.returncode,
    }
    if completed.returncode == 0:
        result.update(json.loads((case_dir / 'metrics.json').read_text()))
        print('DONE', name, result['metrics'], round(result['seconds'], 2), flush=True)
    else:
        result['error'] = (case_dir / 'inference.log').read_text()[-4000:]
        print('FAILED', name, result['error'], flush=True)
    results.append(result)
    (results_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    summary = '\n'.join(
        f"{row['model']}: test_loss={row['metrics']['test']['loss']:.8f}, seconds={row['seconds']:.2f}"
        if row['status'] == 'ok' else f"{row['model']}: FAILED"
        for row in results
    )
    (results_dir / 'results.txt').write_text(summary + '\n')
print('FINISHED', len(results), 'models', flush=True)
