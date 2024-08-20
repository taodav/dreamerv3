from pathlib import Path

exp_name = Path(__file__).stem

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': 'dreamerv3/main.py --configs gymnax size6m',
    'args': [
        {
            'task': 'gymnax_pocman',
            'double_critic': 'True',
            'run.steps': int(1e7),
            'seed': [2020 + i for i in range(10)],
        }
    ]
}
