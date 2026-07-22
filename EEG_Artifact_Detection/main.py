import argparse
import os
import shutil
import subprocess
import warnings

from MLPTrainer import MLPTrainer

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    """
    Parse command-line arguments for dataset, training, logging, and execution settings.

    Returns:
        argparse.Namespace: Parsed configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--outputpath', type=str, default='./output')
    parser.add_argument('--snr_db', type=float, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lower_snr', type=float, default=-7)
    parser.add_argument('--higher_snr', type=float, default=6.5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--no_plot', default=False, action='store_false')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--pca', default=False, action='store_true')
    parser.add_argument('--ica', default=False, action='store_true')
    args = parser.parse_args()
    return args


def get_setup_command(mode):
    script_path = os.path.join(PROJECT_DIR, 'env_setup.sh')
    if os.name == 'nt':
        for candidate in ('bash', 'sh'):
            resolved = shutil.which(candidate)
            if resolved:
                linux_path = script_path.replace('C:\\', '/mnt/c/').replace('\\', '/')
                return [resolved, '-lc', f'"{linux_path}" {mode}']
        raise RuntimeError('Bash or sh is required to run env_setup.sh on Windows.')

    bash_path = shutil.which('bash')
    if bash_path:
        return [bash_path, '-lc', f'"{script_path}" {mode}']
    return ['/bin/sh', '-lc', f'"{script_path}" {mode}']


def run_setup_script(mode):
    subprocess.run(get_setup_command(mode), cwd=PROJECT_DIR, check=True)


if __name__ == "__main__":
    config = load_config()
    run_setup_script(config.mode)
    trainer = MLPTrainer(config)
    trainer.run()

