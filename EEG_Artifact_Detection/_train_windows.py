import sys
sys.argv = [
    "main.py",
    "--datapath", "data",
    "--outputpath", "output",
    "--save_path", "checkpoints",
    "--model", "MLP",
    "--pca",
    "--mode", "train",
    "--num_epochs", "100",
    "--batch_size", "32",
    "--learning_rate", "0.001",
    "--patience", "20",
    "--test_size", "0.2",
    "--val_size", "0.2",
    "--log_file", "train_log.txt",
    "--log_level", "INFO",
    "--no_plot",
]
from MLPTrainer import MLPTrainer
trainer = MLPTrainer(load_config())  if False else None
# minimal config object the trainer expects (argparse Namespace)
import argparse
def load_config():
    """
    Parse command-line options for the training workflow.
    
    Unknown command-line arguments are ignored.
    
    Returns:
        argparse.Namespace: Parsed training configuration options.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--datapath', type=str)
    p.add_argument('--outputpath', type=str)
    p.add_argument('--snr_db', type=float, default=None)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--val_size', type=float, default=0.2)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=0.001)
    p.add_argument('--lower_snr', type=float, default=-7)
    p.add_argument('--higher_snr', type=float, default=6.5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--log_file', type=str, default='log.txt')
    p.add_argument('--log_level', type=str, default='INFO')
    p.add_argument('--no_plot', default=False, action='store_false')
    p.add_argument('--save_path', type=str)
    p.add_argument('--mode', type=str, default='train')
    p.add_argument('--model', type=str, default='MLP')
    p.add_argument('--pca', default=False, action='store_true')
    p.add_argument('--ica', default=False, action='store_true')
    args, _ = p.parse_known_args()
    return args

trainer = MLPTrainer(load_config())
trainer.run()
