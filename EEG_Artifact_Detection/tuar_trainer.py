"""TUAR-specific trainer helpers for EEG_Artifact_Detection."""

import argparse


def make_tuar_trainer_class():
    """Return a trainer subclass that skips MLPTrainer data combiner initialization."""
    from MLPTrainer import MLPTrainer

    class TUARTrainer(MLPTrainer):
        def _init_data_combiner(self):
            # data/train, data/val, data/test/tuar are pre-built from TUAR by this
            # script -- MLPTrainer's default behaviour would overwrite them with
            # synthetic EEGDenoiseNet mixtures, so skip it entirely.
            pass

    return TUARTrainer


def run_training(datapath, args):
    TUARTrainer = make_tuar_trainer_class()
    config = argparse.Namespace(
        datapath=datapath,
        outputpath=args.outputpath,
        snr_db=None,
        test_size=0.0,
        val_size=0.0,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lower_snr=-7.0,
        higher_snr=6.5,
        patience=args.patience,
        log_file=args.log_file,
        log_level=args.log_level,
        no_plot=True,
        save_path=args.save_path,
        mode="train",
        model=args.model,
        pca=args.pca,
        ica=False,
    )
    trainer = TUARTrainer(config)
    trainer.run()
