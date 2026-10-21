import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    print(f"Loading pretrained model: {cfg.model.pretrained_model}")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(cfg.model.pretrained_model)

    # 1. Freeze the FastConformer Encoder (Saves significant VRAM on Colab T4)
    print("Freezing encoder parameters...")
    model.encoder.freeze()

    # 2. Keep Decoder and Joint networks trainable
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.joint.parameters():
        param.requires_grad = True

    # 3. Setup Training & Validation Datasets
    print("Setting up data loaders...")
    model.setup_training_data({
        'manifest_filepath': cfg.data.train_manifest,
        'sample_rate': cfg.model.sample_rate,
        'batch_size': cfg.data.batch_size,
        'shuffle': True,
        'num_workers': cfg.data.num_workers,
        'pin_memory': True,
    })

    model.setup_validation_data({
        'manifest_filepath': cfg.data.val_manifest,
        'sample_rate': cfg.model.sample_rate,
        'batch_size': cfg.data.batch_size,
        'shuffle': False,
        'num_workers': cfg.data.num_workers,
    })

    # 4. Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=10
    )

    print("Starting decoder fine-tuning...")
    trainer.fit(model)

if __name__ == "__main__":
    main()