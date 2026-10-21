import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from src.engram import DeepSeekEngram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    print(f"Loading pretrained model: {cfg.model.pretrained_model}")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(cfg.model.pretrained_model)

    # 1. Freeze the FastConformer Encoder
    print("Freezing encoder parameters...")
    model.encoder.freeze()

    # 2. Instantiate and attach the Engram module matching model dimensions
    vocab_size = model.tokenizer.vocab_size
    d_model = model.decoder.hidden_size # Adjust depending on NeMo RNN-T decoder hidden dim
    
    model.engram = DeepSeekEngram(
        table_size=8191, 
        d_model=d_model, 
        vocab_size=vocab_size
    )
    model.engram.to(model.device)
    model.engram.load_idf("idf_weights.pt") # Ensure you generate or provide an IDF file for Hindi tokens

    # 3. Ensure Decoder, Joint, and Engram networks are Trainable
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.joint.parameters():
        param.requires_grad = True
    for param in model.engram.parameters():
        param.requires_grad = True

    # 4. Setup Data Loaders (NeMo handles target transcript tokens internally in batch)
    model.setup_training_data({
        'manifest_filepath': cfg.data.train_manifest,
        'sample_rate': cfg.model.sample_rate,
        'batch_size': cfg.data.batch_size,
        'shuffle': True,
        'num_workers': cfg.data.num_workers,
    })
    
    model.setup_validation_data({
        'manifest_filepath': cfg.data.val_manifest,
        'sample_rate': cfg.model.sample_rate,
        'batch_size': cfg.data.batch_size,
        'shuffle': False,
        'num_workers': cfg.data.num_workers,
    })

    # 5. PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
    )

    print("Starting training with Engram-augmented decoder...")
    trainer.fit(model)

if __name__ == "__main__":
    main()