import argparse
import types
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

    # 1. Freeze the FastConformer Encoder (NeMo's native freeze method)
    print("Freezing encoder parameters...")
    model.encoder.freeze()

    # 2. Instantiate and attach the Engram module matching model dimensions
    vocab_size = model.tokenizer.vocab_size
    d_model = model.decoder.hidden_size # Matches NeMo RNN-T decoder hidden dimension
    
    model.engram = DeepSeekEngram(
        table_size=8191, 
        d_model=d_model, 
        vocab_size=vocab_size
    )
    model.engram.to(model.device)
    model.engram.load_idf("idf_weights.pt") # Ensure you generated this using generate_idf.py

    # Enable engram flag on the model instance
    model.engram_enabled = True

    # --- 3. Robust Monkey-Patching for NeMo RNN-T Decoder ---
    original_decoder_forward = model.decoder.forward

    # Define wrapper accepting 'self' as the first argument
    def engram_wrapped_decoder_forward(self, tokens, lengths=None, *args, **kwargs):
        # Call the original decoder computation
        decoder_out = original_decoder_forward(tokens, lengths, *args, **kwargs)
        
        # NeMo decoders can return a tuple or just the hidden states tensor
        if isinstance(decoder_out, tuple):
            hidden_states = decoder_out[0]
            rest = decoder_out[1:]
        else:
            hidden_states = decoder_out
            rest = None

        # Apply DeepSeekEngram if enabled
        if hasattr(model, 'engram') and getattr(model, 'engram_enabled', False):
            hidden_states = model.engram(tokens, hidden_states)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    # Properly bind the wrapped function as an instance method using types.MethodType
    model.decoder.forward = types.MethodType(engram_wrapped_decoder_forward, model.decoder)
    print("Successfully monkey-patched NeMo RNN-T decoder with DeepSeekEngram!")

    # 4. Ensure Decoder, Joint, and Engram networks are Trainable
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.joint.parameters():
        param.requires_grad = True
    for param in model.engram.parameters():
        param.requires_grad = True

    # 5. Setup Data Loaders
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

    # 6. PyTorch Lightning Trainer (Optimized for Colab T4 Free Tier)
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