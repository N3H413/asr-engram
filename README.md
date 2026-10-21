# Hindi Nemotron ASR Adapter

Fine-tunes the decoder of NVIDIA's streaming RNN-T model on Hindi speech data while freezing the encoder to preserve memory on Google Colab free tier.

## Quickstart in Colab

1. **Clone Repo & Install Dependencies:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/hindi-nemotron-adapter.git
   cd hindi-nemotron-adapter
   pip install -r requirements.txt