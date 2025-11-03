## Stable Diffusion Model
Short Stable Diffusion implementation from scratch is under `stable_diffusion/`.

Implemented components: UNet denoiser, VAE encoder/decoder, DDPM sampler, attention modules (self / cross / UNet attention), CLIP-related utilities, diffusion pipeline, and supporting blocks (time embeddings, upsampling, etc.). See `stable_diffusion/src/` for code and `stable_diffusion/data/` for tokenizer and checkpoint files.

This is a compact, experimental layout - a CUDA-enabled GPU, the tokenizer data in `stable_diffusion/data/` and <i><a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt">model weights (HuggingFace)</a></i> downloaded in `stable_diffusion/data/` are recommended for practical inference.

## Transformer Model
Short Transformer model implementation from scratch in PyTorch with respect to <i><a href="https://arxiv.org/abs/1706.03762">"Attention Is All You Need"</a></i> paper is under `transformer_model/`.

Dataset: https://huggingface.co/datasets/Helsinki-NLP/opus_books
Model can be trained to perform translation tasks.

See `transformer_model/src/` for training scripts and `transformer_model/src/architecture/` for model code; tokenizer and weights live in `transformer_model/data/`.

