pip3 install numpy \
pandas \
matplotlib \
scipy \
scikit-learn \
seaborn \
tqdm \
requests \
pyyaml \
pillow \
packaging \
pyarrow \
datasets \
transformers \
tokenizers \
safetensors \
huggingface_hub

git clone https://github.com/michal1010/scalable-systems-ravan.git

cd scalable-systems-ravan

T5_GRAD_ACCUM_STEPS=1 T5_GRAD_CHECKPOINT=0 T5_USE_AMP=1 T5_BATCH_SIZE=32 T5_ROUNDS=100 MAX_PARALLEL=1 bash jobs/gpu_vm/sweep_t5_minimal.sh

T5_GRAD_ACCUM_STEPS=1 T5_GRAD_CHECKPOINT=0 T5_USE_AMP=1 T5_BATCH_SIZE=32 T5_ROUNDS=100 MAX_PARALLEL=1 T5_FEDIT_RANK=32 T5_RAVAN_HEADS=12 T5_RAVAN_RANK=64 bash jobs/gpu_vm/sweep_t5_minimal.sh