# DIS round-2 environment

Recorded in `results/throughput.json` and `results/formal_runs.json`:

- Apple arm64 macOS 14.4, M3 Pro, 18 GB unified memory, 12 CPU cores.
- Project virtual environment: Python 3.12.6, PyTorch 2.14.1, torchvision 0.29.2.
- CUDA unavailable. PyTorch MPS was built, but `mps_available=false` on this run; all formal runs used CPU with six Torch threads.
- ResNet-18 ImageNet weights: `ResNet18_Weights.IMAGENET1K_V1`, downloaded from the official PyTorch URL in the throughput JSON. The SHA256 of the cached weight file was recorded during setup and the weight is not committed.
- The formal 12-run script (`src/train_formal.py`) reads each distinct split image before the first epoch, converts it with PIL, applies bilinear resize and ImageNet mean/std normalization, and keeps the resulting tensors in an in-memory cache. Training then stacks cached tensors into batches of 32; it does not decode from disk once per batch. The 20-epoch diagnostic scripts (`train_budget_diagnostic.py`, `train_learning_rate_endpoints.py`, `train_regularization_diagnostic.py`, and `train_dropout_multiseed_confirmation.py`) reuse the same pre-epoch cache through `train_formal.load_images`.
- The historical short grid (`src/train_resnet.py`) is different: its `VoteDataset.__getitem__` decodes and transforms one image when the sample is fetched, with a `DataLoader` using `num_workers=0`. This behaviour is not generalized to the formal or diagnostic scripts. `src/evaluate_formal.py` and `src/evaluate_extra_cross_sample.py` also pre-decode their selected frames into caches before batched forward passes. `src/benchmark_extra_inference.py` deliberately includes PIL decode, resize, normalization, and CPU forward in its timed loop.
- Archive bytes, extracted JPEG bytes, cached decoded tensors, batch tensors, and peak process memory are distinct quantities. Peak memory was not measured.

The short throughput check used 192 timed images per resolution: 65.72 samples/s median at 128 pixels and 29.44 samples/s median at 224 pixels. These measurements motivated the 128-pixel feasibility protocol and do not establish a resolution-independent scientific conclusion.

The fixed supplemental RGB logistic baseline additionally uses Pillow and
SciPy for decoding and L-BFGS-B fitting. It is a post-hoc analysis, uses no
GPU, and does not distribute its in-memory preprocessing parameters as model
weights.

The frozen original-to-extra probe used the same CPU environment, 128-pixel
preprocessing, six existing large_10000 checkpoints and 1,000 fixed objects.
A 64-image, five-repeat decode/resize/normalise/forward probe measured a median
123.67 images/s. The six checkpoint forward-only records sum to 28.20 seconds.
The surrounding scoring loop, including loading and metric/bootstrap work, took
31.62 seconds; the pre-run estimate including the full small-batch boundary was
48.52 seconds. These timings are inference-cost evidence only and were not used
to change the sample or model.
