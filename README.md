# Environment

We only work on Linux platform, and most of the codes are written for Python 3.5.
For extracting deep features, please install `PyTorch`.
You may have to install `joblib` as well.

This work is done for EVVE dataset, you can make slight changes to fit other datasets.
If you find any problem in use, pleace contact `yang at nii.ac.jp`.

# Settings

- Put downloaded videos and annotation file of EVVE dataset in a proper place.
- Put downloaded MVLAD descriptors of EVVE dataset in a proper place (Optional).
- Set EVVE's annotation file path `annot_dir` in `settings.py`.
- Set `videos_dir` in `settings.py` to your videos' directory.
- Set `fvecs_dir` in `settings.py` to your MVLAD descriptors's directory (Optional).
- Set `feature_type` in `settings.py` to preferred feature type, such as resnet50.
- Set `fps` according to the feature type you chose, use `15fps` for MVLAD descriptors and `5fps` for deep features.
- Set `frames_dir` to a directory for saving down-sampled frames, it will be created by default.
- Set `infos_dir` to a directory for saving extracted information of videos, containing each feature vector of each frame.
- Set `data_dir` to a directory for saving video descriptors.
- Set `result_dir` to a directory for saving retrieval results.
- Set `alexnet_model_path` and `resnet_model_path` manually, those are pretrained models on ImageNet. Download link: [AlexNet](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth) and [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth).
- Set `max_freq_num` for Fourier series, it represents the highest number of coefficients in frequency domain you want.
- Set `frame_desc_dim` to `1024` by default.
- Set `short_list_length`, `far_list_length` for query expansion, they stand for the number of videos in short/far list.
- Set `epsilon` for query expansion, the unit is frame. It's a parameter for consistency check, we set it to `10` for `5fps` and `50` for `15fps`.
- Set `periods` as adjacent numbers taken from the Fibonacci sequence, according to our paper.

# Frame Extraction

Run `python frames.py`.
You may have to install `ffmpeg`.

# Feature Extraction

Run `python extractor.py`.
Notice that you may need to edit `CUDA_VISIBLE_DEVICES` and `batch_size` according to your devices.

# Create Video Descriptors

Run `python embed.py`.

# Conduct Retrieval Task

Run `python retrieve.py`

# Evaluate Performance

Run `python combine.py`

