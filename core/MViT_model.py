# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:24:01 2024

@author: Nova18
"""
import os
import cv2
import albumentations as A
import pathlib
import imageio
import evaluate
import torch
import numpy as np
from transformers import TrainingArguments, Trainer
import pytorchvideo.data
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoImageProcessor
from pytorchvideo.transforms import (ApplyTransformToKey, Normalize, RandomShortSideScale, UniformTemporalSubsample, )
from torchvision.transforms import (Compose, Lambda, RandomCrop, RandomHorizontalFlip, Resize,)
from IPython.display import Image
import timm

os.environ["WANDB_DISABLED"] = "true"

batch_size = 8 # batch size for training and evaluation
# Define the directory where your folders with video files are located
video_directory = 'D:/HuggingFace/datasets/UCF101/all_data'

# Define the directories where you want to save your train/validation/test sets
train_dir = 'D:/HuggingFace/datasets/UCF101/train'
val_dir = 'D:/HuggingFace/datasets/UCF101/validation'
test_dir = 'D:/HuggingFace/datasets/UCF101/test'

dataset_root_path = "D:/HuggingFace/datasets/UCF101/subset"
dataset_root_path = pathlib.Path(dataset_root_path)

video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.avi"))
    + list(dataset_root_path.glob("val/*/*.avi"))
    + list(dataset_root_path.glob("test/*/*.avi"))
)
all_video_file_paths[:5]

class_labels = sorted({str(path).split("\\")[6] for path in all_video_file_paths})
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes: {list(label2id.keys())}.")

model_name = 'hf-hub:timm/mvitv2_huge_cls.fb_inw21k'
image_processor = AutoImageProcessor.from_pretrained('D:/HuggingFace/models/Timm/mvitv2_huge_cls.fb_inw21k')
model = timm.create_model(model_name,
                          pretrained = True,
                          num_classes = 101)

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = (224, 224)
resize_size = (256, 256)
transform = A.Compose([
    A.Resize(resize_size[1], resize_size[0], always_apply=True),
    A.CenterCrop(crop_size[1], crop_size[0], always_apply=True),
    A.Normalize(
        mean = [0.45, 0.45, 0.45],
        std = [0.225, 0.225, 0.225], 
        always_apply = True
    )
])
resize_to = (256, 256)

num_frames_to_sample = 16
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps


# Training dataset transformations.
train_transform = Compose(
    [
        ApplyTransformToKey(
            key = "video",
            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

# Training dataset.
train_dataset = pytorchvideo.data.Ucf101(
    data_path = os.path.join(dataset_root_path, "train"),
    clip_sampler = pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio = False,
    transform = train_transform,
)

# Validation and evaluation datasets' transformations.
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform = Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

# Validation and evaluation datasets.
val_dataset = pytorchvideo.data.Ucf101(
    data_path = os.path.join(dataset_root_path, "validation"),
    clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio = False,
    transform = val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path = os.path.join(dataset_root_path, "test"),
    clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio = False,
    transform = val_transform,
)

# We can access the `num_videos` argument to know the number of videos we have in the
# dataset.
train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos

"""Let's now take a preprocessed video from the dataset and investigate it."""

sample_video = next(iter(train_dataset))
sample_video.keys()

def investigate_video(sample_video):
    """Utility to investigate the keys present in a single video sample."""
    for k in sample_video:
        if k == "video":
            print(k, sample_video["video"].shape)
        else:
            print(k, sample_video[k])

    print(f"Video label: {id2label[sample_video[k]]}")


investigate_video(sample_video)

"""We can also visualize the preprocessed videos for easier debugging."""
def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.

    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename

def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)

video_tensor = sample_video["video"]
display_gif(video_tensor)

model_ckpt = model_name.split("/")[-1]
new_model_name = f"C:/Users/Nova18/Desktop/VideoMAE/results/{model_ckpt}--finetuned-ucf101-subset"
num_epochs = 8

args = TrainingArguments(
    new_model_name,
    remove_unused_columns = False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    warmup_ratio = 0.1,
    logging_steps = 10,
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy",
    push_to_hub = False,
    report_to = None,
    max_steps = (train_dataset.num_videos // batch_size) * num_epochs,
)

metric = evaluate.load("accuracy")

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions = predictions, references = eval_pred.label_ids)


def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    tokenizer = image_processor,
    compute_metrics = compute_metrics,
    data_collator = collate_fn,
)

train_results = trainer.train()

trainer.evaluate(test_dataset)

trainer.save_model()
test_results = trainer.evaluate(test_dataset)
trainer.log_metrics("test", test_results)
trainer.save_metrics("test", test_results)
trainer.save_state()


trained_model = VideoMAEForVideoClassification.from_pretrained("C:/Users/Nova18/Desktop/VideoMAE/results/videomae-base--finetuned-ucf101-subset/best_model")

sample_test_video = next(iter(test_dataset))
investigate_video(sample_test_video)

def run_inference(model, video):
    """Utility to run inference given a model and test video.

    The video is assumed to be preprocessed already.
    """
    # (num_frames, num_channels, height, width)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)

    inputs = {
        "pixel_values" : perumuted_sample_test_video.unsqueeze(0),
        "labels" : torch.tensor(
            [sample_test_video["label"]]
        ),  # this can be skipped if you don't have labels available.
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits

logits = run_inference(trained_model, sample_test_video["video"])

"""We can now check if the model got the prediction right."""
display_gif(sample_test_video["video"])

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])