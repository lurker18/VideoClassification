import torch
from torch import nn
import sys
import pdb

import warnings
warnings.filterwarnings('ignore')

# Constants
MODELS = [
    'mvit',
]

DATASET_CLASSES = {
    'ucf101': 101,
    'kinetics400': 400,
    'kinetics700': 700,
    'hmdb51': 51,
    'ssv2': 174
}

def fetch_model(logger, model_type, train_dataset='ucf101', pretrain_dataset=None, pretrained_pth=None):
    assert model_type in MODELS, f"Passed unsupported model. Should be one of {', '.join(MODELS)}"

    if pretrain_dataset is not None:
        n_finetune_classes = DATASET_CLASSES[train_dataset]
        n_classes = DATASET_CLASSES[pretrain_dataset]
    else:
        n_finetune_classes = None
        n_classes = DATASET_CLASSES[train_dataset]

    if model_type == 'mvit':
        logger.info("Loading MViT 16x4 model.")
        model_name = "mvit_base_16x4"
        model = build_pytorchvideo_model(logger, model_name, True if "torchhub" in pretrained_pth else False)
        if n_classes != n_finetune_classes and "torchhub" in pretrained_pth:
            logger.info(f"Changing num classes {n_classes} to {n_finetune_classes}")
            model.head.proj = nn.Linear(model.head.proj.in_features, n_finetune_classes, bias=True)

        # If there is a designated pre-trained_path, load that. This will be based on a model we trained
        if pretrained_pth and "torchhub" not in pretrained_pth:
            logger.info('loading pretrained model {}'.format(pretrained_pth))
            model.head.proj = nn.Linear(model.head.proj.in_features, n_classes, bias=True)
            pretrain = torch.load(pretrained_pth)
            model.load_state_dict(pretrain['state_dict'])

    else:
        logger.error(f"Unsupported model passed. Exiting...")
        sys.exit()
    return model, model.parameters()


def build_pytorchvideo_model(logger, model_name, pretrained):
    logger.info(f"Loading pytorchvideo model {model_name} with pretrained (K400) set to {pretrained}")
    return torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=pretrained)