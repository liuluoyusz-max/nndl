"""AdaIN style transfer for the Photo-only augmentation experiment.

The fixed entry point for member 3 and member 5 is stylize(content, style, alpha).
Feature-level steps are also exposed so member 3 can apply a spatial strength map
before decoding:

    model = AdaINModel()
    content_feat = model.encode(content_tensor)
    style_feat = model.encode(style_tensor)
    mixed = model.mix(content_feat, style_feat, alpha)
    image = model.decode(mixed)

alpha may be a scalar in [0, 1] or a spatial map. A map is resized to the
relu4_1 feature resolution and applied as alpha * AdaIN(content, style)
+ (1 - alpha) * content. Weights are the public AdaIN encoder and decoder.
"""
import shutil
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF

WEIGHT_URLS = {
    'decoder.pth': 'https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth',
    'vgg_normalised.pth': 'https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth',
}
ENCODER_DEPTH = 31


def calc_mean_std(feat, eps=1e-5):
    """Channel mean and standard deviation of a BCHW feature map."""
    if feat.ndim != 4:
        raise ValueError(f'Expected a BCHW feature map, got shape {tuple(feat.shape)}')
    reduced = feat.view(feat.size(0), feat.size(1), -1)
    std = (reduced.var(dim=2) + eps).sqrt().view(feat.size(0), feat.size(1), 1, 1)
    mean = reduced.mean(dim=2).view(feat.size(0), feat.size(1), 1, 1)
    return mean, std


def adaptive_instance_normalization(content_feat, style_feat):
    """Match the channel-wise mean and variance of content features to style features."""
    if content_feat.shape[:2] != style_feat.shape[:2]:
        raise ValueError(
            f'Channel mismatch: content {tuple(content_feat.shape)} vs style {tuple(style_feat.shape)}'
        )
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


def build_decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


def build_vgg():
    """Normalized VGG-19. Inference keeps layers through relu4_1."""
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
    )


def ensure_weights(weights_dir):
    """Download decoder.pth and vgg_normalised.pth when they are missing."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in WEIGHT_URLS.items():
        destination = weights_dir / name
        if destination.is_file() and destination.stat().st_size > 0:
            paths[name] = destination
            continue
        temporary = destination.with_suffix(destination.suffix + '.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as handle:
                shutil.copyfileobj(response, handle)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        temporary.replace(destination)
        paths[name] = destination
    return paths


def _load_state_dict(path):
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError(f'Weight file does not contain a state dict: {path}')
    if any(str(key).startswith('module.') for key in state):
        state = {str(key).removeprefix('module.'): value for key, value in state.items()}
    return state


def _freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class AdaINModel:
    """Encoder, AdaIN mixing, and decoder. Images are RGB float tensors in [0, 1]."""

    def __init__(self, weights_dir=None, device=None, content_size=512, style_size=512):
        root = Path(__file__).resolve().parent
        weights_dir = Path(weights_dir) if weights_dir is not None else root / 'models'
        weights = ensure_weights(weights_dir)
        if content_size < 0 or style_size < 0:
            raise ValueError('content_size and style_size must be non-negative')
        self.content_size = content_size
        self.style_size = style_size
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        encoder = build_vgg()
        encoder.load_state_dict(_load_state_dict(weights['vgg_normalised.pth']))
        self.encoder = _freeze(nn.Sequential(*list(encoder.children())[:ENCODER_DEPTH])).to(self.device)
        self.decoder = build_decoder()
        self.decoder.load_state_dict(_load_state_dict(weights['decoder.pth']))
        self.decoder = _freeze(self.decoder).to(self.device)

    def prepare_image(self, image, size):
        """Return a BCHW tensor in [0, 1]. Paths and PIL images are resized; tensors are not."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                picture = source.convert('RGB')
                picture.load()
        elif isinstance(image, Image.Image):
            picture = image.convert('RGB')
        elif torch.is_tensor(image):
            tensor = image.detach().float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.size(1) != 3:
                raise ValueError(f'Image tensor must have shape CHW or BCHW, got {tuple(image.shape)}')
            if float(tensor.max()) > 1.5:
                raise ValueError('Image tensor values must be in [0, 1].')
            return tensor
        else:
            raise TypeError('content and style must be a path, PIL image, or tensor')
        if size:
            picture = TF.resize(picture, size, interpolation=TF.InterpolationMode.BILINEAR)
        return TF.to_tensor(picture).unsqueeze(0)

    def encode(self, image):
        """relu4_1 features. image is a BCHW tensor in [0, 1]."""
        tensor = self.prepare_image(image, size=0).to(self.device)
        with torch.no_grad():
            return self.encoder(tensor)

    def decode(self, features):
        """Decode features to a BCHW RGB tensor in [0, 1] on CPU."""
        with torch.no_grad():
            image = self.decoder(features.to(self.device))
        return image.clamp(0, 1).cpu()

    def mix(self, content_feat, style_feat, alpha=1.0):
        """AdaIN, then blend with alpha. alpha is a scalar or a spatial map."""
        stylized = adaptive_instance_normalization(content_feat, style_feat)
        alpha_map = _alpha_tensor(alpha, stylized)
        return alpha_map * stylized + (1.0 - alpha_map) * content_feat

    def stylize_tensor(self, content, style, alpha=1.0):
        content_tensor = self.prepare_image(content, self.content_size).to(self.device)
        style_tensor = self.prepare_image(style, self.style_size).to(self.device)
        with torch.no_grad():
            mixed = self.mix(self.encoder(content_tensor), self.encoder(style_tensor), alpha)
            image = self.decoder(mixed)
        return image.clamp(0, 1).cpu()

    def stylize(self, content, style, alpha=1.0):
        """Return one RGB PIL image. For a batch tensor, return the first image."""
        return tensor_to_pil(self.stylize_tensor(content, style, alpha))


def stylize(content, style, alpha=1.0, model=None):
    """Style-transfer one content image. This is the fixed interface for member 3."""
    if model is None:
        model = stylize.model
        if model is None:
            model = AdaINModel()
            stylize.model = model
    return model.stylize(content, style, alpha)


stylize.model = None


def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _alpha_tensor(alpha, features):
    if isinstance(alpha, (int, float)):
        value = float(alpha)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'alpha must be between 0 and 1, got {value}')
        return torch.tensor(value, dtype=features.dtype, device=features.device)
    tensor = torch.as_tensor(alpha, dtype=features.dtype).detach().to(features.device)
    if tensor.numel() == 1:
        value = float(tensor)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'alpha must be between 0 and 1, got {value}')
        return tensor.reshape(1)
    if tensor.ndim == 2:
        tensor = tensor.view(1, 1, tensor.size(0), tensor.size(1))
    elif tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.size(0) == features.size(0):
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f'alpha map must be HW, 1HW, NHW, or N1HW, got shape {tuple(tensor.shape)}')
    if tensor.size(1) != 1:
        raise ValueError(f'alpha map must have one channel, got shape {tuple(tensor.shape)}')
    if tensor.shape[-2:] != features.shape[-2:]:
        tensor = torch.nn.functional.interpolate(tensor, size=features.shape[-2:], mode='bilinear', align_corners=False)
    if float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        raise ValueError('spatial alpha values must be between 0 and 1')
    return tensor
