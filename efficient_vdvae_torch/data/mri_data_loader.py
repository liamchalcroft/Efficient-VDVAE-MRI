import torch.utils.data
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.utils import shuffle
from hparams import HParams
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
import glob
from random import randrange

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class Normalize(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype

        img = np.floor(img / np.uint8(2 ** (8 - hparams.data.num_bits))) * 2 ** (
            8 - hparams.data.num_bits
        )
        img = img.astype(img_dtype)

        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinMax(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Tensor
        """
        img = np.asarray(img)

        shift = scale = (2**8 - 1) / 2
        img = (img - shift) / scale  # Images are between [-1, 1]
        return torch.tensor(img).contiguous().float()[None]

    def __repr__(self):
        return self.__class__.__name__ + "()"


train_transform = transforms.Compose(
    [
        Normalize(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(30, (0.1, 0.1), (0.9, 1.1), 0.5),
        MinMax(),
    ]
)


valid_transform = transforms.Compose(
    [
        Normalize(),
        MinMax(),
    ]
)


def create_filenames_list_ixi(path):
    files = glob.glob(os.path.join(path, "*l1.png"))
    files = [f.replace("_l1", "") for f in files]
    filenames = [f.split("/")[-1] for f in files]
    print(path, len(files))
    return files, filenames


def read_resize_image(image_file):
    return Image.open(image_file).resize(
        (hparams.data.target_res, hparams.data.target_res), resample=Image.BILINEAR
    )


def read_crop_image(image_file):
    img = Image.open(image_file)
    x, y = img.size
    x1 = randrange(0, x - hparams.data.target_res)
    y1 = randrange(0, y - hparams.data.target_res)
    return img.crop(
        (x1, y1, x1 + hparams.data.target_res, y1 + hparams.data.target_res)
    )


class generic_dataset(torch.utils.data.Dataset):
    def __init__(self, files, filenames, mode):

        self.mode = mode
        if mode != "encode":
            self.files, self.filenames = shuffle(files, filenames)
        else:
            self.files = files
            self.filenames = filenames

    def __getitem__(self, idx):
        if self.mode == "train":
            img = read_crop_image(self.files[idx])
            img = train_transform(img)
            return img

        elif self.mode in ["val", "div_stats", "test"]:
            img = read_crop_image(self.files[idx])
            img = valid_transform(img)
            return img

        elif self.mode == "encode":
            filename = self.filenames[idx]
            img = read_crop_image(self.files[idx])
            img = valid_transform(img)
            return img, filename

        else:
            raise ValueError(f"Unknown Mode {self.mode}")

    def __len__(self):
        if self.mode in ["train", "encode", "test"]:
            return len(self.files)
        elif self.mode == "val":
            return hparams.val.n_samples_for_validation
        elif self.mode == "div_stats":
            return round(len(self.files) * hparams.synthesis.div_stats_subset_ratio)


def train_val_data_ixi(
    train_images, train_filenames, val_images, val_filenames, world_size, rank
):
    train_data = generic_dataset(train_images, train_filenames, mode="train")
    train_sampler = (
        DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        if hparams.run.local == 0
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        sampler=train_sampler,
        dataset=train_data,
        batch_size=hparams.train.batch_size
        if hparams.run.num_gpus == 0
        else hparams.train.batch_size // hparams.run.num_gpus,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        prefetch_factor=3,
    )

    val_data = generic_dataset(val_images, val_filenames, mode="val")
    val_sampler = (
        DistributedSampler(
            val_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        if hparams.run.local == 0
        else None
    )
    val_loader = torch.utils.data.DataLoader(
        sampler=val_sampler,
        dataset=val_data,
        batch_size=hparams.val.batch_size
        if hparams.run.num_gpus == 0
        else hparams.val.batch_size // hparams.run.num_gpus,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        prefetch_factor=3,
    )

    return train_loader, val_loader


def synth_generic_data():
    synth_images, synth_filenames = create_filenames_list_ixi(
        hparams.data.synthesis_data_path
    )
    synth_data = generic_dataset(synth_images, synth_filenames, mode="test")
    synth_loader = torch.utils.data.DataLoader(
        dataset=synth_data,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )
    return synth_loader


def encode_generic_data():
    images, filenames = create_filenames_list_ixi(hparams.data.train_data_path)
    data = generic_dataset(images, filenames, mode="encode")
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=hparams.synthesis.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    return data_loader


def stats_generic_data():
    images, filenames = create_filenames_list_ixi(hparams.data.train_data_path)
    data = generic_dataset(images, filenames, mode="div_stats")
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    return data_loader
