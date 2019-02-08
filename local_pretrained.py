#!/usr/bin/env python3

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import re
import os
import os.path
import sys
import glob
import pickle
import numpy as np
import PIL.Image
import tensorflow as tf
import dnnlib.tflib as tflib
import logging
import hashlib
import random
import argparse
import datetime


_log = logging.getLogger(__name__)
_RANDMAX = 2 ** 32

def _hash(thing, alg='md5'):
    h = hashlib.new(alg)
    h.update(bytes(thing))
    return h.hexdigest()


def _timestamp():
    now = datetime.datetime.now()
    now_iso = now.isoformat()
    now_iso = re.sub(r'\.\d{6}$', '', now_iso)  # truncate microseconds
    now_iso = now_iso.replace(':', '')
    return re.sub(r'[^T0-9.]', '', now_iso)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained_path", metavar="FILE", nargs='?', help="pathname of the pickled pretrained model")
    parser.add_argument("--seed", type=int, metavar="N", help="integer to seed random latents")
    parser.add_argument("--print-layers", action='store_true', help="print network details")
    parser.add_argument("--output-dir", metavar="DIR", default=os.path.join(os.getcwd(), "examples"), help="set directory for output files")
    parser.add_argument("--truncation-psi", type=float, metavar="FLOAT", default=0.7, help="set truncation PSI")
    parser.add_argument("--log-level", "-l", choices=('DEBUG', 'INFO', 'WARN', 'ERROR'), default='INFO', help="set log level")
    parser.add_argument("--num-images", "-n", type=int, default=1, help="number of images to generate")
    parser.add_argument("--not-noisy", dest="randomize_noise", action='store_false', help="randomize noise when generating an image")
    args = parser.parse_args()
    logging.basicConfig(level=logging.__dict__[args.log_level])
    if not args.pretrained_path:
        pattern = os.path.join(os.getcwd(), "models", "*.pkl")
        _log.debug("searching pattern %s for model files", pattern)
        candidates = glob.glob(pattern)
        if not candidates:
            print("no suitable candidates for models found; specify model pathname as positionl argument", file=sys.stderr)
            return 0
        pretrained_path = candidates[-1]
    else:
        pretrained_path = args.pretrained_path
    _log.debug("using model path: %s", pretrained_path)
    seed = args.seed if args.seed is not None else random.randrange(_RANDMAX)
    rnd = np.random.RandomState(seed)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    # with tf.Session():
    tflib.init_tf()
    # Load pre-trained network.
    with open(pretrained_path, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    if args.print_layers:
        Gs.print_layers()
    latents_shape = Gs.input_shape[1]
    _log.debug("%s latents", latents_shape)
    os.makedirs(args.output_dir, exist_ok=True)
    for index in range(args.num_images):
        latents = rnd.randn(1, latents_shape)
        _log.debug("image %d: seed %d generated latents with hash %s", index, seed, _hash(latents))
        images = Gs.run(latents, None, truncation_psi=args.truncation_psi, randomize_noise=args.randomize_noise, output_transform=fmt)
        png_filename = os.path.join(args.output_dir, "pretrained-example-{}-{}-{}.png".format(seed, index, _timestamp()))
        image = PIL.Image.fromarray(images[0], 'RGB')
        image.save(png_filename)
        print(png_filename)
    return 0

if __name__ == "__main__":
    exit(main())
