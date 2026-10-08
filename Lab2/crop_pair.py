"""Create p1 and a slightly rotated p2 as overlapping 1000x1000 crops.
Usage: python crop_pair.py [input_image] [--output-dir DIRECTORY] [--angle 5]
Positive angles rotate p2 clockwise. The original image is preserved.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageOps

DEFAULT_INPUT = Path(__file__).parent / 'COMP9517_26T3_Lab2_Images' / '微信图片_20261009082625_194_48.jpg'


def crop_pair(input_path, output_dir=None, angle=5.0):
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir else input_path.parent
    with Image.open(input_path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
    width, height = image.size
    size = 1000
    if width <= size or height < size:
        raise ValueError(f'Need width > {size} and height >= {size}; got {width}x{height}.')
    top = (height-size)//2
    first_box = (0, top, size, top+size)
    first = image.crop(first_box)

    rotated = image.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
    # Rotate a source-validity mask with identical geometry. A three-pixel
    # margin excludes interpolation footprints that could touch outside pixels.
    source_mask = np.zeros((height, width), dtype=np.uint8)
    source_mask[3:-3, 3:-3] = 255
    mask = Image.fromarray(source_mask).rotate(-angle, resample=Image.Resampling.BILINEAR, expand=True)
    invalid = (np.asarray(mask) != 255).astype(np.int32)
    integral = np.pad(invalid, ((1,0),(1,0))).cumsum(0).cumsum(1)
    sums = (integral[size:,size:] - integral[:-size,size:]
            - integral[size:,:-size] + integral[:-size,:-size])
    ys, xs = np.nonzero(sums == 0)
    if len(xs) == 0:
        raise ValueError('No fully valid 1000x1000 crop after rotation; use a smaller angle.')
    # Use the furthest-right valid square, then center vertically among ties.
    rightmost = xs.max()
    candidates = ys[xs == rightmost]
    y = int(candidates[np.argmin(abs(candidates-(rotated.height-size)/2))])
    x = int(rightmost)
    second_box = (x, y, x+size, y+size)
    assert np.all(np.asarray(mask)[y:y+size,x:x+size] == 255)
    second = rotated.crop(second_box)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, crop, box in [('p1.jpg',first,first_box), ('p2.jpg',second,second_box)]:
        target = output_dir/name
        crop.save(target, quality=95, subsampling=0)
        with Image.open(target) as check:
            assert check.size == (size,size)
        print(f'{target}: {size}x{size}, crop box {box}')
    print(f'p2: clockwise rotation {angle:g} degrees; all crop pixels validated against source mask.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_image', nargs='?', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--angle', type=float, default=5.0)
    args = parser.parse_args()
    crop_pair(args.input_image, args.output_dir, args.angle)
