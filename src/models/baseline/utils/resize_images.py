import os
from PIL import Image


def resize_image(image, size):
    return image.resize(size, Image.Resampling.LANCZOS)


def resize_images(input_dir, output_dir, size):
    for idir in os.scandir(input_dir):
        if not idir.is_dir():
            continue

        if not os.path.exists(output_dir + '/' + idir.name):
            os.makedirs(output_dir + '/' + idir.name)

        images = os.listdir(idir.path)
        n_images = len(images)
        for iimage, image in enumerate(images):
            try:
                with open(os.path.join(idir.path, image), 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img.save(os.path.join(output_dir + '/' + idir.name, image), img.format)
            except(IOError, SyntaxError) as e:
                pass

            if (iimage + 1) % 1000 == 0:
                print("[{}/{}] Resized the images and saved into '{}'."
                      .format(iimage + 1, n_images, output_dir + '/' + idir.name))