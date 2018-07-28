import yaml


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG"])


def deprocess(img):
    img = img.add_(1).div_(2)

    return img


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


