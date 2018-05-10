import torch.utils.data as data

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions, class_list):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        if class_to_idx[target] in class_list: # make sure that only class in class_list is collected
            print('found class {}, with path {}'.format(class_to_idx[target], d))
            cnt = 0
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
                        cnt += 1
            print('total img: {}'.format(cnt))
                

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, train, class_list, loader, extensions, transform=None, target_transform=None):
        if train:
            f = open('/home/zhuo/caffe/data/ilsvrc12/train.txt', 'r')
            lines = f.readlines()
            classes = [line.strip().split('/')[0] for line in lines]
            idx = [int(line.strip().split(' ')[1]) for line in lines]
            class_to_idx = {classes[i]:idx[i] for i in range(len(classes))}
            classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            assert len(classes) == len(class_to_idx)
            samples = make_dataset(root, class_to_idx, extensions, class_list)
        else:
            class_to_idx = None
            classes = None
            f = open('/home/zhuo/caffe/data/ilsvrc12/val.txt', 'r')
            lines = f.readlines()
            samples = []
            for line in lines:
                fn = line.strip().split(' ')[0]
                label = int(line.strip().split(' ')[1])
                if label in class_list:
                    samples.append((os.path.join(root,fn), label))

        print('a sample: {}'.format(samples[-1]))
        print('sample size: {}'.format(len(samples)))
            

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.class_list = class_list

        global_2_subset = {}
        for i,val in enumerate(class_list):
            global_2_subset[val] = i
        self.global_2_subset = global_2_subset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        print(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # transform to subset label index
        target = self.global_2_subset[target]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.JPEG']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, train, class_list, transform=None, target_transform=None,
                 loader=default_loader):
        if train:
            root = '/home/zhuo/train'
        else:
            root = '/home/zhuo/val'
        super(ImageFolder, self).__init__(root, train, class_list, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
