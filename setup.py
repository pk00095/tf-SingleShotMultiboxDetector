from importlib.machinery import SourceFileLoader
import setuptools,os

from setuptools import setup
from setuptools.command.build_py import build_py as build_py_orig

import numpy

version = SourceFileLoader('tf_ssd.version', os.path.join('tf_ssd', 'version.py')).load_module().VERSION


setuptools.setup(
    name             = 'tf_ssd',
    version          = version,
    description      = 'Tensorflow implementation of SSD object detection using keras api.',
    url              = 'https://github.com/pk00095/tf-SingleShotMultiboxDetector',
    author           = 'T Pratik',
    author_email     = 'pk00095@gmail.com',
    maintainer       = 'T Pratik',
    maintainer_email = 'pk00095@gmail.com',
    # cmdclass         = {'build_ext': BuildExtension},
    packages         = setuptools.find_packages(),
    install_requires = ['Pillow', 'opencv-python', 'tqdm'],
    entry_points={
      "console_scripts": [
          "tf_ssd_prepare=tf_ssd.cli:main",
          "tf_ssd_train300=tf_ssd.train_ssd300:main",
          "tf_ssd_getchessdataset=tf_ssd.cli:download_chess_dataset"]
    },
    # ext_modules    = cythonize(extensions, exclude=cython_excludes, compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()],
    # setup_requires = ["numpy>=1.14.0"]
)

# pydoc-markdown ./pydoc-markdown.yml
# mkdocs build -f pydocs/build/mkdocs.yml -d ../site