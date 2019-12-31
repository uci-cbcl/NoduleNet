from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='box',
      ext_modules=[CppExtension('box', ['box.cpp'])],
      cmdclass={'build_ext': BuildExtension})

# setuptools.Extension(
#    name='box',
#    sources=['overlap.cpp'],
#    include_dirs=torch.utils.cpp_extension.include_paths(),
#    language='c++')