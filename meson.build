project('dem-bones-python', 'cpp', default_options: ['cpp_std=c++20'])

source_files = files([
  'src/main.cpp',
])

eigen_dep = dependency('eigen3')
pybind_dep = dependency('pybind11')
dembones_dep = dependency('dembones')
py_dep = dependency('python3')

pymod = import('python')
py = pymod.find_installation('python3')
outlib = py.extension_module(
  meson.project_name(),
  sources : source_files,
  install: true,
  install_dir : meson.global_source_root() / 'output_Python',
  include_directories : include_directories(['src']),
  dependencies : [eigen_dep, pybind_dep, dembones_dep, py_dep],
)
