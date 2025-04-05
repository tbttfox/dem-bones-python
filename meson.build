project('dem-bones-python', 'cpp', default_options: ['cpp_std=c++20'], meson_version : '>=1.7.0')

source_files = files([
  'src/main.cpp',
  'src/dbmodel.cpp',
])

# DemBones cmake is only for its command line tool
# and that has all kinds of extra annoying dependencies
# so its easier to just use a git submodule and add the includes directly
includes = include_directories([
  'src',
  'extern/dem-bones/include',
])

eigen_dep = dependency('eigen3')
pybind_dep = dependency('pybind11')
py_dep = dependency('python3')


os_name = build_machine.system()
db_compile_args = []
if os_name == 'windows'
  db_compile_args += ['/bigobj']
endif

pymod = import('python')
pyversion = get_option('pyversion')

py = pymod.find_installation('python' + pyversion)
outlib = py.extension_module(
  '_dem_bones_core',
  sources : source_files,
  install: true,
  install_dir : meson.global_source_root() / 'output_Python',
  include_directories : includes,
  cpp_args : db_compile_args,
  cpp_pch : 'pch/db_pch.h',
  dependencies : [eigen_dep, pybind_dep, py_dep],
)
