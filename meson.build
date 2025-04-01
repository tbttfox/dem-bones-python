project('dem-bones-python', 'cpp', default_options: ['cpp_std=c++20'])

source_files = files([
  'src/py_dem_bones.cpp',
  'src/py_dem_bones_ext.cpp',
  'src/py_dem_bones_module.cpp',
])

# DemBones has a pretty annoying cmake setup
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
py = pymod.find_installation('python3')
outlib = py.extension_module(
  meson.project_name(),
  sources : source_files,
  install: true,
  install_dir : meson.global_source_root() / 'output_Python',
  include_directories : includes,
  cpp_args : db_compile_args,
  dependencies : [eigen_dep, pybind_dep, py_dep],
)
