project(
  'dembonespython',
  'cpp',
  default_options: ['cpp_std=c++20'],
  meson_version : '>=1.7.0',
  version : '0.1.0',
)

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
#pybind_dep = dependency('pybind11')
nanobind_dep = dependency('nanobind')
py_dep = dependency('python3')
omp_dep = dependency('openmp')

os_name = build_machine.system()
db_compile_args = []

compiler_id = meson.get_compiler('cpp').get_id()

if os_name == 'windows'
  if meson.get_compiler('cpp').get_id() == 'msvc'
    db_compile_args += ['/bigobj']
  endif
endif

pymod = import('python')
py = pymod.find_installation(pure:false)

if py.language_version().version_compare('< 3.8')
  error('Invalid Python version, only >= 3.8 is supported.')
endif

outlib = py.extension_module(
  '_dem_bones_core',
  sources : source_files,
  subdir: 'dembonespython',
  install: true,
  include_directories : includes,
  cpp_args : db_compile_args,
  dependencies : [omp_dep, eigen_dep, nanobind_dep, py_dep],
)

py.install_sources(
  'dembonespython/__init__.py',
  'dembonespython/maya.py',
  subdir: 'dembonespython',
)
