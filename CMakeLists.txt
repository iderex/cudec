cmake_minimum_required(VERSION 3.24)

project(cudec VERSION 0.0.1 LANGUAGES C)

# Host-only stub for now: it gives CI a real build gate from day one. The
# CUDA language, the kernels, and the toolchain requirements arrive with M1.
add_library(cudec src/version.c)
target_include_directories(cudec PUBLIC include)

if(NOT MSVC)
  target_compile_options(cudec PRIVATE -Wall -Wextra -Werror)
endif()
