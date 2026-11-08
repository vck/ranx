cmake_minimum_required(VERSION 3.15)
project(ranx VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)

add_executable(ranx
    src/main.c
    src/gguf.c
    src/model.c
    src/ops.c
    src/tokenizer.c
)
target_include_directories(ranx PRIVATE src)
target_link_libraries(ranx PRIVATE ${ACCELERATE_LIBRARY})
target_compile_options(ranx PRIVATE -Wall -Wextra -Wpedantic -O3 -march=armv8.5-a+fp16 -DACCELERATE_NEW_LAPACK)
