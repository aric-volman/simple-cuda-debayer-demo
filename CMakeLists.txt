cmake_minimum_required(VERSION 3.18)
project(cuda_debayer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(debayer
    main.cpp
    kernel.cu
)

target_include_directories(debayer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(debayer PRIVATE ${OpenCV_LIBS})

set_target_properties(debayer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(debayer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
