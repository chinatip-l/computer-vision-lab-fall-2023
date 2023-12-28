cmake_minimum_required(VERSION 3.8.0)
project(playground VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )



include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_SOURCE_FILES
    cuda_example.cu
)
set(CUDA_NVCC_FLAGS "-arch=native")
set(CMAKE_CUDA_ARCHITECTURES 70) 


# ADD_EXECUTABLE(playground main.cpp)
cuda_add_executable(playground main.cpp kernel.cu)
target_link_libraries( playground ${OpenCV_LIBS} )

# set_target_properties(playground PROPERTIES CUDA_ARCHITECTURES "70;75")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
