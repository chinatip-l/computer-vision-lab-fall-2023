cmake_minimum_required(VERSION 3.0.0)
project(homework3_opencv VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )




add_executable(homework3_opencv main.cpp func.cpp)
target_link_libraries( homework3_opencv ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
