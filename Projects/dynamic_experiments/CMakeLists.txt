cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(mosek_solution)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/finders")

find_package(Mosek)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Threads)

add_executable(${PROJECT_NAME} "experiment.cpp")
#above means your main file might be called mosek_solution.cpp

target_include_directories(${PROJECT_NAME} PRIVATE ${MOSEK_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE Mosek::FusionCXX Eigen3::Eigen Threads::Threads)
