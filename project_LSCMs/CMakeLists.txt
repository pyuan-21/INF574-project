cmake_minimum_required(VERSION 3.1)
project(LSCM)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)

file(GLOB SRCFILES *.cpp)
add_executable(ex_bin ${SRCFILES})

target_link_libraries(ex_bin igl::core igl::opengl_glfw)
