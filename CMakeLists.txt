cmake_minimum_required(VERSION 3.16)

project(mills C)
set(CMAKE_C_STANDARD 11)

add_executable(mills prog.c drand48.c)

set_property(TARGET mills PROPERTY 
	INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET mills PROPERTY
	MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")