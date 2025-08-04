cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(main VERSION 1.0 LANGUAGES CXX)

include(cmake/common.cmake) 


set(SOURCES
    main.cpp

)


add_subdirectory(ai_src)
add_subdirectory(examples)

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}

)

target_link_libraries(main PRIVATE
    ai_solution
    ByteTrackEigen
 
)
target_link_options(main PRIVATE
    -fopenmp
    -s
    -ldl
    -lpthread
    -pthread
    -lgomp
    -rdynamic
    -march=armv8.2-a
)
