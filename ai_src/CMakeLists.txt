file(GLOB AI_SOURCES "src/*.cpp")
file(GLOB AI_HEADERS "include/*.h")

find_package(OpenCV REQUIRED)

add_library(ai_solution STATIC ${AI_SOURCES})

target_include_directories(ai_solution PUBLIC
    ${CMAKE_SOURCE_DIR}/ai_src/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)



add_subdirectory(face_tracking)


target_link_libraries(ai_solution PRIVATE
    ${OpenCV_LIBS}
    ByteTrackEigen
    /usr/local/lib/librknnrt.so

)
target_link_options(ai_solution PRIVATE
    -fopenmp
    -s
    -ldl
)
