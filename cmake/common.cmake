

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fexceptions -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_PREFIX_PATH "/usr/local/lib")



find_package(OpenCV REQUIRED)


set(UTILS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/utils")
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen")
set(ENGINE_TOOL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/engine-tool")
set(BYTE_TRACK_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ai_src/face_tracking/include")
set(AI_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ai_src/include")
set(HNSW_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/hnswlib")


include_directories(
    ${HNSW_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${ENGINE_TOOL_INCLUDE_DIR}
    ${AI_INCLUDE_DIR}
    ${UTILS_INCLUDE_DIR}
    ${BYTE_TRACK_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${UTILS_INCLUDE_DIR}
)



set(COMMON_LIBS
    ${OpenCV_LIBRARIES}
)