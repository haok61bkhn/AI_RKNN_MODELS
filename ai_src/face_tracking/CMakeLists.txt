add_library(ByteTrackEigen SHARED
    src/BaseTrack.cpp
    src/BoundingBoxTrackUtils.cpp
    src/BYTETracker.cpp
    src/HungarianAlgorithmEigen.cpp
    src/KalmanBBoxTrack.cpp
    src/KalmanFilter.cpp
    src/LinearAssignment.cpp
    src/BoundingBoxIoUMatching.cpp
)

target_compile_options(ByteTrackEigen PRIVATE 
  $<$<CONFIG:Debug>:-DDEBUG -g>
  $<$<CONFIG:Release>:-O3>
  -fPIC
)

set_target_properties(ByteTrackEigen PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
)

set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../third_party/eigen")

target_include_directories(ByteTrackEigen PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(ByteTrackEigen PRIVATE include)
target_include_directories(ByteTrackEigen PRIVATE ${CMAKE_SOURCE_DIR}/../src/utils)
target_compile_definitions(ByteTrackEigen PRIVATE BUILDING_BYTE_TRACK_EIGEN)
target_link_libraries(ByteTrackEigen PRIVATE ${OpenCV_LIBS} )
