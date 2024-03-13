cmake_minimum_required(VERSION 3.16)
project(DBoW2)
include(ExternalProject)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set (CMAKE_CXX_STANDARD 17)

add_compile_options(-Wextra -Wpedantic)

set(HDRS_DBOW2
  include/DBoW2/BowVector.h           include/DBoW2/FBrief.h
  include/DBoW2/QueryResults.h        include/DBoW2/TemplatedDatabase.h   
  include/DBoW2/DBoW2.h               include/DBoW2/FClass.h              include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       include/DBoW2/TemplatedVocabulary.h
  include/DBoW2/FOrb.h
  )
  
set(SRCS_DBOW2 
  src/BowVector.cpp     src/FBrief.cpp        
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp
  src/FOrb.cpp
  )
  
set(HDRS_DUTILS
  include/DBoW2/Random.h
  include/DBoW2/Timestamp.h)

set(SRCS_DUTILS
  src/Random.cpp
  src/Timestamp.cpp)
  
set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} SHARED ${SRCS_DBOW2} ${SRCS_DUTILS})
target_include_directories(${PROJECT_NAME} PUBLIC include/DBoW2/ include/)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

add_executable(demo executables/demo.cpp)
target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBRARIES})
file(COPY executables/images DESTINATION ${CMAKE_BINARY_DIR}/)

add_executable(createVocabulary executables/createVocabulary.cpp)
target_link_libraries(createVocabulary ${PROJECT_NAME} ${OpenCV_LIBRARIES})

configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DBoW2)
  install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)

