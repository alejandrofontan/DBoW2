cmake_minimum_required(VERSION 3.16)
project(DBoW2)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_compile_options(-Wextra -Wpedantic)

set(HDRS_DBOW2
  include/DBoW2/BowVector.h           
  include/DBoW2/FBrief.h
  include/DBoW2/QueryResults.h        
  include/DBoW2/TemplatedDatabase.h   
  include/DBoW2/DBoW2.h               
  include/DBoW2/FClass.h              
  include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       
  include/DBoW2/TemplatedVocabulary.h
  include/DBoW2/FOrb.h
  include/DBoW2/FR2d2.h
  include/DBoW2/FSift128.h
  include/DBoW2/FKaze64.h
  include/DBoW2/FSurf64.h
  include/DBoW2/FBrisk.h
  include/DBoW2/FAkaze61.h
  include/DBoW2/FAnyFeatBin.h
  include/DBoW2/FAnyFeatNonBin.h
  )
  
set(SRCS_DBOW2 
  src/BowVector.cpp     
  src/FBrief.cpp        
  src/FeatureVector.cpp 
  src/QueryResults.cpp  
  src/ScoringObject.cpp
  src/FOrb.cpp
  src/FR2d2.cpp
  src/FSift128.cpp
  src/FKaze64.cpp
  src/FSurf64.cpp
  src/FBrisk.cpp
  src/FAkaze61.cpp
  src/FAnyFeatBin.cpp
  src/FAnyFeatNonBin.cpp
)
  
set(HDRS_DUTILS
  include/DBoW2/Random.h
  include/DBoW2/Timestamp.h)

set(SRCS_DUTILS
  src/Random.cpp
  src/Timestamp.cpp)
  
find_package(OpenCV REQUIRED)
find_package(brisk REQUIRED)
find_package(akaze REQUIRED)
find_package(SiftGPU REQUIRED)

add_library(${PROJECT_NAME} SHARED ${SRCS_DBOW2} ${SRCS_DUTILS})

target_include_directories(${PROJECT_NAME} PUBLIC 
	include/DBoW2/ 
	include/
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBRARIES}
	${brisk_LIBRARIES}
	${akaze_LIBRARIES}
	${SiftGPU_LIBRARIES}
)


add_executable(createVocabulary executables/createVocabulary.cpp)
target_link_libraries(createVocabulary ${PROJECT_NAME})

configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)  
