cmake_minimum_required(VERSION 3.16)
project(DBoW2)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)

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

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC 
	include/DBoW2/ 
	include/
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBRARIES}
	${brisk_LIBRARIES}
	${akaze_LIBRARIES}
	${SiftGPU_LIBRARIES}
)

add_executable(createVocabulary executables/createVocabulary.cpp)
target_link_libraries(createVocabulary ${PROJECT_NAME})

# Config file for find_package support
configure_file(src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

# Install the shared library
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Install the header files
install(DIRECTORY DBoW2/ DESTINATION include/DBoW2 FILES_MATCHING PATTERN "*.h")
install(DIRECTORY DUtils/ DESTINATION include/DUtils FILES_MATCHING PATTERN "*.h")

# Optionally install the config file
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake"
        DESTINATION lib/cmake/DBoW2/)