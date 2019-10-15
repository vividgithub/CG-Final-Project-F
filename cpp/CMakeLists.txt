project(pointsynthsis_custom_ops)

cmake_minimum_required( VERSION 3.05 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DEBUG_POSTFIX "_debug")

# Set helper macros
set(SOURCE_CODE_DIR ${PROJECT_SOURCE_DIR}/src)
set(SOURCE_CODE_LEGACY_DIR ${SOURCE_CODE_DIR}/legacy)
set(SOURCE_CODE_EXTERNAL_DIR ${SOURCE_CODE_DIR}/external)
message(STATUS "SOURCE_CODE_DIR: ${SOURCE_CODE_DIR}")
message(STATUS "SOURCE_CODE_LEGACY_DIR: ${SOURCE_CODE_LEGACY_DIR}")
message(STATUS "SOURCE_CODE_EXTERNAL_DIR: ${SOURCE_CODE_EXTERNAL_DIR}")

set(
	EXTERNAL_SOURCES
	${SOURCE_CODE_EXTERNAL_DIR}/cloud/cloud.cpp
)

# Add link directories
include_directories(
	"C:/Users/vivid/Desktop/PointSynthesis/Python/Tensorflow200/Lib/site-packages/tensorflow_core/include"
	${SOURCE_CODE_DIR}
)

# Add library paths
link_directories(
	"C:/Users/vivid/Desktop/PointSynthesis/Python/Tensorflow200/Lib/site-packages/tensorflow_core/python"
)
link_libraries(_pywrap_tensorflow_internal) 

# Add definitions
if (MSVC)
	message(STATUS "Detect using msvc compiler, add definitions")
	add_definitions("-DCOMPILER_MSVC" "-DNOMINMAX")
endif ()

add_library(
	pointsynthesis_ops
	SHARED
	${SOURCE_CODE_LEGACY_DIR}/tf_neighbors/neighbors/neighbors.cpp
	${SOURCE_CODE_LEGACY_DIR}/tf_neighbors/tf_batch_neighbors.cpp
	${SOURCE_CODE_LEGACY_DIR}/tf_subsampling/grid_subsampling/grid_subsampling.cpp
	${SOURCE_CODE_LEGACY_DIR}/tf_subsampling/tf_batch_subsampling.cpp
	${SOURCE_CODE_DIR}/fixed_radius_search.cpp
	${EXTERNAL_SOURCES}
)

set_target_properties(pointsynthesis_ops PROPERTIES OUTPUT_NAME "ops")

# Add install command
install(
	TARGETS pointsynthesis_ops
	DESTINATION ${PROJECT_SOURCE_DIR}/../ops
)