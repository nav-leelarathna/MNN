# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)
SET(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
SET(TRAIN_SOURCE ${ROOT_DIR}/tools/train/source)
include_directories(${TRAIN_SOURCE}/grad)
include_directories(${TRAIN_SOURCE}/optimizer)
include_directories(${TRAIN_SOURCE}/transformer)
include_directories(${TRAIN_SOURCE}/data)
include_directories(${TRAIN_SOURCE}/models)
include_directories(${TRAIN_SOURCE}/datasets)
include_directories(${TRAIN_SOURCE}/demo)
include_directories(${TRAIN_SOURCE}/nn)
file(GLOB GRAD ${TRAIN_SOURCE}/grad/*)
file(GLOB OPT ${TRAIN_SOURCE}/optimizer/*)
file(GLOB TRANS ${TRAIN_SOURCE}/transformer/*)
file(GLOB DATA ${TRAIN_SOURCE}/data/*)
file(GLOB MODELS ${TRAIN_SOURCE}/models/*)
file(GLOB DATASET ${TRAIN_SOURCE}/datasets/*)
file(GLOB DEMOSOURCE ${TRAIN_SOURCE}/demo/*)
file(GLOB NNFILES ${TRAIN_SOURCE}/nn/*)
set(MNN_JNI_SRCS kernelProfiling.cpp ${GRAD} ${OPT} ${TRANS} ${DATA} ${MODELS} ${DATASET} ${DEMOSOURCE} ${NNFILES})
# Declares and names the project.

project("mnnconvolutionoptimisation")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        mnnconvolutionoptimisation
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${MNN_JNI_SRCS})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.
target_compile_definitions(mnnconvolutionoptimisation PRIVATE STB_IMAGE_STATIC STB_IMAGE_IMPLEMENTATION)

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

#find_package(OpenCL REQUIRED)
# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        mnnconvolutionoptimisation
        ${MNN_DEPS}
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})