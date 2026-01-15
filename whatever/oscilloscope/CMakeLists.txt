cmake_minimum_required(VERSION 3.20)

project(Oscilloscope 
    VERSION 1.0
    LANGUAGES C CXX OBJCXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Objective-C ARC (Automatic Reference Counting)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")

# macOS deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS deployment version")

# Source files
set(SOURCES
    LockFreeRingBuffer.c
    AGC.c
    AudioCaptureManager.mm
    OscilloscopeView.mm
    main.mm
)

# Header files
set(HEADERS
    LockFreeRingBuffer.h
    AGC.h
    AudioCaptureManager.h
    OscilloscopeView.h
)

# Resource files
set(RESOURCES
    Shaders.metal
    Info.plist
)

# Find required frameworks
find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(COREAUDIO_FRAMEWORK CoreAudio REQUIRED)
find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox REQUIRED)
find_library(AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

# Create macOS app bundle
add_executable(Oscilloscope MACOSX_BUNDLE
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

# Link frameworks
target_link_libraries(Oscilloscope PRIVATE
    ${COCOA_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${COREAUDIO_FRAMEWORK}
    ${AUDIOTOOLBOX_FRAMEWORK}
    ${AVFOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
)

# Set bundle properties
set_target_properties(Oscilloscope PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist"
    MACOSX_BUNDLE_BUNDLE_NAME "Vintage Oscilloscope"
    MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.vintage.oscilloscope"
    BUNDLE TRUE
    MACOSX_BUNDLE TRUE
)

# Copy shader file to app bundle Resources
set_source_files_properties(Shaders.metal PROPERTIES
    MACOSX_PACKAGE_LOCATION Resources
)

# Install target (optional - for 'make install')
install(TARGETS Oscilloscope
    BUNDLE DESTINATION .
    RUNTIME DESTINATION bin
)

# =============================================================================
# Unit Tests (with Address Sanitizer and Thread Sanitizer)
# =============================================================================
enable_testing()

# Option to disable sanitizers (e.g., for release builds)
option(ENABLE_SANITIZERS "Enable ASAN and TSAN test builds" ON)

# Test sources
set(TEST_RINGBUFFER_SOURCES test/test_ringbuffer.c LockFreeRingBuffer.c)
set(TEST_AGC_SOURCES test/test_agc.c AGC.c)

# Base test executables (no sanitizers, fast)
add_executable(test_ringbuffer ${TEST_RINGBUFFER_SOURCES})
target_include_directories(test_ringbuffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(test_ringbuffer PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
add_test(NAME RingBufferTests COMMAND test_ringbuffer)

add_executable(test_agc ${TEST_AGC_SOURCES})
target_include_directories(test_agc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_agc PRIVATE m)  # Link math library for tanhf, fabsf, etc.
set_target_properties(test_agc PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
add_test(NAME AGCTests COMMAND test_agc)

if(ENABLE_SANITIZERS)
    # RingBuffer ASAN build (catches memory errors: buffer overflow, use-after-free, leaks)
    add_executable(test_ringbuffer_asan ${TEST_RINGBUFFER_SOURCES})
    target_include_directories(test_ringbuffer_asan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(test_ringbuffer_asan PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
    target_compile_options(test_ringbuffer_asan PRIVATE -fsanitize=address -g -fno-omit-frame-pointer)
    target_link_options(test_ringbuffer_asan PRIVATE -fsanitize=address)
    add_test(NAME RingBufferTests_ASAN COMMAND test_ringbuffer_asan)

    # RingBuffer TSAN build (catches race conditions, deadlocks)
    add_executable(test_ringbuffer_tsan ${TEST_RINGBUFFER_SOURCES})
    target_include_directories(test_ringbuffer_tsan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(test_ringbuffer_tsan PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
    target_compile_options(test_ringbuffer_tsan PRIVATE -fsanitize=thread -g)
    target_link_options(test_ringbuffer_tsan PRIVATE -fsanitize=thread)
    add_test(NAME RingBufferTests_TSAN COMMAND test_ringbuffer_tsan)

    # AGC ASAN build
    add_executable(test_agc_asan ${TEST_AGC_SOURCES})
    target_include_directories(test_agc_asan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_agc_asan PRIVATE m)
    set_target_properties(test_agc_asan PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
    target_compile_options(test_agc_asan PRIVATE -fsanitize=address -g -fno-omit-frame-pointer)
    target_link_options(test_agc_asan PRIVATE -fsanitize=address)
    add_test(NAME AGCTests_ASAN COMMAND test_agc_asan)

    # AGC TSAN build (AGC is single-threaded, but good to verify no hidden issues)
    add_executable(test_agc_tsan ${TEST_AGC_SOURCES})
    target_include_directories(test_agc_tsan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_agc_tsan PRIVATE m)
    set_target_properties(test_agc_tsan PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
    target_compile_options(test_agc_tsan PRIVATE -fsanitize=thread -g)
    target_link_options(test_agc_tsan PRIVATE -fsanitize=thread)
    add_test(NAME AGCTests_TSAN COMMAND test_agc_tsan)

    message(STATUS "Sanitizers ENABLED: ASAN + TSAN test builds included")
endif()

# Collect all test targets
set(ALL_TEST_TARGETS test_ringbuffer test_agc)
if(ENABLE_SANITIZERS)
    list(APPEND ALL_TEST_TARGETS test_ringbuffer_asan test_ringbuffer_tsan test_agc_asan test_agc_tsan)
endif()

# Run all tests after building
add_custom_target(run_tests ALL
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all unit tests (standard + ASAN + TSAN)..."
)
add_dependencies(run_tests ${ALL_TEST_TARGETS})

# Main app depends on tests passing
add_dependencies(Oscilloscope run_tests)

# Print configuration summary
message(STATUS "=== Oscilloscope Build Configuration ===")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
