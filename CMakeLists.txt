cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tinychirp_cnn_mel_axon)

# Set via env var: sysbuild doesn't forward arbitrary -D args to the app image.
set(MEL_BACKEND "$ENV{MEL_BACKEND}" CACHE STRING "" FORCE)

target_sources(app PRIVATE
    src/main.c
    src/sample_input_raw.c
)
target_compile_options(app PRIVATE -Wa,-I${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MEL_BACKEND)
    target_compile_definitions(app PRIVATE MEL_FROM_RAW)
    target_sources(app PRIVATE src/mel_compute.c)

    if(MEL_BACKEND STREQUAL "cpu")
        target_compile_definitions(app PRIVATE MEL_BACKEND_CPU)
        target_sources(app PRIVATE src/mel_fft_cpu.c)
    elseif(MEL_BACKEND STREQUAL "axon")
        target_compile_definitions(app PRIVATE MEL_BACKEND_AXON)
        target_sources(app PRIVATE src/mel_fft_axon_split.c)
    else()
        message(FATAL_ERROR "MEL_BACKEND must be 'cpu' or 'axon' (got '${MEL_BACKEND}')")
    endif()
endif()

target_include_directories(app PRIVATE src)
