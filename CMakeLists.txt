cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tinychirp)

# Set via env vars: sysbuild doesn't forward arbitrary -D args to the app image.
set(MODEL       "$ENV{MODEL}"       CACHE STRING "" FORCE)
set(MEL_BACKEND "$ENV{MEL_BACKEND}" CACHE STRING "" FORCE)

target_compile_options(app PRIVATE -Wa,-I${CMAKE_CURRENT_SOURCE_DIR}/src/nrf)
target_include_directories(app PRIVATE src/nrf)

if(MODEL STREQUAL "sincnet_chunked")
    # Raw audio in, sincnet_chunked Axon model, no mel preprocessing.
    target_sources(app PRIVATE
        src/nrf/main_sincnet.c
        src/nrf/sample_input_sincnet.c
    )
else()
    # Default: cnn_mel path (optionally with on-device mel compute).
    target_sources(app PRIVATE
        src/nrf/main.c
        src/nrf/sample_input_raw.c
    )
    if(MEL_BACKEND)
        target_compile_definitions(app PRIVATE MEL_FROM_RAW)
        target_sources(app PRIVATE src/nrf/mel_compute.c)

        if(MEL_BACKEND STREQUAL "cpu")
            target_compile_definitions(app PRIVATE MEL_BACKEND_CPU)
            target_sources(app PRIVATE src/nrf/mel_fft_cpu.c)
        elseif(MEL_BACKEND STREQUAL "axon")
            target_compile_definitions(app PRIVATE MEL_BACKEND_AXON)
            target_sources(app PRIVATE src/nrf/mel_fft_axon_split.c)
        else()
            message(FATAL_ERROR "MEL_BACKEND must be 'cpu' or 'axon' (got '${MEL_BACKEND}')")
        endif()
    endif()
endif()
