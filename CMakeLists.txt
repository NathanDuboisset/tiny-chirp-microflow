cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tinychirp_cnn_mel_axon)

option(MEL_FROM_RAW "Compute log-mel on the Axon DSP engine from raw PCM" OFF)

target_sources(app PRIVATE
    src/main.c
    src/sample_input.c
)

if(MEL_FROM_RAW)
    target_compile_definitions(app PRIVATE MEL_FROM_RAW)
    target_sources(app PRIVATE
        src/sample_input_raw.c
        src/mel_axon.c
    )
endif()

target_include_directories(app PRIVATE src)
