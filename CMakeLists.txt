cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tinychirp_cnn_mel_axon)

target_sources(app PRIVATE
    src/main.c
    src/sample_input.c
)
target_include_directories(app PRIVATE src)
