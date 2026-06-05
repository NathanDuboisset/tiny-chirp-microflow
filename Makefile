NCS_VERSION ?= v3.3.0
BOARD       ?= nrf54lm20dk/nrf54lm20b/cpuapp
BUILD_DIR   ?= build_ncs
ZEPHYR_BASE ?= $(HOME)/ncs/zephyr
SERIAL      ?= /dev/ttyACM1
LOG_FILE    ?= logs/listen.log

LAUNCH = nrfutil toolchain-manager launch --ncs-version $(NCS_VERSION) -- bash -c
WEST   = ZEPHYR_BASE=$(ZEPHYR_BASE) west

.PHONY: build mel_cpu mel_axon sincnet_chunked flash listen clean menuconfig gen_assets compile_cnn_mel compile_sincnet

build:
	$(LAUNCH) '$(WEST) build -b $(BOARD) -p always -d $(BUILD_DIR) .'

mel_cpu:
	$(LAUNCH) 'MEL_BACKEND=cpu  $(WEST) build -b $(BOARD) -p always -d $(BUILD_DIR) .'

mel_axon:
	$(LAUNCH) 'MEL_BACKEND=axon $(WEST) build -b $(BOARD) -p always -d $(BUILD_DIR) .'

sincnet_chunked:
	$(LAUNCH) 'MODEL=sincnet_chunked $(WEST) build -b $(BOARD) -p always -d $(BUILD_DIR) .'

compile_cnn_mel:
	bash scripts/compile.sh cnn_mel.yaml

compile_sincnet:
	bash scripts/compile.sh sincnet_chunked.yaml

gen_assets:
	.venv/bin/python scripts/gen_assets.py

gen_assets_sincnet:
	.venv/bin/python scripts/gen_sincnet_assets.py

flash:
	$(LAUNCH) '$(WEST) flash -d $(BUILD_DIR)'

listen:
	@mkdir -p $(dir $(LOG_FILE))
	@: > $(LOG_FILE)
	@echo "Logging to $(LOG_FILE) (Ctrl-A Ctrl-X to exit)"
	picocom -b 115200 --quiet --logfile $(LOG_FILE) $(SERIAL)

menuconfig:
	$(LAUNCH) '$(WEST) build -t menuconfig -d $(BUILD_DIR)'

clean:
	rm -rf $(BUILD_DIR)
