# Makefile — builds ONLY juicy_bank~ for Pd
SRC_DIR    := src
BUILD_ROOT := build

BANK_SRC  := $(SRC_DIR)/juicy_bank_tilde.c
BANK_BIN  := juicy_bank~

UNAME_S := $(shell uname -s)

# Detect Pd headers
ifeq ($(UNAME_S),Darwin)
  PD_APP := $(firstword \
    $(wildcard /Applications/Pd*.app) \
    $(wildcard $(HOME)/Applications/Pd*.app) \
    $(wildcard /usr/local/Caskroom/pd/*/Pd*.app) \
    $(wildcard /opt/homebrew/Caskroom/pd/*/Pd*.app))
  PDINC ?= $(PD_APP)/Contents/Resources/src
  ifeq ($(PD_APP),)
    $(warning Could not find Pd*.app automatically. Set PDINC=/path/to/Pd.app/Contents/Resources/src)
  endif
  EXT      := pd_darwin
  PLAT     := macos
  CFLAGS  ?= -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type -I"$(PDINC)"
  LDFLAGS ?= -bundle -undefined dynamic_lookup
  LDLIBS  ?=
else ifeq ($(UNAME_S),Linux)
  PDINC ?= /usr/include/pd
  EXT      := pd_linux
  PLAT     := linux
  CFLAGS  ?= -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type -I"$(PDINC)"
  LDFLAGS ?= -shared -fPIC -Wl,-export-dynamic
  LDLIBS  ?= -lm
else
  PDINC ?= C:/Pd/src
  EXT      := pd_win
  PLAT     := windows
  CFLAGS  ?= -O3 -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type -I"$(PDINC)"
  LDFLAGS ?= -shared
  LDLIBS  ?=
endif

BUILD_DIR := $(BUILD_ROOT)/$(PLAT)
OUT := $(BUILD_DIR)/$(BANK_BIN).$(EXT)

.PHONY: all clean dirs help
all: dirs $(OUT)

dirs:
	@mkdir -p "$(BUILD_DIR)"

$(OUT): $(BANK_SRC) | dirs
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	@rm -rf "$(BUILD_ROOT)"

help:
	@echo "PDINC=$(PDINC)"
	@echo "Platform=$(PLAT)  Ext=$(EXT)"
	@echo "Build dir=$(BUILD_DIR)"
