name: build-juicy-bank-universal

on:
  push:
    branches: [ main, master ]
  pull_request:

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      # -------- ARMv7 (Bela / Debian 9 "stretch") build on Linux runner --------
      - name: Set up QEMU (Linux)
        if: runner.os == 'Linux'
        uses: docker/setup-qemu-action@v3
        with:
          platforms: arm

      - name: Build ARMv7 (hard-float) in Debian Stretch container
        if: runner.os == 'Linux'
        shell: bash
        run: |
          set -euxo pipefail

          docker run --rm --platform linux/arm/v7             -v "$GITHUB_WORKSPACE:/work" -w /work             arm32v7/debian:stretch bash -lc '
              set -euxo pipefail

              # Debian 9 "stretch" is EOL; use archive.debian.org mirrors.
              printf "%s\n"                 "deb http://archive.debian.org/debian stretch main"                 "deb http://archive.debian.org/debian-security stretch/updates main"                 > /etc/apt/sources.list

              # Disable "Valid-Until" checks (archive metadata is expired).
              echo "Acquire::Check-Valid-Until \"false\";" > /etc/apt/apt.conf.d/99no-check-valid-until

              apt-get -o Acquire::Check-Valid-Until=false update
              apt-get install -y --no-install-recommends                 build-essential make wget ca-certificates xz-utils file

              # Pd headers: match Bela Pd 0.51-4 (grab source and use m_pd.h)
              mkdir -p /tmp/pdsrc
              cd /tmp/pdsrc
              wget -qO pd.tar.gz https://github.com/pure-data/pure-data/archive/refs/tags/0.51-4.tar.gz
              tar -xzf pd.tar.gz
              export PDINC="/tmp/pdsrc/pure-data-0.51-4/src"

              cd /work
              make all CC=gcc PDINC="$PDINC"

              echo "---- file output ----"
              file build/**/juicy_bank~.pd_linux || true
            '

      - name: Upload ARMv7 artifact (Linux)
        if: runner.os == 'Linux'
        uses: actions/upload-artifact@v4
        with:
          name: juicy_bank-bela-armv7-debian9
          path: build/**/juicy_bank~.pd_linux

      # ------------------------------ macOS universal build ------------------------------
      - name: Fetch Pd headers (macOS)
        if: runner.os == 'macOS'
        shell: bash
        run: |
          set -euxo pipefail
          # Use Pure Data source headers (m_pd.h) matching Bela's Pd version.
          mkdir -p /tmp/pdsrc
          cd /tmp/pdsrc
          curl -L -o pd.tar.gz https://github.com/pure-data/pure-data/archive/refs/tags/0.51-4.tar.gz
          tar -xzf pd.tar.gz
          echo "PDINC=/tmp/pdsrc/pure-data-0.51-4/src" >> "$GITHUB_ENV"

      - name: Build (macOS)
        if: runner.os == 'macOS'
        shell: bash
        run: |
          set -euxo pipefail
          make all PDINC="$PDINC"

      - name: Upload macOS artifact
        if: runner.os == 'macOS'
        uses: actions/upload-artifact@v4
        with:
          name: juicy_bank-macos-universal
          path: build/**/juicy_bank~.pd_darwin
