# Pinned EDA image for chiprl: Yosys, Icarus, Verilator (apt) + OpenSTA built from source.
# Build:  docker build -t chiprl-eda -f docker/Dockerfile .
# Run:    docker run --rm --network none -v $PWD:/w -w /w chiprl-eda python3 -m chiprl.pipeline tests/fixtures/verilogeval/Prob068_countbcd
# Pin by digest once built (docker inspect --format '{{.Id}}' chiprl-eda) and record it in README.
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG OPENSTA_REF=master
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates cmake build-essential tcl-dev swig bison flex automake autoconf libtool \
      libeigen3-dev libfmt-dev zlib1g-dev python3 python3-pytest \
      yosys iverilog verilator && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/cuddorg/cudd.git /opt/cudd-src && cd /opt/cudd-src \
 && autoreconf -fi && ./configure --prefix=/opt/cudd && make -j"$(nproc)" && make install
RUN git clone https://github.com/The-OpenROAD-Project/OpenSTA.git /opt/OpenSTA && cd /opt/OpenSTA \
 && git checkout ${OPENSTA_REF} && mkdir build && cd build && cmake -DCUDD_DIR=/opt/cudd .. \
 && make -j"$(nproc)" && cp sta /usr/local/bin/sta
RUN useradd -m grader
USER grader
