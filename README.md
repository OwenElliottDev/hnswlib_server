# HNSWLib Server

This is a lightweight HTTP server that wraps the HNSWLib Library. It allows you to build a HNSW index and query it using a simple REST API. The server is written in C++.

A bunch of optimisations are applied for compiling and static linking is also used to make it self-contained. The binary is about 2.6MB and we can package it into a Docker container with a busybox base image to make a portable version that is 7MB.


## Building

To build the server you need to have the submodules initialized. You can do this by running:

```bash
git submodule update --init --recursive
```

Then you can build the server by running:

```bash
mkdir build
cd build
cmake ..
make
```

## Running

Run the server by executing the binary from the `build` directory:

```bash
./bin/server
```

## Docker

### Building

```bash
docker build -t hnswlib_server .
```

### Running

```bash
docker run -p 8685:8685 hnswlib_server
```
