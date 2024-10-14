# HNSWLib Server

This is a lightweight HTTP server that wraps the HNSWLib Library. It allows you to build a HNSW index and query it using a simple REST API. 

The server is written in C++.

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
