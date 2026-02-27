# /---------------------------------------\
# | Stage 1: Build the application binary |
# \---------------------------------------/

FROM gcc:15 AS build
RUN apt-get update && apt-get install -y cmake
COPY . /app
WORKDIR /app
RUN mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j $(nproc)
RUN ./build/test_filters && ./build/test_data_store && ./build/test_save_load

# /------------------------------\
# | Stage 2: Build minimal image |
# \------------------------------/
FROM scratch
COPY --from=build /app/build/bin/server /server
CMD ["./server"]
