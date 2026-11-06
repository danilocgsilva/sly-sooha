FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y build-essential cmake git libcurl4-openssl-dev
CMD while : ; do sleep 1000; done
WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp . && git checkout <tag-or-commit>
RUN cmake -B build -DGGML_RPC=ON && cmake --build build --config Release -j$(nproc)
CMD while : ; do sleep 1000; done

# FROM debian:bookworm-slim
# RUN apt-get update && apt-get install -y libcurl4 libgomp1 && rm -rf /var/lib/apt/lists/*
# COPY --from=build /src/build/bin/ /app/
# COPY --from=build /src/build/src/*.so /app/ 
# ENV LD_LIBRARY_PATH=/app