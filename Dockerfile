FROM ghcr.io/astral-sh/uv:0.6.17-debian AS build
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LOCKED=1

RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv run --no-dev echo hello

# Pre-download WeSpeaker ONNX model for speaker gate
RUN mkdir -p /data && apt-get update && apt-get install -y --no-install-recommends wget \
    && wget -q -O /data/wespeaker.onnx \
       'https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx' \
    && apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . .
ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --start-period=15s \
    CMD curl --fail http://localhost:80/metrics || exit 1

FROM build AS prod
# Running through uvicorn directly to be able to deactive the Websocket per message deflate which is slowing
# down the replies by a few ms.
CMD ["uv", "run", "--no-dev", "uvicorn", "unmute.main_websocket:app", "--host", "0.0.0.0", "--port", "80", "--ws-per-message-deflate=false"]


FROM build AS hot-reloading
CMD ["uv", "run", "--no-dev", "uvicorn", "unmute.main_websocket:app", "--reload", "--host", "0.0.0.0", "--port", "80", "--ws-per-message-deflate=false"]
