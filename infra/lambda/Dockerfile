# The champion as a Lambda container: DINOv2 ViT-B/14 and the promoted
# probe head, with the weights inside the image so a cold start downloads
# nothing. Build with `make lambda-image`, which stages the weights first.
#
# The base is the plain Python image plus AWS's runtime interface client
# (awslambdaric), not AWS's own Lambda image: Docker Hub's copy of that
# image stops at Python 3.13 and uv.lock holds only 3.14 wheels. The base
# and uv are pinned by digest so a rebuild gets the same layers.
ARG BASE=python:3.14-slim@sha256:fbecd461fc72d21b2143c6eaba25143738079154a6d0ea3da3aec724f50eed4a
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv
FROM ${BASE}

# The lambda dependency group exactly as uv.lock pins it, hashes checked.
# The project itself is not a package, so only dependencies are installed.
COPY --from=uv /uv /bin/uv
WORKDIR /var/task
COPY pyproject.toml uv.lock .python-version ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-default-groups --group lambda --no-install-project --no-cache

# The model: staged by `make lambda-model` from the Hugging Face cache at
# the revision the embedding cache records, and the committed head.
COPY build/lambda/model.safetensors /opt/model/model.safetensors
COPY results/holdout/probe_head.npz /opt/model/probe_head.npz

# Only the modules the handler imports: the champion core and the handler.
COPY src/cameratrap/__init__.py cameratrap/__init__.py
COPY src/cameratrap/embed/__init__.py src/cameratrap/embed/dinov2.py cameratrap/embed/
COPY src/cameratrap/arms/__init__.py src/cameratrap/arms/head.py src/cameratrap/arms/champion.py cameratrap/arms/
COPY src/cloud/__init__.py src/cloud/handler.py cloud/

# Lambda's filesystem is read-only apart from /tmp. HF_HUB_OFFLINE makes a
# missing weights file fail at once, not try a download.
ENV PYTHONPATH=/var/task \
    MODEL_DIR=/opt/model \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["/opt/venv/bin/python", "-m", "awslambdaric"]
CMD ["cloud.handler.handler"]
