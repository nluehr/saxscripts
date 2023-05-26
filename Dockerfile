FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt update \
 && apt install -y --no-install-recommends curl vim git \
 && git config --global user.email "someone@somewhere" \
 && git config --global user.name "Someone" \
 && git clone https://github.com/google/saxml /saxml \
 && cp /saxml/requirements-cuda.txt /saxml/requirements.txt \
 && sed -i /saxml/requirements.txt \
        -e 's/^jaxlib[= @].*$/jaxlib==0.4.10+cuda12.cudnn88/' \
        -e 's/^jax[= @].*$/jax==0.4.10/' \
 && sed -i 's/sudo //g' /saxml/saxml/tools/init_cloud_vm.sh \
 && cd /saxml && saxml/tools/init_cloud_vm.sh \
 && rm -rf /var/lib/apt/lists/*

# TensorFlow 2.11 needs a minor tweak in order to build with CUDA 12
RUN git clone https://github.com/tensorflow/tensorflow.git \
        -b v2.11.0 /tensorflow \
 && cd /tensorflow \
 && git cherry-pick be94c459eaffd51e9cf1f96c13385f6fed9d6752

WORKDIR /saxml

# SAX_ROOT consumed implicitly by saxml/server:server
ENV SAX_ROOT=/sax-root \
    SAX_CELL=/sax/test

# Configure sax admin server
RUN mkdir -p $SAX_ROOT \
 && mkdir -p /sax-fs-root \
 && bazel run \
        --override_repository=org_tensorflow=/tensorflow \
        saxml/bin:admin_config \
        -- \
        --sax_cell=$SAX_CELL \
        --sax_root=$SAX_ROOT \
        --fs_root=/sax-fs-root \
        --alsologtostderr \
 && bazel clean --expunge

# Convenience Scripts
COPY saxgpuserve saxadminserve saxutil /usr/bin/
