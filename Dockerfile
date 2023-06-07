FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt update \
 && apt install -y --no-install-recommends \
        cuda-nsight-systems-12-1 curl vim git \
 && git config --global user.email "someone@somewhere" \
 && git config --global user.name "Someone" \
 && git clone https://github.com/google/saxml /saxml \
 && cd /saxml \
 && git checkout 95243a8764d92a3cfd999d3045c5ddd464147b59 \
 && cp requirements-cuda.txt requirements.txt \
 && sed -i requirements.txt \
        -e '/jax_cuda_releases.html/a --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html' \
        -e 's/^jaxlib[= @].*$/jaxlib==0.4.11.dev20230526+cuda12.cudnn88/' \
        -e 's|^jax[= @].*$|jax[cuda12_local] @ git+https://github.com/google/jax@9615a31a73f16c83ac2e1bd1c444221cbccb5abc|' \
        -e 's|\(^paxml.*$\)|\1@32de4662600c5e3fafcd63dd493c7f7abd4692ee|' \
        -e 's|\(^praxis.*$\)|\1@d39c631c9482950d542672c282e7fa88aab48bff|' \
        -e 's|^orbax-checkpoint==.*$|orbax-checkpoint==0.2.3|' \
 && sed -i 's/sudo //g' saxml/tools/init_cloud_vm.sh \
 && saxml/tools/init_cloud_vm.sh \
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
        --alsologtostderr

# Convenience Scripts
COPY saxgpuserve saxadminserve saxutil /usr/bin/
