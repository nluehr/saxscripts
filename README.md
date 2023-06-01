Docker Environment to run saxml in OSS
======================================

Step 1: Build and start the container
```
docker build -t saxtest:latest .
docker run --rm --name saxcont --gpus all saxtest:latest saxadminserve
```

Step 2: Update the model MESH and start the GPU server
```
docker exec -it saxcont /bin/bash

# Limit the number of visible devices if needed
#export CUDA_VISIBLE_DEVICES=0,1,2,3

# Edit MESH_SHAPEs for LmCloudSpmd175BTest to match the number
# of visible GPUs (e.g., 8)
vim /saxml/saxml/server/pax/lm/params/lm_cloud.py

saxgpuserve
```

Step 3: Publish the model and run inference
```
docker exec -it saxcont /bin/bash

saxutil publish /sax/test/lm175b \
    saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175BTest \
    None \
    1

# Wait for 'loading completed.' message in gpu server log.

saxutil lm.generate /sax/test/lm175b "Your query here."
```
