# Guide 
This guide helps run training without network.

## Step
In the 1st machine which can connect to internet, assume the training data locates at below path.

`/home/username/`

Run below steps.
```
$ docker run -it --rm -v /var/run/docker.sock:/var/run/docker.sock mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04 /bin/bash

# mkdir /home/username    (Note: make sure the username is the same as above)

# apt-get update

# apt-get -y install python3-pip unzip vim

# pip3 install --ignore-installed --no-cache-dir pip

# pip3 install nvidia-pyindex

# pip3 install nvidia-tao

# curl -sSL https://get.docker.com/ | sh

# docker login --username=\$oauthtoken --password="your_ngc_key"  nvcr.io

# touch ~/.tao_mounts.json

# vim ~/.tao_mounts.json 

{
   "Mounts":[
         {
            "source": "/home/username",
            "destination": "/workspace"
         }
   ],
   "Envs": [
         {
            "variable": "CUDA_DEVICE_ORDER",
            "value": "PCI_BUS_ID"
         }
   ],
   "DockerOptions":{
         "shm_size": "16G",
         "ulimits": {
            "memlock": -1,
            "stack": 67108864
         }
   }
}
```


Run below command to pull 4 kinds of tao dockers.
```
# docker pull nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.11-tf1.15.4-py3

# docker pull nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.11-tf1.15.5-py3

# docker pull nvcr.io/nvidia/tao/tao-toolkit-pyt:v3.21.11-py3

# docker pull nvidia/tao/tao-toolkit-lm:v3.21.08-py3
```


(Optional) Run tao training, for example
```
# tao classification train -k nvidia_tlt -r /workspace/demo/result -e /workspace/demo/classification_spec.cfg
```




Open another terminal in the 1st machine,  save the new_azure docker and 4 kinds of tao dockers into files.
```
$  docker ps

$  docker commit  <CONTAINER ID of mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04>  new_azure:version_1 

$  docker save -o new_azure_version_1.tar.gz  new_azure:version_1

$  docker save -o tao-toolkit-tf-v3.21.11-tf1.15.5-py3.tar.gz nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.11-tf1.15.5-py3

$  docker save -o tao-toolkit-tf-v3.21.11-tf1.15.4-py3.tar.gz nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.11-tf1.15.4-py3

$  docker save -o tao-toolkit-pyt-v3.21.11-py3.tar.gz  nvcr.io/nvidia/tao/tao-toolkit-pyt:v3.21.11-py3

$  docker save -o tao-toolkit-lm-v3.21.08-py3.tar.gz   nvcr.io/nvidia/tao/tao-toolkit-lm:v3.21.08-py3
```




Copy all the tar.gz files into the 2nd machine which has no internet.
```
$ docker load -i new_azure_version_1.tar.gz

$ docker load -i tao-toolkit-tf-v3.21.11-tf1.15.4-py3.tar.gz

$ docker load -i tao-toolkit-tf-v3.21.11-tf1.15.5-py3.tar.gz

$ docker load -i tao-toolkit-pyt-v3.21.11-py3.tar.gz

$ docker load -i tao-toolkit-lm-v3.21.08-py3.tar.gz
```

Copy the training dataset into below path of the 2nd machine. If the path is not available, please generate the same as the 1st machine.
`/home/username/`



In the 2nd machine, login the new azure docker
```
$ docker run -it --rm -v /var/run/docker.sock:/var/run/docker.sock new_azure:version_1 /bin/bash
```

Then run training.
