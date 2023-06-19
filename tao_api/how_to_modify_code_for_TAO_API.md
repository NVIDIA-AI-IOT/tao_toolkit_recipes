# How to modify code for TAO API

This guide help you go through the detailed steps of how to modify code and generate new docker for TAO API.


## Trigger docker in one host machine and modify code
Open a terminal, trigger 4.0.0 tao api docker.
```shell
$ docker run -it --name tao-api-fixed nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api /bin/bash
```

Open another terminal. You need to build a new docker image based on nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api.
First, create a folder, copy the file from the container to your local host, and modify it as below.
```
morganh@host:~$ mkdir docker_build && cd docker_build
morganh@host:~/docker_build$ docker cp tao-api-fixed:/opt/api ./
morganh@host:~/docker_build$ cd api
morganh@host:~/docker_build/api$ vim handlers/actions.py
```

Go to line:779 and change the code from
```shell
if find_trained_weight == []:
    if not ptm_id == "":
        model_dir = f"/shared/users/00000000-0000-0000-0000-000000000000/models/{ptm_id}"
        if job_context.network == "lprnet":
            pretrained_model_file = glob.glob(model_dir+"/*/*.tlt")
        else:
            pretrained_model_file = glob.glob(model_dir+"/*/*.hdf5")
else:
    find_trained_weight.sort(reverse=False)
    trained_weight = find_trained_weight[0]
```
to

```shell
if find_trained_weight == []:
    if not ptm_id == "":
        model_dir = f"/shared/users/00000000-0000-0000-0000-000000000000/models/{ptm_id}"
        pretrained_model_file = []
        pretrained_model_file = glob.glob(model_dir+"/*/*.hdf5") + glob.glob(model_dir+"/*/*.tlt")
        if len(pretrained_model_file) > 1:
            pretrained_model_file = pretrained_model_file[0]

        assert pretrained_model_file != [], "error pretrained_model_file"
else:
    find_trained_weight.sort(reverse=False)
    trained_weight = find_trained_weight[0]
```


Change docker_images.py and change the code
```shell
morganh@host:~/docker_build/api$ vim handlers/docker_images.py
```
Go to line 23 and replace the docker image name from
```shell
"api": os.getenv('IMAGE_API', default='nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api')
```
To

```shell
"api": os.getenv('IMAGE_API', default='nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api-fix')
```

## Generate a new docker
Create a Dockerfile
```shell
morganh@host:~/docker_build/api$ mv Dockerfile Dockerfile_bak
morganh@host:~/docker_build/api$ vim Dockerfile
```

Below is the content of Dockerfile
```shell
morganh@host:~/docker_build/api$ cat Dockerfile
################ BUILD IMAGE #################
FROM nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api
# Copy project files
WORKDIR /opt/api
COPY handlers/actions.py handlers/actions.py
COPY handlers/docker_images.py handlers/docker_images.py
ENV PATH=“/opt/ngccli/ngc-cli:${PATH}”
# Default command
CMD /bin/bash app_start.sh
```

```shell
morganh@host:~/docker_build/api$ docker build . -t nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api-fix
```

## Save the docker to tar file.
```shell
$ docker save -o tao-api.tar nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api-fix
```

Copy the tar file to k8s machine
```shell
$ scp tao-api.tar ip_k8s_machine:/path/to/save
```

## Import the new image
In k8s machines,
```shell
$ sudo ctr -n=k8s.io image import tao-api.tar
```

## Install the new chart
Delete existing tao-toolkit-api pods
```shell
$ helm delete tao-toolkit-api
```

Download latest helm chart.
```shell
$ helm fetch https://helm.ngc.nvidia.com/nvidia/tao/charts/tao-toolkit-api-4.0.2.tgz --username=‘$oauthtoken’ --password=<NGC key>
$ mkdir tao-toolkit-api && tar -zxvf tao-toolkit-api-4.0.2.tgz -C tao-toolkit-api
$ cd tao-toolkit-api/
```

Modify the image name.
```shell
$ vi tao-toolkit-api/values.yaml

# in line 2
From
image: nvcr.io/nvidia/tao/tao-toolkit:4.0.2-api
To
image: nvcr.io/nvidia/tao/tao-toolkit:4.0.0-api-fix

#in line 4
From
imagePullPolicy: Always
To
imagePullPolicy: IfNotPresent
```

## Install latest chart
```shell
$ helm install tao-toolkit-api tao-toolkit-api/ --namespace default
```

Verify the latest code inside the docker
```shell
$ kubectl get pods
$ kubectl exec -it tao-toolkit-api-app-pod-5d4d74c65c-k8zt5 -- /bin/bash
root@tao-toolkit-api-app-pod-5d4d74c65c-k8zt5:/opt/api# apt-get install vim
root@tao-toolkit-api-app-pod-5d4d74c65c-k8zt5:/opt/api# vim handlers/actions.py
```
