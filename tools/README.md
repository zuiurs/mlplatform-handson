Generate inputs.

```
# take number of inputs
python3 generate_input.py 5
```

Request with `curl`.

```
SV_IP=<ip>
SV_PORT=<port>
SV_HOST=fmnist.dev.example.com
MODEL_NAME=fmnist
SESSION=<your_token>

curl -H "Host: ${SV_HOST}" -H "Cookie: authservice_session=${SESSION}" http://${SV_IP}:${SV_PORT}/v1/models/${MODEL_NAME}
```

Request with script.

```
export SV_IP=<ip>
export SV_PORT=<port>
export SV_HOST=fmnist.dev.example.com
export MODEL_NAME=fmnist
export KF_USERNAME=<username>
export KF_PASSWORD=<password>

python predict_rest.py 5
```
