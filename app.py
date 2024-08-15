from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import json
import uvicorn

from nn_train import nn_train
from nn_test import nn_test
from nn_infer import nn_infer
from nn_feature import nn_feature, summarize

from tr_model import tr_train, tr_test, tr_infer
from tr_feature import tr_feature, tr_get_dict, tr_add_dict, tr_remove_dict

SOCKET_BACKEND_URL = 'http://text-malicious-service:13013'
PORT = 13015

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncClient(logger=True, engineio_logger=True)



@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to Core server API!"}




@app.on_event('startup')
async def startup():
    await sio.connect(SOCKET_BACKEND_URL)


@sio.event
async def connect():
    print('connection established')

async def start_summarizing(data):
    response = await summarize(data["feature_name"])
    await sio.emit(
        f'receive_summarizing_process',
        json.dumps({
            "response": response,
            "labId": data["labId"],
        })
    )
    await sio.sleep(0.1)

async def start_feature_nn(data):
    response = await nn_feature(data["data_dir"], data["feature_name"], data["labId"])
    await sio.emit('receive_feature_nn_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "trainId": data["trainId"]
    }))
    await sio.sleep(0.1)


async def start_train_nn(data):
    response = await nn_train(data["data_dir"], data["learning_rate"], data["epochs"], data["batch_size"], data["val_size"], data["model_type"], data["labId"])
    await sio.emit('receive_train_nn_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "trainId": data["trainId"]
    }))
    await sio.sleep(0.1)

async def start_test_nn(data):
    response = await nn_test(data["data_dir"], data["model_type"], data["ckpt_number"], data["labId"])
    await sio.emit('receive_test_nn_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "testId": data["testId"]
    }))
    await sio.sleep(0.1)

async def start_infer_nn(data):
    response = await nn_infer(data["text"], data["model_type"], data["ckpt_number"], data["labId"])
    await sio.emit(f'receive_infer_nn_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "inferId": data["inferId"]
    }))
    await sio.sleep(0.1)


async def start_get_dict_tr(data):
    response = await tr_get_dict(data["data_dir"], data["min_frequency"], data["max_frequency"], data["lower_case"], data["labId"])
    await sio.emit('receive_get_dict_tr_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "trainId": data["trainId"]
    }))
    await sio.sleep(0.1)

async def start_add_dict_tr(data):
    response = await tr_add_dict(data["word"], data["labId"])
    await sio.emit('receive_add_dict_tr_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "trainId": data["trainId"]
    }))
    await sio.sleep(0.1)

async def start_remove_dict_tr(data):
    response = await tr_remove_dict(data["word"], data["labId"])
    await sio.emit('receive_remove_dict_tr_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "trainId": data["trainId"]
    }))
    await sio.sleep(0.1)

async def start_feature_tr(data):
    response = await tr_feature(data["data_dir"], data["feature_name"], data["labId"])
    await sio.emit('receive_feature_tr_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "trainId": data["trainId"]
    }))
    await sio.sleep(0.1)

async def start_train_tr(data):
    response = await tr_train(data["data_dir"], data["model_type"], data["feature_name"], data["val_size"],  data["labId"], **data["kargs"])
    await sio.emit('receive_train_tr_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "trainId": data["trainId"]
    }))
    await sio.sleep(0.1)

async def start_test_tr(data):
    response = await tr_test(data["data_dir"], data["model_type"], data["feature_name"], data["labId"])
    await sio.emit('receive_test_tr_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "testId": data["testId"]
    }))
    await sio.sleep(0.1)

async def start_infer_tr(data):
    response = await tr_infer(data["text"], data["model_type"], data["feature_name"], data["labId"])
    await sio.emit(f'receive_infer_tr_process', json.dumps({
        "response": response,
        "labId": data["labId"],
        "inferId": data["inferId"]
    }))
    await sio.sleep(0.1)

@sio.on("start_summarizing")
async def start_summarizing_listener(data):
    Thread(target=await start_summarizing(data)).start()

@sio.on("start_feature_nn")
async def start_feature_nn_listener(data):
    Thread(target=await start_feature_nn(data)).start()

@sio.on("start_train_nn")
async def start_train_nn_listener(data):
    Thread(target=await start_train_nn(data)).start()

@sio.on("start_test_nn")
async def start_test_nn_listener(data):
    Thread(target=await start_test_nn(data)).start()

@sio.on("start_infer_nn")
async def start_infer_nn_listener(data):
    Thread(target=await start_infer_nn(data)).start()





@sio.on("start_get_dict_tr")
async def start_get_dict_tr_listener(data):
    Thread(target=await start_get_dict_tr(data)).start()

@sio.on("start_add_dict_tr")
async def start_add_dict_tr_listener(data):
    Thread(target=await start_add_dict_tr(data)).start()

@sio.on("start_remove_dict_tr")
async def start_remove_dict_tr_listener(data):
    Thread(target=await start_remove_dict_tr(data)).start()

@sio.on("start_feature_tr")
async def start_feature_tr_listener(data):
    Thread(target=await start_feature_tr(data)).start()

@sio.on("start_train_tr")
async def start_train_tr_listener(data):
    Thread(target=await start_train_tr(data)).start()

@sio.on("start_test_tr")
async def start_test_tr_listener(data):
    Thread(target=await start_test_tr(data)).start()

@sio.on("start_infer_tr")
async def start_infer_tr_listener(data):
    Thread(target=await start_infer_tr(data)).start()


@sio.event
async def disconnect():
    print('disconnected from server')

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True,
                debug=True, ws_ping_interval=99999999, ws_ping_timeout=99999999)
