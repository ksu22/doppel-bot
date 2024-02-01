import os
import time
import re
import json

from fastapi import FastAPI

from modal import asgi_app, Mount

from .common import (
    VOL_MOUNT_PATH,
    LOCAL_DATA_DIR,
    VOL_SAMPLES_PATH,
    MOUNT_DATA_DIR,
    DATA_FILENAME,
    vol,
    stub, get_finished_user,
)
from .inference import OpenLlamaModel
from .finetune import finetune

MAX_INPUT_LENGTH = 512  # characters, not tokens.
# TODO(kevinsu): Remove usage
SHARED_USER = "Josh"


@stub.function(
    image=stub.doppel_image,
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
    network_file_systems={VOL_MOUNT_PATH: vol},
    keep_warm=1,
    mounts=[Mount.from_local_dir(LOCAL_DATA_DIR, remote_path=MOUNT_DATA_DIR)],
)
@asgi_app(label="doppel")
def _asgi_app():
    fastapi_app = FastAPI()

    @fastapi_app.get("/api/generate")
    def generate(prompt: str):
        print(f"Prompt: {prompt}")

        user = SHARED_USER
        user = get_finished_user(user)
        if user is None:
            print("No users trained yet. Run /api/train first.")
            return

        model = OpenLlamaModel.remote(user)
        res = model.generate(
            prompt,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            num_beams=1,
            max_new_tokens=600,
            repetition_penalty=1.2,
        )

        exp = "|".join([f"{u}: " for u in [user]])
        messages = re.split(exp, res)

        print("Generated: ", res, messages)
        return messages

    @fastapi_app.post("/api/train")
    def train():
        # Load the json data from the mounted FS to the Volume to provide access to subsequent functions
        def load_mounted_data(user: str):
            with open(MOUNT_DATA_DIR / DATA_FILENAME, 'r') as file:
                text_list = [line.strip() for line in file]
            data = []
            for idx, response in enumerate(text_list[1:]):
                data.append({"input": text_list[idx], "output": response, "user": user})
            return data

        # TODO(kevinsu): Dont buffer finetune data in memory
        fine_tune_data = load_mounted_data(SHARED_USER)

        with open(VOL_SAMPLES_PATH, 'w') as file:
            json.dump(fine_tune_data, file, indent=2)
        print(fine_tune_data)
        print(os.listdir("/vol"))
        user = SHARED_USER
        user_pipeline.spawn(user)

    @fastapi_app.post("/")
    def hi():
        return "hi!!"

    return fastapi_app


@stub.function(
    image=stub.doppel_image,
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
    network_file_systems={VOL_MOUNT_PATH: vol},
)
def user_pipeline(user: str):
    try:
        t0 = time.time()
        finetune.remote(user)
        print(f"Finished training {user} after {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"Failed to train {user} ({e}). Try again in a bit!")
        raise e
