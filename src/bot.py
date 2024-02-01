import time
import re
import json

from fastapi import FastAPI

from modal import Dict, asgi_app

from common import (
    VOL_MOUNT_PATH,
    LOCAL_DATA_PATH,
    VOL_SAMPLES_PATH,
    vol,
    stub, get_finished_user,
)
from inference import OpenLlamaModel
from finetune import finetune

MAX_INPUT_LENGTH = 512  # characters, not tokens.
# TODO(kevinsu): Remove usage
SHARED_USER = "Josh"


# Load the json data from the local FS to the Volume to provide access to subsequent functions.
def load_local_data(user: str):
    with open(LOCAL_DATA_PATH, 'r') as file:
        text_list = [line.strip() for line in file]
    data = []
    for idx, response in enumerate(text_list[1:]):
        data.append({"input": text_list[idx], "output": response, "user": user})
    return data


# TODO(kevinsu): Dont buffer finetune data in memory
fine_tune_data = load_local_data(SHARED_USER)


@stub.function(
    image=stub.doppel_image,
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
    volumes={VOL_MOUNT_PATH: vol},
    keep_warm=1,
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

        exp = "|".join([f"{u}: " for u, _ in [user]])
        messages = re.split(exp, res)

        print("Generated: ", res, messages)

    @fastapi_app.post("/api/train")
    def train():
        with open(VOL_SAMPLES_PATH, 'w') as file:
            json.dump(fine_tune_data, file, indent=2)
        user = SHARED_USER
        user_pipeline.spawn(user)

    return fastapi_app


@stub.function(
    image=stub.doppel_image,
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
)
def user_pipeline(user: str):
    try:
        vol.persist()
        print(f"Successfully loaded data to persistent volume at {VOL_SAMPLES_PATH}")
        t0 = time.time()

        finetune.call(user, VOL_SAMPLES_PATH)

        print(f"Finished training {user} after {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"Failed to train {user} ({e}). Try again in a bit!")
        raise e
