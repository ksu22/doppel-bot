import time
import re

from fastapi import FastAPI

from modal import Dict, Secret, asgi_app

from .common import (
    MULTI_WORKSPACE_SLACK_APP,
    VOL_MOUNT_PATH,
    output_vol,
    stub, get_finished_user,
)
from .inference import OpenLlamaModel
from .samples import create_samples
from .finetune import finetune

# Ephemeral caches
stub.users_cache = Dict.new()
stub.self_cache = Dict.new()

MAX_INPUT_LENGTH = 512  # characters, not tokens.
# TODO(kevinsu): Remove usage
SHARED_USER = "Josh"

@stub.function(
    image=stub.slack_image,
    secrets=[
        Secret.from_name("slack-finetune-secret"),
        # TODO: Modal should support optional secrets.
        *([Secret.from_name("neon-secret")] if MULTI_WORKSPACE_SLACK_APP else []),
    ],
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
    network_file_systems={VOL_MOUNT_PATH: output_vol},
    cloud="gcp",
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
            print("No users trained yet. Run /api/doppel first.")
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
        user = SHARED_USER
        user_pipeline.spawn(user)

    return fastapi_app


@stub.function(
    image=stub.slack_image,
    # TODO: Modal should support optional secrets.
    secret=Secret.from_name("neon-secret") if MULTI_WORKSPACE_SLACK_APP else None,
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
)
def user_pipeline(user: str):
    try:
        file_path = "collected_data/conversations/J Squared Friendmoon.json"
        print(f"Loading samples from `{file_path}`")
        # TODO(kevinsu): Support different users
        user = SHARED_USER
        samples_path = create_samples(file_path, user)
        t0 = time.time()

        finetune.call(user, samples_path)

        print(f"Finished training {user} after {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"Failed to train {user} ({e}). Try again in a bit!")
        raise e
