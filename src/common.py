from modal import Image, Stub, NetworkFileSystem
from typing import Optional
from pathlib import Path

VOL_MOUNT_PATH = Path("/vol")

vol = NetworkFileSystem.persisted("vol")

WANDB_PROJECT = ""

MODEL_PATH = "/model"

DATA_FILENAME = "J Squared Friendmoon.json"
LOCAL_DATA_DIR = Path("collected_data/conversations")
LOCAL_DATA_PATH = LOCAL_DATA_DIR / DATA_FILENAME
MOUNT_DATA_DIR = Path("/mounted")
VOL_SAMPLES_PATH = VOL_MOUNT_PATH / "samples.json"


def download_models():
    from transformers import LlamaForCausalLM, LlamaTokenizer

    model_name = "openlm-research/open_llama_7b"

    print("Downloading LlamaForCausalLM...")
    model = LlamaForCausalLM.from_pretrained(model_name)
    model.save_pretrained(MODEL_PATH)
    print("Done")

    print("Downloading LlamaTokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MODEL_PATH)
    print("Done")


openllama_image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "accelerate==0.18.0",
        "bitsandbytes==0.37.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "datasets==2.14.6",
        "fire==0.5.0",
        "gradio==3.23.0",
        "peft @ git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08",
        "transformers @ git+https://github.com/huggingface/transformers.git@a92e0ad2e20ef4ce28410b5e05c5d63a5a304e65",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
    )
    .run_function(download_models)
    .pip_install("wandb==0.15.0")
)

stub = Stub(name="doppel-bot", image=openllama_image)

stub.doppel_image = (
    Image.debian_slim()
    .apt_install("wget")
    .apt_install("libpq-dev")
    .pip_install("psycopg2")
)


def generate_prompt(user, input, output=""):
    return f"""You are {user}, employee at a fast-growing startup. Below is an input conversation that takes place in the company's internal Slack. Write a response that appropriately continues the conversation.

### Input:
{input}

### Response:
{output}"""


def user_model_path(user: str, checkpoint: Optional[str] = None) -> Path:
    path = VOL_MOUNT_PATH / user
    if checkpoint:
        path = path / checkpoint
    return path


def get_finished_user(user: str) -> Optional[str]:
    path = VOL_MOUNT_PATH
    for p in path.iterdir():
        # Check if finished fine-tuning.
        if (path / p / "adapter_config.json").exists() and p.name == user:
            return p.name
    return None
