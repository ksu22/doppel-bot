import json


def create_samples(user: str, file_path: str):
    with open(file_path, 'r') as file:
        text_list = [line.strip() for line in file]
    fine_tune_data = []
    for idx, response in enumerate(text_list[1:]):
        fine_tune_data.append({"input": text_list[idx], "output": response, "user": user})

    samples_path = "samples.json"
    with open(samples_path, "w") as f:
        json.dump(fine_tune_data, f, indent=2)
    print(f"Finished scrape for {user} ({len(fine_tune_data)} samples found).")
    return samples_path
