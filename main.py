import modal

app = modal.App("grade-vs-ppo")

image = modal.Image.debian_slim("3.12").uv_sync().add_local_python_source("training_grade").add_local_python_source("analysis_script")
vol = modal.Volume.from_name("grade-vs-ppo-data", create_if_missing=True)
VOL_DIR = "/data"
OUTPUT_DIR = f"{VOL_DIR}/results"
HOURS= 60*60
@app.function(image=image, gpu="A100", volumes={
    VOL_DIR: vol,
}, timeout=15*HOURS, secrets=[modal.Secret.from_name("wandb-secret")])
def train_reward_model():
    from training_grade import main
    from analysis_script import main as analysis_script_main
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main(output_dir=OUTPUT_DIR)
    analysis_script_main(results_dir=OUTPUT_DIR, eval_every=100)


@app.local_entrypoint()
def local():
    train_reward_model.remote()

