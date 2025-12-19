from huggingface_hub import hf_hub_download

REPO_ID = "amirsh1376/FeelAnyForce"
FILENAME = "checkpoint_v1.pth.tar"

def download_pretrained_weights(local_files_only=False):
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_files_only=local_files_only,
    )