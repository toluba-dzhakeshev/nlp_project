from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="./bert/bert_model",  # Path to the directory containing model files (e.g., config.json, model.safetensors)
    repo_id="tolubai/tolubai-bert-test",  # Replace with your Hugging Face username and model name
    repo_type="model"
)

api.upload_folder(
    folder_path="./bert/bert_tokenizer",  # Path to the directory containing model files (e.g., config.json, model.safetensors)
    repo_id="tolubai/tolubai-bert-test",  # Replace with your Hugging Face username and model name
    repo_type="model"
)