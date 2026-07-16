# Model artifacts

KubeTag loads a frozen Hugging Face sequence-classification model from this directory.
The production bundle contains:

```text
COMPLETED
final_model/
label_schema.json
model_manifest.json
thresholds.json
tokenizer/
```

Large model files are intentionally excluded from Git. Set `KUBETAG_MODEL_REPOSITORY`
and run `kubetag-download-model`, or place the complete bundle here manually.
