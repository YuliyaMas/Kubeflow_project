"""The full pipeline Kubeflow"""

import kfp
from kfp import dsl

# Load each component from the YAML files
preprocess_op = kfp.components.load_component_from_file("components/preprocess/preprocess_component.yaml")
train_op = kfp.components.load_component_from_file("components/train/train_component.yaml")
eval_op = kfp.components.load_component_from_file("components/evaluate/eval_component.yaml")
generate_op = kfp.components.load_component_from_file("components/predict/predict_component.yaml")

@dsl.pipeline(
    name="Author Classification and Text Generation Pipeline",
    description="Pipeline that preprocesses data, trains a model, evaluates it, and generates text."
)
def author_classification_pipeline(input_path: str, output_path: str):
    # Preprocessing step
    preprocess_step = preprocess_op(
        input_path=input_path,
        output_data_path=f"{output_path}/data",
        output_metadata_path=f"{output_path}/metadata"
    )

    # Training step
    train_step = train_op(
        data_path=preprocess_step.outputs["output_data_path"],
        metadata_path=preprocess_step.outputs["output_metadata_path"],
        output_model_path=f"{output_path}/model"
    )

    # Evaluation step
    eval_step = eval_op(
        data_path=preprocess_step.outputs["output_data_path"],
        model_path=train_step.outputs["output_model_path"]
    )

    # Prediction and text generation step
    generate_step = generate_op(
        model_path=train_step.outputs["output_model_path"],
        metadata_path=preprocess_step.outputs["output_metadata_path"],
        input_text="Sample input text for generation"
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(author_classification_pipeline, "author_classification_pipeline.yaml")
