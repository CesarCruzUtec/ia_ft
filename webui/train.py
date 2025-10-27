import gradio as gr
import handlers
from gradio_image_annotation import image_annotator

with gr.Blocks(theme=gr.themes.Default(), delete_cache=(60, 3600)) as demo:
    gr.Markdown("# Dataset and Training Interface")
    with gr.Tabs(selected="annotate_augment"):
        with gr.Tab("Dataset Preparation", id="dataset_prep"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Dataset Manager")

                    gr.Markdown("""
                    Use the controls below to create a new dataset or load an existing one.
                    If you create a new dataset it will create a folder in the `datasets/` directory with
                    the following structure:
                    ```
                    datasets/
                    └── <dataset_name>/
                        ├── originals/
                        ├── images/
                        │   ├── train/
                        │   ├── val/
                        │   └── test/
                        ├── labels/
                        │   ├── train/
                        │   ├── val/
                        │   └── test/
                        ├── data.yaml
                        └── README.md
                    ```
                    You can then add your data to the `originals/` folder and use the annotation tools
                    to label and split your dataset. Don't forget to refresh the dataset info after making changes.
                    """)

                    ds_name = gr.Textbox(label="Dataset Name", max_lines=1)
                    ds_create_btn = gr.Button("Create New Dataset", size="md")

                    ds_list = gr.Dropdown(choices=handlers.get_dataset_names(), label="Available Datasets")

                    ds_refresh_btn = gr.Button("Load or Refresh Dataset", size="md")

                with gr.Column(scale=3):
                    gr.Markdown("## Dataset Preview")
                    ds_preview = gr.Gallery(
                        label="Dataset Preview Area",
                        file_types=["image"],
                        columns=5,
                    )

                    ds_summary = gr.Markdown(value="Info will be displayed here", container=True)

                ds_create_btn.click(
                    fn=handlers.create_dataset,
                    inputs=[ds_name],
                    outputs=[ds_list],
                )

                ds_refresh_btn.click(
                    fn=handlers.refresh_dataset,
                    inputs=[ds_list],
                    outputs=[ds_summary, ds_preview],
                )

                ds_preview.select(
                    fn=handlers.select_image,
                    inputs=[ds_list],
                    outputs=[ds_summary],
                )

        with gr.Tab("Annotation and Augmentation", id="annotate_augment"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## Annotation Tools")

                    with gr.Row():
                        ds_list2 = gr.Dropdown(
                            choices=handlers.get_dataset_names(), label="Select Dataset for Annotation"
                        )

                    annotator = image_annotator(
                        boxes_alpha=0.5,
                        label_list=["class1", "class2", "class3"],
                        label="Image Annotation Area",
                        sources=None,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("## Augmentation Tools")

        with gr.Tab("Training", id="training"):
            gr.Markdown("## Train your model here.")
            # Add training components here


demo.launch()
