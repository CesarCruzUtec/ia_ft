import extra
import gradio as gr
import handlers

with gr.Blocks(theme=gr.themes.Default(), delete_cache=(60, 3600)) as demo:
    gr.Markdown("# Dataset and Training Interface")
    with gr.Tabs():
        with gr.TabItem("Dataset Preparation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Dataset Manager")

                    with gr.Accordion(label="Dataset Instructions", open=True):
                        gr.Markdown("""
                        Use the accordions below to create a new dataset or load an existing one.
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
                        to label and split your dataset.
                        """)

                    with gr.Accordion(label="Create Dataset", open=False):
                        ds_name = gr.Textbox(label="Dataset Name", max_lines=1)
                        ds_create_btn = gr.Button("Create New Dataset", size="md")
                        ds_summary = gr.Textbox(label="Dataset Summary", lines=4)
                        ds_refresh_btn = gr.Button("Refresh Dataset List", size="md")

                    with gr.Accordion(label="Load Existing Dataset", open=False):
                        ds_list = gr.Dropdown(choices=extra.get_dataset_names(), label="Available Datasets")
                        ds_load_btn = gr.Button("Load Selected Dataset", size="md")
                        ds_info = gr.Textbox(label="Dataset Information", lines=4)

                with gr.Column(scale=2):
                    gr.Markdown("## Dataset Preview")
                    ds_preview = gr.Dataset(
                        label="Dataset Preview Area",
                        layout="gallery",
                        components=[gr.Image()],
                        headers="Sample Images from Dataset",
                    )

                ds_create_btn.click(
                    fn=handlers.create_dataset,
                    inputs=[ds_name],
                    outputs=[ds_summary, ds_list],
                )
                ds_refresh_btn.click(
                    fn=handlers.refresh_dataset,
                    inputs=[ds_name],
                    outputs=[ds_summary, ds_list],
                )
                ds_load_btn.click(
                    fn=handlers.load_dataset,
                    inputs=[ds_list],
                    outputs=[ds_info, ds_preview],
                )
        with gr.TabItem("Annotation and Augmentation"):
            gr.Markdown("## Annotate and augment your dataset here.")
            # Add annotation and augmentation components here

        with gr.TabItem("Training"):
            gr.Markdown("## Train your model here.")
            # Add training components here


demo.launch()
