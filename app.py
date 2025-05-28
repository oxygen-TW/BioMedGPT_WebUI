from shiny import App, ui, render, reactive
from pathlib import Path
import base64
import sys
from PIL import Image
from io import BytesIO
import io
import zipfile

sys.path.append("/home/howardliu/work_space/BiomedGPT/BiomedGPTvisulize/BiomedGPT")
from BiomedGPTvisulize.BiomedGPT.core import predict_caption

global_results: list[dict] = []

# Convert image to base64
def image_to_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

# Load default annotated image
def get_default_annotated_base64() -> str:
    path = Path("/work104/irischen/BiomedGPT/Heatmap.png")
    if path.exists():
        try:
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        except Exception:
            return ""
    return ""

# CSS
custom_css = ui.tags.style("""
    
    html, body {
        background-color: #eef3f7;
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
        margin: 0;
        padding: 0;
        height: 100%;
    }
    .shiny-container {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #ffffff;
        padding: 10px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        box-sizing: border-box;
        z-index: 1000;
    }
    .top-bar img {
        height: 36px;
    }
    .main-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        /* 上下間距一致 */
        padding: 100px 20px 100px 20px;
        box-sizing: border-box;
        width: 100%;
    }
    .content-wrapper {
        width: 100%;
        max-width: 1000px;
        margin: 0 auto;
        box-sizing: border-box;
    }
    .top-block {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        text-align: center;
        width: 100%;
        margin-bottom: 30px;
    }
    .middle-content {
        display: flex;
        justify-content: space-between;
        align-items: stretch;
        gap: 30px;
        width: 100%;
    }
    .upload-box {
        margin: 20px 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .xray-column {
        flex: 0 0 48%;
        box-sizing: border-box;
    }
    .xray-image {
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        max-width: 100%;
        margin: 5px;
        cursor: pointer;
        transition: transform 0.2s ease;
    }
    .xray-image:hover {
        transform: scale(1.02);
    }
    .xai-block, .caption-block {
        background-color: #fff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        flex: 1 1 48%;
        min-width: 320px;
        box-sizing: border-box;
    }
    .xai-block h3 {
        margin-bottom: 16px;
        line-height: 1.4;
    }
    .navigation-block {
        margin-bottom: 16px;
        line-height: 1.4;
    }
    .modal-overlay {
        display: none;
        position: fixed;
        z-index: 2000;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.7);
        justify-content: center;
        align-items: center;
    }
    .modal-overlay.show {
        display: flex !important;
    }
    .modal-overlay img {
        max-width: 90%;
        max-height: 90%;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .arrow {
        font-size: 18px;
        margin: 0 20px;
        cursor: pointer;
        user-select: none;
    }
    .arrow.disabled {
        color: lightgray;
        cursor: default;
    }
    .arrow.enabled {
        color: #666;
    }
    .bottom-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #ffffff;
        text-align: center;
        padding: 14px 20px;
        font-size: 14px;
        color: #555;
        border-top: 1px solid #ccc;
        z-index: 1000;
    }
    .custom-download-btn {
        padding:8px 20px;
        border-radius:999px;
        border:2px solid #00B0C8;
        background:white;
        color:#00B0C8;
        font-weight:bold;
    }
    .custom-download-btn:hover {
        background:#00B0C8;
        color:white;
    }
"""
)

# Shared JS snippets
NAV_TRIGGER_JS = ui.tags.script("""
setTimeout(() => {
    const prev = document.getElementById("arrow-prev");
    const next = document.getElementById("arrow-next");
    if (prev) prev.onclick = () => { if (!prev.classList.contains("disabled")) Shiny.setInputValue("arrow_prev", Math.random()); };
    if (next) next.onclick = () => { if (!next.classList.contains("disabled")) Shiny.setInputValue("arrow_next", Math.random()); };
}, 100);
"""
)

MODAL_JS = ui.tags.script("""
function showModal(src) {
    const modal = document.getElementById('img-modal');
    document.getElementById('modal-img').src = src;
    modal.classList.add('show');
}
function hideModal() {
    const modal = document.getElementById('img-modal');
    modal.classList.remove('show');
}
"""
)

# Helper functions
def make_top_bar(logos: list[str]):
    imgs = [
        ui.tags.img(src="data:image/jpeg;base64," + image_to_base64(p), style="height:36px; margin-right:12px;")
        for p in logos
    ]
    return ui.div({"class": "top-bar"}, ui.div(*imgs, style="display:flex; align-items:center;"))


def make_download_buttons(labels: list[str]):
    ids = [label.replace(" ", "_").lower() for label in labels]

    btns = [
        ui.download_button(
            id=id_,
            label=label,
            class_="custom-download-btn"
        )
        for id_, label in zip(ids, labels)
    ]

    return ui.div(
        *btns,
        style="display:flex; justify-content:center; gap:16px; margin-top:20px;"
    )

# UI definition
app_ui = ui.page_fluid(
    custom_css,
    ui.div({"class": "shiny-container"},
        make_top_bar([
            "/work104/irischen/BiomedGPT/logo.jfif",
            "/work104/irischen/BiomedGPT/logo2.jfif"
        ]),
        ui.div({"class": "main-content"},
            ui.div({"class": "content-wrapper"},
                ui.div({"class": "top-block"},
                    ui.h2("Upload X-ray Images"),
                    ui.div({"class": "upload-box"},
                        ui.input_file("xray_files", "", multiple=True, accept=[".png", ".jpg", ".jpeg"])
                    )
                ),
                ui.output_ui("results")
            )
        )
    ),
    ui.div("AI-assisted Radiological Imaging Diagnostic Platform @ NTU", {"class": "bottom-bar"})
)

def save_report(report_text: str, filename: str):
    """Save the report text to a file."""
    try:
        with open(filename, "w") as f:
            f.write(report_text)
        return True
    except Exception as e:
        print(f"Error saving report: {e}")
        return False
    
# Server logic
def server(input, output, session):
    annotated_b64 = get_default_annotated_base64()
    current_index = reactive.Value(0)
    uploaded_files = reactive.Value([])

    @reactive.effect
    @reactive.event(input.xray_files)
    def _():
        current_index.set(0)
        uploaded_files.set(input.xray_files())

    @output
    @render.ui
    def navigation():
        files = uploaded_files.get()
        if not files:
            return None
        total = len(files)
        idx = current_index.get()
        left_class = "arrow disabled" if idx == 0 else "arrow enabled"
        right_class = "arrow disabled" if idx == total - 1 else "arrow enabled"
        return ui.div(
            NAV_TRIGGER_JS,
            ui.div({"class": "navigation-block"},
                ui.div(f"Image {idx + 1} of {total}", style="text-align:center; font-size:18px; font-weight:normal; color:#666; margin-bottom:8px; line-height:1.4;"),
                ui.div(
                    ui.tags.span("← Previous", id="arrow-prev", class_=left_class, style="margin-right:20px;"),
                    ui.span("|", style="font-size:18px; color:#666; margin: 0 8px;"),
                    ui.tags.span("Next →", id="arrow-next", class_=right_class, style="margin-left:20px;"),
                    style="text-align:center; font-size:18px; color:#666;"
                )
            )
        )

    @reactive.effect
    @reactive.event(input.arrow_prev)
    def _():
        if current_index.get() > 0:
            current_index.set(current_index.get() - 1)

    @reactive.effect
    @reactive.event(input.arrow_next)
    def _():
        if current_index.get() < len(uploaded_files.get()) - 1:
            current_index.set(current_index.get() + 1)

    @output
    @render.ui
    def results():
        files = uploaded_files.get()
        if not files:
            return ui.p(" ")
        idx = current_index.get()
        datapath = files[idx]["datapath"]

        # 定義 prompt
        prompt_ = "what does the image describe?"

        # 在 Shiny 觸發 busy/idle 時，游標會自動變換
        # 呼叫 LLM 分析
        report_text, heatmap_img = predict_caption(datapath, prompt_)

        heatmap_b64 = image_to_base64(heatmap_img) 
        orig_b64 = image_to_base64(datapath)

        result = {
        "caption": report_text,
        "img": orig_b64,
        "heatmap": heatmap_b64,
        }
        global_results.append(result)

        return ui.div(
            MODAL_JS,
            ui.tags.div({"id": "img-modal", "class": "modal-overlay", "onclick": "hideModal()"},
                ui.tags.img(id="modal-img")
            ),
            ui.hr(),
            ui.div({"class": "middle-content"},
                ui.div({"class": "xai-block"},
                    ui.h3("Explainable AI (XAI)"),
                    ui.output_ui("navigation"),
                    ui.div({"style": "display:flex; gap:20px; justify-content:center; align-items:flex-start;"},
                        ui.div({"class": "xray-column"},
                            ui.div("Original Image", style="font-weight:600; margin-bottom:8px; text-align:center;"),
                            ui.img(src=f"data:image/png;base64,{orig_b64}", class_="xray-image", onclick="showModal(this.src)")
                        ),
                        ui.div({"class": "xray-column"},
                            ui.div("Attention Map", style="font-weight:600; margin-bottom:8px; text-align:center;"),
                            ui.img(src=f"data:image/png;base64,{heatmap_b64}", class_="xray-image", onclick="showModal(this.src)") if annotated_b64 else ui.p("Annotated image not available")
                        )
                    )
                ),
                ui.div({"class": "caption-block"},
                    ui.div({"style": "display:flex; flex-direction:column; height:100%; justify-content:space-between;"},
                        ui.div(
                            ui.h3("Caption"),
                            ui.p(
                                report_text,
                                style="line-height:1.6; color:#444;"
                            )
                        ),
                        make_download_buttons(["Download This Report", "Download All Reports"])    
                    )
                )
            )
        )
    def generate_all_results(uploaded_files, prompt="what does the image describe?"):
        global global_results
        global_results.clear()  # 每次重新產生

        files = uploaded_files.get()
        if not files:
            return

        for idx, file in enumerate(files):
            datapath = file["datapath"]
            report_text, heatmap_img = predict_caption(datapath, prompt)  # 你有 cache 就不怕慢

            heatmap_b64 = image_to_base64(heatmap_img)
            orig_b64 = image_to_base64(datapath)

            result = {
                "caption": report_text,
                "img": orig_b64,
                "heatmap": heatmap_b64,
            }
            global_results.append(result)


    @render.download(filename="results.zip")
    def download_all_reports():
        generate_all_results(uploaded_files)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for idx, res in enumerate(global_results):
                if "caption" not in res or "img" not in res or "heatmap" not in res:
                    continue
                zf.writestr(f"caption_{idx+1}.txt", res["caption"])
                zf.writestr(f"image_{idx+1}.png", base64.b64decode(res["img"]))
                zf.writestr(f"heatmap_{idx+1}.png", base64.b64decode(res["heatmap"]))
        buf.seek(0)
        return buf  # 或 yield buf

    @render.download(filename="result.zip")
    def download_this_report():
        idx = current_index.get()
        res = global_results[idx]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("caption.txt", res["caption"])
            # 圖片：base64 decode 回 bytes 再寫進 zip
            zf.writestr("image.png", base64.b64decode(res["img"]))
            zf.writestr("heatmap.png", base64.b64decode(res["heatmap"]))
        buf.seek(0)
        return buf

# App instantiation
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1028)