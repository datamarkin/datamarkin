"""AgentUI tool definitions for Datamarkin.

Registered at app startup via agentui.register_tool().
"""

import json
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from agentui.core.tool import Port, PortType, Tool, ToolOutput

try:
    from routes.predict_route import load_model_from_training
    PREDICT_AVAILABLE = True
except ImportError:
    PREDICT_AVAILABLE = False


class DatamarkinLocalModel(Tool):
    """Run inference using a locally trained Datamarkin model."""

    @property
    def tool_type(self) -> str:
        return "DatamarkinLocalModel"

    @property
    def input_ports(self) -> Dict[str, Port]:
        return {"image": Port("image", PortType.IMAGE, "Input image")}

    @property
    def output_ports(self) -> Dict[str, Port]:
        return {"detections": Port("detections", PortType.DETECTIONS, "Detected objects")}

    def get_parameter_options(self) -> dict:
        """Populate the training_id dropdown from the live database."""
        from queries import get_done_trainings, get_project_by_id
        trainings = get_done_trainings()
        options = []
        for t in trainings:
            config = json.loads(t.get("config") or "{}")
            project = get_project_by_id(t["project_id"]) if t.get("project_id") else None
            project_name = project["name"] if project else config.get("name", "Built-in Model")
            arch = config.get("model_architecture", "rfdetr")
            if arch == "detectron2":
                label = f"{project_name} ({config.get('variant', 'detectron2')})"
            else:
                model_size = config.get("model_size", "base")
                metrics = json.loads(t.get("metrics") or "{}")
                map_val = metrics.get("mAP50", "")
                label = f"{project_name} ({model_size})"
                if map_val:
                    label += f" — mAP50: {map_val:.2f}"
            options.append({"value": t["id"], "label": label})
        return {
            "training_id": {
                "type": "select",
                "options": options,
            }
        }

    def process(self) -> bool:
        if not PREDICT_AVAILABLE:
            print("DatamarkinLocalModel: predict route not available")
            return False
        if "image" not in self.inputs:
            print("DatamarkinLocalModel: no input image")
            return False

        training_id = self.parameters.get("training_id", "").strip()
        if not training_id:
            print("DatamarkinLocalModel: no training_id selected")
            return False

        from queries import get_training
        training = get_training(training_id)
        if not training or training["status"] != "done":
            print(f"DatamarkinLocalModel: training {training_id!r} not found or not complete")
            return False

        training["config"] = json.loads(training.get("config") or "{}")

        try:
            pil_image = self.inputs["image"].data
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            model = load_model_from_training(training)
            detections = model.predict(cv2_image)
            self.outputs["detections"] = ToolOutput(detections, PortType.DETECTIONS)
            return True

        except Exception as e:
            import traceback
            print(f"DatamarkinLocalModel error: {e}")
            traceback.print_exc()
            return False


METADATA = {
    "name": "My Trained Model",
    "category": "My Models",
    "description": "Run inference using your locally trained Datamarkin models.",
    "parameters": {
        "training_id": "",
    },
}
