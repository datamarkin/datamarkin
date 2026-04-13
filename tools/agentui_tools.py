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
    from mozo.manager import ModelManager
    MOZO_AVAILABLE = True
except ImportError:
    MOZO_AVAILABLE = False


class DatamarkinLocalModel(Tool):
    """Run inference using a locally trained Datamarkin model."""

    _model_manager: Optional[object] = None

    def __init__(self, tool_id=None, **kwargs):
        super().__init__(tool_id, **kwargs)
        if MOZO_AVAILABLE and DatamarkinLocalModel._model_manager is None:
            DatamarkinLocalModel._model_manager = ModelManager()

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
            project = get_project_by_id(t["project_id"])
            project_name = project["name"] if project else "Unknown"
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
        if not MOZO_AVAILABLE:
            print("DatamarkinLocalModel: mozo not installed")
            return False
        if "image" not in self.inputs:
            print("DatamarkinLocalModel: no input image")
            return False

        training_id = self.parameters.get("training_id", "").strip()
        if not training_id:
            print("DatamarkinLocalModel: no training_id selected")
            return False

        confidence_threshold = float(self.parameters.get("confidence_threshold", 0.5))

        from queries import get_done_trainings
        trainings = get_done_trainings()
        training = next((t for t in trainings if t["id"] == training_id), None)
        if not training:
            print(f"DatamarkinLocalModel: training {training_id!r} not found or not complete")
            return False

        config = json.loads(training.get("config") or "{}")
        class_names = [label["name"] for label in config.get("labels", [])]

        try:
            pil_image = self.inputs["image"].data
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            model = DatamarkinLocalModel._model_manager.get_model(
                "rfdetr",
                training_id,
                checkpoint_path=training["model_path"],
                model_size=config.get("model_size", "base"),
                project_type=config.get("project_type", "detection"),
                resolution=config.get("resolution", 560),
                class_names=class_names,
            )

            detections = model.predict(cv2_image, threshold=confidence_threshold)
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
        "confidence_threshold": 0.5,
    },
}
