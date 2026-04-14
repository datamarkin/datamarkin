import json


class Project(dict):
    """Wrapper for project DB rows. All columns available as attributes."""

    def __init__(self, data):
        super().__init__(data)
        # Plain columns (from db.py schema)
        self.id = self.get("id")
        self.name = self.get("name")
        self.status = self.get("status")
        self.sort_order = self.get("sort_order")
        self.created_at = self.get("created_at")
        self.updated_at = self.get("updated_at")
        self.type = self.get("type")
        self.train = self.get("train")
        self.model_architecture = self.get("model_architecture")
        self.description = self.get("description")
        # JSON columns — parse and keep dict in sync
        self.labels = json.loads(self["labels"]) if self.get("labels") else []
        self["labels"] = self.labels
        self.configuration = json.loads(self["configuration"]) if self.get("configuration") else {}
        self["configuration"] = self.configuration
        self.augmentation = json.loads(self["augmentation"]) if self.get("augmentation") else {}
        self["augmentation"] = self.augmentation
        self.preprocessing = json.loads(self["preprocessing"]) if self.get("preprocessing") else {}
        self["preprocessing"] = self.preprocessing


class File(dict):
    """Wrapper for file DB rows. All columns available as attributes."""

    def __init__(self, data):
        super().__init__(data)
        # Plain columns
        self.id = self.get("id")
        self.project_id = self.get("project_id")
        self.filename = self.get("filename")
        self.extension = self.get("extension")
        self.width = self.get("width")
        self.height = self.get("height")
        self.filesize = self.get("filesize")
        self.checksum = self.get("checksum")
        self.split = self.get("split")
        self.sort_order = self.get("sort_order")
        self.created_at = self.get("created_at")
        self.updated_at = self.get("updated_at")
        # JSON columns — parse and keep dict in sync
        self.annotations = json.loads(self["annotations"]) if self.get("annotations") else None
        self["annotations"] = self.annotations
        self.analyzed_labels = json.loads(self["analyzed_labels"]) if self.get("analyzed_labels") else []
        self["analyzed_labels"] = self.analyzed_labels
