from flask import render_template

from sam3_backend.status import get_sam_status
from sam3_backend.downloader import VARIANT_URLS
from config import SAM_MODELS_DIR


def settings_page_route():
    status = get_sam_status(SAM_MODELS_DIR)
    variants = list(VARIANT_URLS.keys())
    return render_template(
        "settings.html",
        active_tab="settings",
        sam_status=status,
        sam_variants=variants,
    )
