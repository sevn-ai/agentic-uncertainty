from minisweagent.utils.log import get_logger
from pathlib import Path
import json

"""Merge multiple predictions into a single file."""


logger = get_logger("merge", emoji="➕")


def merge_predictions(directories: list[Path], output: Path | None = None) -> None:
    """Merge predictions found in `directories` into a single JSON file.

    Args:
        directory: Directory containing predictions.
        output: Output file. If not provided, the merged predictions will be
            written to `directory/preds.json`.
    """
    preds = []
    for directory in directories:
        new = list(directory.rglob("*.pred"))
        preds.extend(new)
        logger.debug("Found %d predictions in %s", len(new), directory)
    logger.info("Found %d predictions", len(preds))
    if not preds:
        logger.warning("No predictions found in %s", directory)
        return
    if output is None:
        output = directories[0] / "preds.json"
    data = {}
    for pred in preds:
        _data = json.loads(pred.read_text())
        instance_id = _data["instance_id"]
        if "model_patch" not in _data:
            logger.warning(
                "Prediction %s does not contain a model patch. SKIPPING", pred
            )
            continue
        # Ensure model_patch is a string
        _data["model_patch"] = (
            str(_data["model_patch"]) if _data["model_patch"] is not None else ""
        )
        if instance_id in data:
            msg = f"Duplicate instance ID found: {instance_id}"
            raise ValueError(msg)
        data[instance_id] = _data
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=4))
    logger.info("Wrote merged predictions to %s", output)
