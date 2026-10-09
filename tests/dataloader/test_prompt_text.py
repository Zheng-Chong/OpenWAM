from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.utils.prompt_text import has_asset_id, normalize_prompt


def test_normalize_prompt():
    cases = {
        "hit-ball-with-gripper-to-score-goal": "Hit ball with gripper to score goal.",
        "fold_mat": "Fold mat.",
        "Galbot_G1_Pick_up_trash_on_the_table_new1": "Pick up trash on the table.",
        "Galbot_G1_Put_on_a_garbage_bag_2CopyCopy": "Put on a garbage bag.",
        "Galbot_G1_Pick_up_trash_on_the_table_0501_02·": "Pick up trash on the table.",
        "Close the microwave_gr with left arm": "Close the microwave with left arm.",
        "Wipe it with the brown_cleaning_cloth": "Wipe it with the brown cleaning cloth.",
        "Take the  bread and steak from the rotating tray v3": "Take the bread and steak from the rotating tray.",
        "close the microwave door, please": "Close the microwave door.",
        "please move the bottle": "Move the bottle.",
        "Please put the glass into the glass box carefully.": "Put the glass into the glass box carefully.",
        "Pick_up the cup": "Pick up the cup.",  # "_up" is not an asset code
        "Pick up the cup.": "Pick up the cup.",  # already clean: unchanged
        "Sort the cups with the same-color plates.": "Sort the cups with the same-color plates.",  # hyphens kept in sentences
    }
    for raw, want in cases.items():
        assert normalize_prompt(raw) == want, (raw, normalize_prompt(raw))
    assert normalize_prompt(normalize_prompt("fold_mat")) == "Fold mat."  # idempotent


def test_has_asset_id():
    for t in ("Close the microwave_gr with left arm", "Galbot_G1_Clean_the_sink_1", "real three fold v2"):
        assert has_asset_id(t), t
    for t in ("Pick up the new trash bag.", "Choose the 30 dollars gift box", "Place the T-shirt in the box."):
        assert not has_asset_id(t), t


def test_reader_config_key():
    assert "normalize_prompt" in LeRobotV3Reader.CONFIG_KEYS
