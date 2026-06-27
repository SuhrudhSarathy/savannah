from enum import StrEnum


class ObservationKey(StrEnum):
    images = "observation.images"
    state = "observation.state"
    gt_actions = "gt_action"
    actions = "action"
    time = "time"
