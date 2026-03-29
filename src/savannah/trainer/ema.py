import torch
import copy


class EMA:
    """
    Exponential Moving Average utility for model weights.
    A shadow copy of the model is maintained and updated after each optimizer step.
    """

    def __init__(self, model, decay=0.9999):
        # Initialize EMA weights from the current model weights
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(self.model)
        self.shadow.train()  # Set to train mode for compatibility with UNet

    @torch.no_grad()
    def update(self):
        """Update the shadow weights using the current model weights."""
        model_params = dict(self.model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name, param in model_params.items():
            shadow_params[name].data = shadow_params[
                name
            ].data * self.decay + param.data * (1.0 - self.decay)

    def copy_to(self, target_model):
        """Copy EMA weights to a target model (e.g., for checkpointing or inference)."""
        target_params = dict(target_model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())
        for name, param in target_params.items():
            param.data.copy_(shadow_params[name].data)
