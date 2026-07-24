from transformers import PretrainedConfig


class VisionConfig(PretrainedConfig):
    model_type = "pbsbench_vision"

    def __init__(self, model_name="dinobloom_vitl14", embed_dim=1024, weights_path=None,
                 pretrained=True, instance_norm=False, identity=False, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.pretrained = pretrained
        self.instance_norm = instance_norm
        self.weights_path = weights_path
        self.identity = identity


class CellRepresentationConfig(PretrainedConfig):
    model_type = "pbsbench_cell_representation"

    def __init__(self, vision=None, blip2_model_type="pretrain_vitL", num_query_tokens=32,
                 freeze_vision=True, itm_loss_weight=1.0, itc_loss_weight=1.0,
                 lm_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.vision = VisionConfig(**vision) if isinstance(vision, dict) else (vision or VisionConfig())
        self.blip2_model_type = blip2_model_type
        self.num_query_tokens = num_query_tokens
        self.freeze_vision = freeze_vision
        self.itm_loss_weight = itm_loss_weight
        self.itc_loss_weight = itc_loss_weight
        self.lm_loss_weight = lm_loss_weight


class CellPatchAlignmentConfig(PretrainedConfig):
    model_type = "pbsbench_cell_patch_alignment"

    def __init__(self, cell=None, patch_vision=None, patch_hidden_size=768,
                 patch_num_latents=32, patch_depth=2, patch_heads=8,
                 projection_dim=256, freeze_cell_encoder=True,
                 global_loss_weight=1.0, local_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.cell = CellRepresentationConfig(**cell) if isinstance(cell, dict) else (cell or CellRepresentationConfig())
        self.patch_vision = (
            VisionConfig(**patch_vision)
            if isinstance(patch_vision, dict)
            else (patch_vision or VisionConfig())
        )
        self.patch_hidden_size = patch_hidden_size
        self.patch_num_latents = patch_num_latents
        self.patch_depth = patch_depth
        self.patch_heads = patch_heads
        self.projection_dim = projection_dim
        self.freeze_cell_encoder = freeze_cell_encoder
        self.global_loss_weight = global_loss_weight
        self.local_loss_weight = local_loss_weight
