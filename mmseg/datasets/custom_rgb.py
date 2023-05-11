from .custom import CustomDataset, DATASETS
from .pipelines import LoadAnnotationsRGB


@DATASETS.register_module()
class CustomRGBDataset(CustomDataset):
    def __init__(self, **kwargs):
        super(CustomRGBDataset, self).__init__(**kwargs)
        gt_seg_map_loader_cfg = kwargs.get("gt_seg_map_loader_cfg", None)
        self.gt_seg_map_loader = (
            LoadAnnotationsRGB(palette=self.PALETTE)
            if gt_seg_map_loader_cfg is None
            else LoadAnnotationsRGB(palette=self.PALETTE, **gt_seg_map_loader_cfg)
        )
