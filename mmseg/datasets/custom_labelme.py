from .custom import CustomDataset, DATASETS
from .pipelines import LoadAnnotationsByLabelme


@DATASETS.register_module()
class CustomLabelmeDataset(CustomDataset):
    def __init__(self, thickness=-1, **kwargs):
        super(CustomLabelmeDataset, self).__init__(**kwargs)
        self.gt_seg_map_loader = LoadAnnotationsByLabelme(list(self.CLASSES)[1:], thickness)
