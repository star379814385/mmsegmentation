from mmcv.utils import Config
import argparse
from pathlib import Path



if __name__ == "__main__":
    cfg_path = r"D:\code\mmsegmentation-0.30.0\configs\unet\fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py"
    # save_path = None
    # save_path = str(Path(__file__).parent / "full_configs" / Path(cfg_path).name)
    save_path = str(Path("D:\code\mmsegmentation-0.30.0") / "full_configs" / Path(cfg_path).name)
    cfg: Config = Config.fromfile(cfg_path)
    
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True)
    cfg.dump(save_path)
    