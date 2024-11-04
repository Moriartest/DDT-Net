"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 July 7, 2020
"""
from pathlib import Path

project_root = Path(__file__).parent
dataset_root = Path(r"E:/Projects/Python/DataSet/")
# dataset_root = Path(r"F:/DataSet")
dataset_paths = {
    # Specify where are the roots of the datasets.
    'FR': dataset_root / "FantasticReality_v1",
    'IMD2020': dataset_root / "IMD2020",
    'DIS25k': dataset_root / "DIS25k",
    'CASIA': dataset_root / "CASIA",
    'CASIA2': dataset_root / "CASIA2",
    'defacto-splicing': dataset_root / "defacto-splicing",
    'defacto-copymove': dataset_root / "defacto-copymove",
    'defacto-inpainting': dataset_root / "defacto-inpainting",
    'MS-COCO': dataset_root / "MS-COCO",
    'nist16': dataset_root / "nist16/NC2016_Test0613",
    'Columbia': dataset_root / "Columbia",
    'tampCOCO': dataset_root / "tampCOCO",
    'COVERAGE': dataset_root / "COVERAGE",
}
