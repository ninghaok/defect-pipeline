from pathlib import Path
import tempfile
import unittest
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from detected_pipeline.roi import WhiteMaskRoiCropper, write_image
from detected_pipeline.masks import external_gt


class RoiCropTests(unittest.TestCase):
    def test_white_roi_crops_image_and_external_gt_consistently(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);roi=np.zeros((10,12),np.uint8);roi[2:9,3:11]=255
            roi_path=root/"roi.png";write_image(roi_path,roi)
            cropper=WhiteMaskRoiCropper(roi_path)
            image=np.zeros((10,12,3),np.uint8);image[:]=30
            external=np.full((10,12),255,np.uint8);external[5:7,6:8]=0
            cropped_image=cropper.crop_image(image);cropped_gt=cropper.crop_external_gt(external)
            self.assertEqual(cropped_image.bbox_xyxy,(3,2,11,9))
            self.assertEqual(cropped_image.image.shape[:2],(7,8))
            self.assertEqual(cropped_gt.image.shape,(7,8))
            self.assertTrue((cropped_gt.image==0).any())
            self.assertTrue((cropped_gt.image==255).any())

    def test_roi_resizes_with_nearest_neighbor(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);roi=np.zeros((5,5),np.uint8);roi[1:4,2:5]=255
            roi_path=root/"roi.png";write_image(roi_path,roi)
            result=WhiteMaskRoiCropper(roi_path).crop_image(np.zeros((10,10,3),np.uint8))
            self.assertGreater(result.image.size,0)
            self.assertTrue(result.roi_mask.any())

    def test_external_mask_reads_chinese_windows_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"底面_593_t.png"
            mask=np.full((8,9),255,np.uint8);mask[2:4,3:6]=0
            write_image(path,mask)
            anomaly=external_gt(path)
            self.assertEqual(int(anomaly.sum()),6)


if __name__=="__main__":unittest.main()
