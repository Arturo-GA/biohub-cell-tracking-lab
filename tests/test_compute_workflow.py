import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from event_stages import device_policy
from import_cpu_inputs import import_package
from launch_cpu_notebook import validate
from biohub_lab.event_portable import signature


class ComputeWorkflowTests(unittest.TestCase):
    def test_remote_cuda_and_colab_are_rejected(self):
        with patch.dict(os.environ,{'KAGGLE_KERNEL_RUN_TYPE':'Batch'}):
            with self.assertRaisesRegex(RuntimeError,'Kaggle GPU'):device_policy('cuda')
            device_policy('cpu')
        with patch.dict(os.environ,{'COLAB_RELEASE_TAG':'test'}):
            with self.assertRaisesRegex(RuntimeError,'laptop'):device_policy('cpu')

    def test_cpu_launcher_rejects_accelerator_or_public_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);good=dict(is_private=True,enable_gpu=False,enable_tpu=False)
            for change in ({'enable_gpu':True},{'enable_tpu':True},{'is_private':False},{'machine_shape':'NvidiaTeslaT4'}):
                (folder/'kernel-metadata.json').write_text(json.dumps(dict(good,**change)))
                with self.assertRaises(ValueError):validate(folder)

    def test_image_import_hashes_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.mkdir();dest=Path(temp)/'images'
            archive=source/'video.zip';content=b'opaque image chunk';digest=hashlib.sha256(content).hexdigest()
            hashes={'chunk':digest}
            with zipfile.ZipFile(archive,'w') as output:output.writestr('v.zarr/chunk',content)
            manifest=dict(archive='video.zip',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                video='v',image_files=hashes,image_sha256=signature(hashes),source_bytes=len(content),annotations_read=False)
            (source/'image_package.json').write_text(json.dumps(manifest))
            with patch('import_cpu_inputs.shutil.disk_usage',return_value=type('Space',(),{'free':30*2**30})()):
                result=import_package(source,dest)
                self.assertEqual(result['verified_files'],1)
                self.assertEqual((dest/'v.zarr/chunk').read_bytes(),content)
                with zipfile.ZipFile(archive,'w') as output:output.writestr('v.zarr/../../outside',content)
                hashes={'../../outside':digest};manifest.update(image_files=hashes,image_sha256=signature(hashes),
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                (source/'image_package.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'Unsafe'):import_package(source,dest)
            self.assertFalse((Path(temp)/'outside').exists())


if __name__=='__main__':unittest.main()
