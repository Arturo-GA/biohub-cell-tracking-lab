import sys
from pathlib import Path
from spatial_probe_features_runner import one
from dense_supervision_prepare_runner import main
if __name__=='__main__':main(Path(sys.argv[1]),archive_source=one('dense_supervision/NIS3D.zip'))
