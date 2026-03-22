import re

# Subdirectories in the dataset directory
DATASET_SUBDIR_BRIGHTFIELD = "brightfield"
DATASET_SUBDIR_GFP = "gfp"

# Subdirectories in the results directory
RUN_SUBDIR_FILAMENT_MASK = "filament_mask"
RUN_SUBDIR_DIAGNOSTICS = "diagnostics"
RUN_SUBDIR_STATISTICS = "statistics"
RUN_SUBDIR_CELLPOSE_MASK = "cellpose_mask"
RUN_SUBDIR_BRIGHTFIELD = "brightfield"
RUN_SUBDIR_GFP = "gfp"

# File names
REMOVED_PILLAR_TRACKS_CSV_NAME = "removed_pillar_tracks_v3.csv"

# Frame file pattern:
# <movie_name>_frame_<frame_number>_<channel>.tif(f)
FRAME_FILE_PATTERN = re.compile(
    r"^(?P<movie>.+)_frame_(?P<frame>\d+)_(?P<channel>BF|GFP)\.tif{1,2}$",
    re.IGNORECASE,
)