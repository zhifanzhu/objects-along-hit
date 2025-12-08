
# Reconstructing Objects along Hand Interaction Timelines in Egocentric Video

![Video](https://github.com/user-attachments/assets/fd55aec9-1195-4491-8f9c-4d61c8d26f5e)

[[Project Page]](https://zhifanzhu.github.io/objects-along-hit) [[arxiv]](https://arxiv.org/abs/xxxx.yyyy)


## Annotations

### Hand Interaction Timelines (HIT)

#### EPIC-HIT

The EPIC-HIT annotations can be found in [code_epichor/timelines/epic_hit.json](code_epichor/timelines/epic_hit.json), which contains 96 HITs. 
The important fields are explained as below:
- `vid`: video id in EPIC-KITCHENS-100
- `cat`: object category
- `segments`: list of segments in the HIT, each segment has:
  - `st`: start frame (inclusive)
  - `ed`: end frame (inclusive)
  - `side`: the hand that grasps (for _Stable Grasp_) or touches (for _Unstable Contact_) the object. For _Static_, this will be `null`.
  - `ref`: segment type, one of `scene_static` (_Static_), `scene_dynamic` (_Unstable Contact_), or `inhand` (_Stable Grasp_)

  The other fields, `timeline_name`, `mp4_name`, `total_start`, and `total_end`, are for logging and bookkeeping. `main_side` is not used.


<details>
<summary>An example HIT annotation</summary>

```json
  {
    "timeline_name": "P01_09_left_plate_155777_156033",
    "vid": "P01_09",
    "mp4_name": "P01_09_155208_155859_156206_left_plate",
    "cat": "plate",
    "main_side": "left",
    "segments": [
      {
        "st": 155777,
        "ed": 155840,
        "side": null,
        "ref": "scene_static"
      },
      {
        "st": 155840,
        "ed": 155866,
        "side": "left",
        "ref": "scene_dynamic"
      },
      {
        "st": 155868,
        "ed": 156033,
        "side": "left",
        "ref": "inhand"
      }
    ],
    "total_start": 155777,
    "total_end": 156033
  }
```
</details>

#### HOT3D-HIT

The HOT3D-HIT annotation can be found in [code_hot3d/timelines/hot3d_hit.json](code_hot3d/timelines/hot3d_hit.json), which contains 113 HITs. The format is the same as EPIC-HIT.

### EPIC Stable Grasps

The 2.4K stable grasp annotations for EPIC-HITCHENS-100 can be found in [code_epichor/image_sets/epicgrasps_2431.csv](code_epichor/image_sets/epicgrasps_2431.csv). Each entry contains:
- `vid`: the video ID
- `st`: the start frame of the grasp
- `et`: the end frame of the grasp
- `cat`: the object category
- `handside`: either "left hand" or "right hand"

Additionally:
- `fmt`: The prefix of output, e.g. `bottle/P01_14_left_hand_57890_57947_*`

Errata: we found a few (6) incorrect annotations in the original CSV file are incorrect. The corrections are not integrated into the original annotation, and can be found in [code_epichor/image_sets/epicgrasps_2431_errata.csv](code_epichor/image_sets/epicgrasps_2431_errata.csv).


## Installation

```
# Clone only the main branch only to avoid large video files in gh-pages branch
git clone -b main --single-branch https://github.com/zhifanzhu/objects-along-hit.git
cd objects-along-hit/

conda create --name rohit-env python=3.10
conda activate rohit-env

# Install cuda toolchain for pytorch and pytorch3d compatibility (?)
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc=11.8 cudatoolkit=11.8
pip install "chumpy==0.70" --no-build-isolation  # chumpy needs to be installed without build isolation
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install --no-deps vrs==1.0.4 projectaria-tools==1.5.1  # required by HOT3D
sh scripts_sh/install_third_party.sh
# Lastly, install MANO, see below
```

This repos has been tested on:
- Ubuntu 22.04, RTX 4090, python 3.10, torch 2.0.0+cu118, pytorch3d 0.7.3
  - The System CUDA is 11.5 at the time of testing.
- This code should also work on Ubuntu 22.04.5, GTX 1080 Ti, CUDA 11.5, python 3.8, torch 1.8.1, pytorch3d 0.6.2


### Setup MANO

Source:  https://github.com/JudyYe/ihoi/blob/main/docs/install.md

  - Download MANO Model (Neutral model: MANO_LEFT.pkl, MANO_RIGHT.pkl):
        - Download ```Models & Code``` in the original [MANO website](https://mano.is.tue.mpg.de/). You need to register to download the MANO data.
        - Put the ```models/MANO_LEFT.pkl``` ```models/MANO_RIGHT.pkl``` file in: `./externals/mano/`

## Download Data

At this moment, we provide pre-extracted masks, HaMeR results for running EPIC-HIT experiments. 
The example data for EPIC-HIT can be downloaded from: [Temporary One Drive Link](https://uob-my.sharepoint.com/:u:/g/personal/ms21614_bristol_ac_uk/EZf3Sx6ArOtIhwybgsQA51IBGGwCA7UYnYzVP_FjAp-k8w?e=FbG4wd).

<details>
<summary>Structure of the data directory after downloading and extracting:</summary>

```
# DATA_STORAGE/ directory is at the same level as README.md

DATA_STORAGE/
└── epic/
    ├── cache_mask_valid_frames/
    │   ├── P01_01_21784_22242_22872_left_pan.json
    │   └── ...
    ├── cache_metric_epic_fields/
    │   ├── P01_01.pkl
    │   └── ...
    ├── hamer_hoa_potim/
    │   ├── P01_01/
    │   │   ├── frame_0000022047.pt
    │   │   └── ...
    │   └── ...
    ├── images/
    │   ├── P01_01/
    │   │   ├── P01_01_frame_0000021903.jpg
    │   │   └── ...
    │   └── ...
    ├── timeline_sam_masks/
    │   ├── left-hand/
    │   │   ├── P01_01_21784_22242_22872_left_pan/
    │   │   │   ├── 000000021498.png
    │   │   │   └── ...
    │   │   └── ...
    │   ├── right-hand/
    │   │   ├── P01_01_21784_22242_22872_left_pan/
    │   │   │   ├── 000000021498.png
    │   │   │   └── ...
    │   │   └── ...
    │   ├── obj/
    │   │   ├── P01_01_21784_22242_22872_left_pan/
    │   │   │   ├── 000000021498.png
    │   │   │   └── ...
    │   │   └── ...
    └── visor_meta_infos/
        ├── 00000.png
        ├── frame_to_mappingId.csv
        └── unfiltered_color_mappings.csv
```

</details>

## Run 

### EPIC-HIT

```python
# Example sequence. Remove "debug_locate" to run all sequences.
HYDRA_FULL_ERROR=1 python potim/run_e2e_v6_segwise_ondemand.py \
  +exp=epic_e2e_v6 \
  dataset.json_path=./code_epichor/timelines/epic_hit.json \
  hydra.run.dir=outputs/(date "+%Y-%m-%d")-epic/e2e_v6 \
  segi_strategy='circle' \
  optim_mv.num_iters=200 \
  use_old_sca=False \
  debug_locate=P12_101_left_mug_29154_30004
```

### Checking EPIC-HIT results

The visualisation of the 3D result will be saved to
`outputs/<date>-epic/e2e_v6/<timeline-name>/FullAfter.mp4`.
  -  e.g. `outputs/2025-12-07-epic/e2e_v6/P12_101_left_mug_29154_30004/FullAfter.mp4` for the timeline above.

The quantitative result of each sequence is saved to `outputs/<date>-epic/e2e_v6/evaluation/<timeline-name>.csv`
  - e.g. `outputs/2025-12-07-epic/e2e_v6/evaluation/P12_101_left_mug_29154_30004.csv` for the timeline above.
  - `segi` indicates the segment index in the timeline. `ref` indicates the segment type (static/unstable/stable-grasp). `oiou` and `avg_sca` correspond to the IOU and SCA in the paper.

To reproduce Table-5 in the paper, run:

```python
python draft/tables/epic_v6_results.py <path-to-result-folder> --separate_hand False
# e.g. python draft/tables/epic_v6_results.py outputs/2025-12-07-epic/e2e_v6/ --separate_hand False
```

<details>
<summary> It should produce the output below (click to expand):</summary>

![Table-5 Screenshot](./docs/assets/Table-5-screenshot.png)

which corresponds to the following entries in the paper:
| Method              | Stable Grasp IOU | Stable Grasp SCA@0.8 |
|---------------------|------------------|------------------------|
| COP w/o propagation | 68.0             | 23.8                   |
| COP                 | 69.2             | 25.5                   |

The other entries can also be found in the output in the according to the segment type and metric.
</details>

### HOT3D-HIT

The HOT3D data extraction code is under development, but will work with the code below eventually.

```python
# Example sequence. Remove debug_locate to run all sequences.
HYDRA_FULL_ERROR=1 python potim/run_e2e_v6_segwise_ondemand.py \
    +exp=hot3d_e2e_v6 \
    hydra.run.dir=outputs/(date "+%Y-%m-%d")-hot3d/e2e_v6_hot3d_iter100 \
    segi_strategy='circle' \
    dataset.static_init_method='multi_upright' \
    optim_mv.num_iters=100 \
    debug_locate=P0001_8d136980_mug_white_00000_03826
```

## Notes

- The provided input data contains HaMeR results, but the procedure of running HaMeR is not included in this repo (yet). Do refer to [HaMeR](https://github.com/geopavlakos/hamer) if you use this code. 

## License Notice

The code in this repository is released under the MIT License.

This repository also provides download links to derivative data generated from 
the EPIC-KITCHENS dataset and the HOT3D dataset. Redistribution of this data is 
permitted under the respective dataset licenses:

- [EPIC-KITCHENS](https://epic-kitchens.github.io/2025), [VISOR](https://epic-kitchens.github.io/VISOR/), and [EPIC Fields](https://epic-kitchens.github.io/epic-fields/) data and derivatives are redistributed 
  under the CC BY-NC 4.0 License. These assets are for non-commercial use only.
  Attribution to the EPIC-KITCHENS dataset must be maintained.

- [HOT3D](https://www.projectaria.com/datasets/hot3d/) object models and derivatives are redistributed under the 
  CC BY-SA 4.0 License (with HOT3D's added "no selling" restriction). 
  Attribution to HOT3D must be maintained, and derivatives must remain under 
  CC BY-SA.

Users are responsible for complying with all terms of the EPIC-KITCHENS and 
HOT3D dataset licenses.

[ARCTIC](https://arctic.is.tue.mpg.de/) data and all derivatives cannot be redistributed under the ARCTIC license. Users must download ARCTIC themselves, and this repository provides only code and scripts for local processing.


## Acknowledgement

This code uses parts of code from the following repositories:
[HOMan](https://github.com/hassony2/homan) and [iHOI](https://github.com/JudyYe/ihoi).


## Citation

Arxiv version coming soon.
