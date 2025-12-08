""" Mar 02 2025 
EPIC e2e needs special treatment as inhand and passing live in the same table,
But luckily, the rows `ref` always start with `inhand`, then `jointstatic`, then handscale, then passing

So we need to do some hacking here.
"""
from colorama import Fore, Style
from draft.tex_tabler import TexTabler
from draft.tables.v6_utils import (
    read_e2e_df_from_dir, 
    gather_intersection_uuids, reshape_dataframe,
    get_compare_results,
)
import fire


def main(result_dir, separate_hand=True):
    raw_method_dfs = read_e2e_df_from_dir(result_dir)

    for ref in ['scene_static', 'scene_dynamic', 'inhand']:
        print(Fore.GREEN + f"{ref}" + Style.RESET_ALL)

        method_dfs, avail_uuids, avail_timelines = gather_intersection_uuids(
            raw_method_dfs, separate_hand, ref)
        print(f"{len(avail_uuids)=}, {len(avail_timelines)=}")

        if ref == 'scene_static':
            # order = ['static_only', 'static_propx1', 'static_propx2']
            order = ['static_only', 'static_propx1'] # , 'static_propx2']
            KEYS = ['oiou']
            latex_metric_keys = ['IOU']
        elif ref == 'scene_dynamic':
            # order = ['dynamic_only', 'dynamic_propx1'] # , 'inhand_propx2']
            order = ['dynamic_only', 'dynamic_propx1'] # , 'inhand_propx2']
            KEYS = ['oiou', 'SCA@0.8']
            latex_metric_keys = ['IOU', 'SCA@0.8']
        elif ref == 'inhand':
            order = ['inhand_only', 'inhand_propx1'] # , 'inhand_propx2']
            KEYS = ['oiou', 'SCA@0.8']
            latex_metric_keys = ['IOU', 'SCA@0.8']
        else:
            raise ValueError(f"Unknown ref {ref}")

        res = get_compare_results(
            method_dfs, avail_timelines, order=order,
            metric_keys=KEYS, has_3d=False)
        print(res)
        join = reshape_dataframe(res, latex_metric_keys)
        tabler = TexTabler(join)
        print(tabler.to_latex())


if __name__ == '__main__':
    fire.Fire(main)