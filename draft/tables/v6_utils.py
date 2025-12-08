import os
import os.path as osp
import numpy as np
from collections import namedtuple
import pandas as pd
import tqdm

def get_separate_df(df):
    """ process single df """
    df['uuid'] = df['timeline_name'] + '_' + df['segi'].apply(lambda x: str(x))
    max_segi = max(df.segi)
    prop_x1_work_ids = set(range(2*max_segi + 1))
    prop_x2_work_ids = set(range(4*max_segi + 1))

    static = df[df.ref == 'scene_static']
    static_only = static[static.init_type == 'Init']
    static_propx1 = static[static.work_id.isin(prop_x1_work_ids)]
    static_propx2 = static[static.work_id.isin(prop_x2_work_ids)]

    dynamic = df[df.ref == 'scene_dynamic']
    dynamic_only =   dynamic[dynamic.init_type == 'Init']
    dynamic_propx1 = dynamic[dynamic.work_id.isin(prop_x1_work_ids)]
    dynamic_propx2 = dynamic[dynamic.work_id.isin(prop_x2_work_ids)]

    inhand = df[df.ref == 'inhand']
    inhand_only = inhand[inhand.init_type == 'Init']
    inhand_propx1 = inhand[inhand.work_id.isin(prop_x1_work_ids)]
    inhand_propx2 = inhand[inhand.work_id.isin(prop_x2_work_ids)]

    df_dicts = {
        'static_only': static_only,
        'static_propx1': static_propx1,
        'static_propx2': static_propx2,
        'dynamic_only':   dynamic_only,
        'dynamic_propx1': dynamic_propx1,
        'dynamic_propx2': dynamic_propx2,
        'inhand_only': inhand_only,
        'inhand_propx1': inhand_propx1,
        'inhand_propx2': inhand_propx2,
    }
    return df_dicts


def read_e2e_df_from_dir(results_dir):
    """ process a directory of csv files """
    results_dir = osp.join(results_dir, 'evaluation')
    raw_method_dfs = dict()
    for name in tqdm.tqdm([v for v in os.listdir(results_dir) if v.endswith('.csv')]):
        result_path = osp.join(results_dir, name)
        df_dicts = get_separate_df(pd.read_csv(result_path))
        if df_dicts is None:
            continue
        for name, df in df_dicts.items():
            if name not in raw_method_dfs:
                raw_method_dfs[name] = []
            raw_method_dfs[name].append(df)

    for name, dfs in raw_method_dfs.items():
        raw_method_dfs[name] = pd.concat(dfs, ignore_index=True)
    return raw_method_dfs


def gather_intersection_uuids(raw_method_dfs, separate_hand, only_ref):
    retVal = namedtuple(
        'retVal', ['method_dfs', 'available_uuids', 'available_timelines'])
    method_dfs = dict()
    available_uuids = None

    for name, df in raw_method_dfs.items():
        if df is None:
            print(f"Skip {name=}")
            continue

        df = df[df.ref == only_ref]
        if df.empty:
            continue

        df = df.loc[df.groupby(['timeline_name', 'segi'])['oiou'].idxmax()]
        df = df.reset_index(drop=True)

        # Extract the category from 'timeline_name'
        if separate_hand:
            df['cat'] = df['timeline_name'].apply(lambda s: '_'.join(s.split('_')[2:-2]))
        else:
            df['cat'] = df['timeline_name'].apply(lambda s: s.split('_')[3])
        df['name'] = name

        if available_uuids is None:
            available_uuids = set(df['uuid'])
        else:
            available_uuids = available_uuids.intersection(set(df['uuid']))
        method_dfs[name] = df

    _df = list(method_dfs.values())[0]
    available_timelines = set(_df[_df['uuid'].isin(available_uuids)].timeline_name)

    return retVal(method_dfs, available_uuids, available_timelines)


def reshape_dataframe(df, metrics):
    # Rename columns for convenience
    df = df.rename(columns={
        "oiou": "IOU",
        "symADD_0.1": "ADD",
        })

    # Extract the unique categories and methods
    categories = df['cat'].unique()
    methods = df['name'].unique()

    # Separate 'ALL' category to add it at the bottom later
    categories = [cat for cat in categories if cat != 'ALL'] + ['ALL']

    # Prepare the data for the final DataFrame
    # metrics = ["IOU", "SCA@0.8"] # "SCA-IOU"]
    data = []

    # Loop through each category to create the new rows
    for category in categories:
        # Get the rows corresponding to the current category
        category_rows = df[df['cat'] == category]

        # Create a new row starting with the category name
        new_row = [category]

        # Add the metric values for each method
        for method in methods:
            method_data = category_rows[category_rows['name'] == method]
            if not method_data.empty:
                new_row.extend(method_data.iloc[0][metrics].tolist())
            else:
                new_row.extend([None] * len(metrics))

        data.append(new_row)

    # Create the final DataFrame with multi-level columns
    final_df = pd.DataFrame(data)
    method_headers = ["Category"] + [method for method in methods for _ in metrics]
    metric_headers = [""] + list(metrics) * len(methods)

    final_df.columns = pd.MultiIndex.from_arrays([method_headers, metric_headers])

    return final_df


def get_compare_results(method_dfs, available_tlnames,
                        metric_keys,
                        with_sym_metrics=True,
                        order: list = None,
                        has_3d=True):
    watch_tlnames = set()
    watch_tlnames = available_tlnames

    method_averages = dict()
    for name, df in method_dfs.items():
        if name not in order:
            continue
        df = df[df['timeline_name'].isin(watch_tlnames)].copy()
        if 'symADD_0.1' in df.columns:
            df['SCA-ADD'] = df['symADD_0.1'] * df['avg_sca']
        df['SCA@0.8'] = df['avg_sca'] * (df['oiou'] > 0.8)
        df['SCA@0.6'] = df['avg_sca'] * (df['oiou'] > 0.6)
        df['SCA-IOU'] = df['SCA@0.8']

        # **Compute average metrics per category**
        df_grouped = df.groupby('cat').mean(numeric_only=True).reset_index()
        df_grouped['name'] = name

        # **Compute overall average (ALL category)**
        df_overall = df.mean(numeric_only=True).to_frame().T
        df_overall['cat'] = 'ALL'
        df_overall['name'] = name

        # **Combine per-category averages with overall average**
        df_avg = pd.concat([df_grouped, df_overall], ignore_index=True)

        # **Append to method_averages list**
        method_averages[name] = df_avg

    df_results = pd.concat(list(method_averages.values()), ignore_index=True)
    # **Define the desired order of categories, with 'ALL' first**
    categories_order = ['ALL'] + sorted([cat for cat in df_results['cat'].unique() if cat != 'ALL'])
    # **Convert 'cat' column to a categorical type with the specified order**
    df_results['cat'] = pd.Categorical(df_results['cat'], categories=categories_order, ordered=True)

    df_results = df_results.sort_values(by=['cat', 'name']).reset_index(drop=True)
    if order is not None:
        order_map = {name: i for i, name in enumerate(order)}
        df_results = df_results.sort_values(by=['cat', 'name'], key=lambda col: col.map(order_map) if col.name == 'name' else col)

    # columns_of_interest = ['name', 'cat', 'T_err', 'R_err', 'ADD_0.01', 'ADD_0.05', 'ADD_0.1']
    if has_3d:
        if with_sym_metrics:
            # columns_of_interest = ['name', 'cat', 'symT_err', 'symR_err'] + metric_keys
            columns_of_interest = ['name', 'cat'] + metric_keys
        else:
            columns_of_interest = ['name', 'cat'] + metric_keys
    else:
        columns_of_interest = ['name', 'cat', ] + metric_keys
    res = df_results[columns_of_interest]
    return res
