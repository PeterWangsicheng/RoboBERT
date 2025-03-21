import argparse

def separate_data(dataset_name="ABCD_D", separate_mode="language"):
    import pickle
    import numpy as np
    import json
    
    loaded_json = dict()
    with open("./config_path.json", "r") as f:
       loaded_json = json.load(f)
       
    dataset_path = loaded_json["CALVIN_dataset_path"]
    file_prefix = dataset_path + "/task_" + dataset_name + "/training"
    print("extract path is:", file_prefix)

    ep_start_end_ids = None
    if separate_mode == "all":
        ep_start_end_ids = np.load(file_prefix + "/ep_start_end_ids.npy")

    elif separate_mode == "language":
        ep_start_end_ids = np.load(
                f"{file_prefix}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
            ).item()["info"]["indx"]

    dataset_wo_image = dict()
    ids_len = len(ep_start_end_ids)
    for id, start_end_pair in enumerate(ep_start_end_ids):
        start_idx, end_idx = start_end_pair[0], start_end_pair[1]

        for frame_idx in range(start_idx, end_idx+1):
            frame = np.load(f"{file_prefix}/episode_{frame_idx:07d}.npz")
            rel_actions = frame["rel_actions"]
            robot_obs = frame["robot_obs"]
            dataset_wo_image[frame_idx] = {
                "rel_actions": rel_actions,
                #"robot_obs": robot_obs
            }
        print((id + 1) / ids_len * 100)
    

    with open(f'dataset_wo_image_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(dataset_wo_image, f)
    
    print("complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configurating the extraction specifications')
    parser.add_argument('--dataset_name', type=str, default="ABCD_D")
    parser.add_argument('--separate_mode', type=str, default="language", help="only sparate the data with lanuage label")
    args = parser.parse_args()
    separate_data(dataset_name=args.dataset_name, separate_mode=args.separate_mode)
