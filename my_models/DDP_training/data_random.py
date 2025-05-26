import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import configargparse
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import cv2 as cv
import random
import torchvision.transforms as transforms
from data_enhence import AddPepperNoise, AddResizeCrop, AddRotation, EnhenceSequence, AddAffine, QuadrantShuffle
from data_enhence import MySequenceAffine, MySequenceColorJitter, MySequenceRanResizedCrop, KeepCriticalColor, MyRandomTranslation
import open_clip
import copy
import pickle

observe_space_items = {"robot_obs", "rgb_static", "rgb_gripper", "rgb_tactile"}
action_space_items = {"actions"}

class CalvinDataset(Dataset):
    """Naive implementation of dataset to store
    calvin debug dataset, may be changed to WDS for the full dataset
    """

    def __init__(self, dataset_name, image_processor, obs_horizon, pred_horizon, training_mode, is_train=True, test=False) -> None:
        super().__init__()

        self.image_processor = image_processor
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.test = test
        self.language_tokens = -1

        import json
        loaded_config = dict()
        with open("../../config_path.json", "r") as f:
            loaded_config = json.load(f)
        
        dataset_path = loaded_config["CALVIN_dataset_path"]
        dataset_wo_image_path = loaded_config["dataset_wo_image_path"]
        BERT_PATH = loaded_config["BERT_path"]
        
        dataset_path = dataset_path + "task_" + dataset_name
        print("using dataset in path:", dataset_path)

        if training_mode == "first":
            print("the data set operate in first mode")
            self.language_tokens = 8
        elif training_mode == "second":
            print("the data set operate in second mode")
            self.language_tokens = 16
        else:
            raise("Only support 'first' and 'second' training modes!")
        
        tag = "training" if is_train else "validation"
        self.file_prefix = f"{dataset_path}/{tag}"

        self.dataset_wo_image = None

        dataset_wo_image_path = dataset_wo_image_path + "dataset_wo_image_" + dataset_name + ".pkl"
        print("reading sparated action data in path:", dataset_wo_image_path)
        with open(dataset_wo_image_path, 'rb') as f:
            self.dataset_wo_image = pickle.load(f)

        self.anns = np.load(
            f"{self.file_prefix}/lang_annotations/auto_lang_ann.npy", allow_pickle=True
        ).item()

        selected_tasks = {'move_slider_right': 0, #1 
                            'lift_blue_block_table': 0, #1 
                            'lift_pink_block_table': 0, #1
                            'lift_blue_block_drawer': 0, 
                            'push_pink_block_right': 0, 
                            'turn_on_led': 0, 
                            'push_into_drawer': 0, 
                            'close_drawer': 0, #1
                            'push_blue_block_left': 0, 
                            'lift_red_block_drawer': 0, 
                            'lift_pink_block_slider': 0, 
                            'rotate_red_block_left': 0, 
                            'rotate_red_block_right': 0, 
                            'unstack_block': 0, 
                            'place_in_drawer': 0, #1 
                            'place_in_slider': 0, #1
                            'open_drawer': 0, #1
                            'lift_blue_block_slider': 0, 
                            'rotate_pink_block_left': 0, 
                            'move_slider_left': 0, #1 
                            'turn_off_lightbulb': 0, 
                            'lift_pink_block_drawer': 0, 
                            'push_blue_block_right': 0, 
                            'turn_off_led': 0, 
                            'lift_red_block_table': 0, 
                            'turn_on_lightbulb': 0, 
                            'rotate_blue_block_left': 0, 
                            'push_pink_block_left': 0, 
                            'stack_block': 0, 
                            'lift_red_block_slider': 0, 
                            'rotate_blue_block_right': 0, 
                            'push_red_block_left': 0, 
                            'push_red_block_right': 0, 
                            'rotate_pink_block_right': 0
                        }

        selected_tasks_samples = selected_tasks.copy()

        self.samples_lookup = []
        num_task = 0
        #num_samples_per_task = 10
        task_languages_dict = dict()
        languages_set = set()

        for task_id, (start_idx, end_idx) in enumerate(self.anns["info"]["indx"]):
            task_name = self.anns["language"]["task"][task_id]
            text = self.anns["language"]["ann"][task_id]
            if task_name in selected_tasks.keys():
                num_task += 1
                selected_tasks[task_name] += 1
                if task_name not in task_languages_dict:
                    if training_mode == "first":
                        task_languages_dict[task_name] = {task_name.replace("_"," ")}
                    if training_mode == "second":
                        task_languages_dict[task_name] = {task_name.replace("_"," "), text}
                else:
                    if training_mode == "first":
                        pass
                    if training_mode == "second":
                        task_languages_dict[task_name].add(text)

                for pred_start_idx in range(start_idx, end_idx - pred_horizon + 2):
                    pred_end_idx = min(end_idx, pred_start_idx + pred_horizon - 1)
                    obs_start_idx = pred_start_idx
                    obs_end_idx = obs_start_idx + obs_horizon - 1
                    sample_info = {
                                    "task_id" : task_id, 
                                    "pred_start_idx": pred_start_idx, 
                                    "pred_end_idx": pred_end_idx,
                                    "obs_start_idx": obs_start_idx,
                                    "obs_end_idx": obs_end_idx
                                   }
                    self.samples_lookup.append(sample_info)
                    selected_tasks_samples[task_name] += 1

        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        bert = BertModel.from_pretrained(BERT_PATH)

        self.task_languages_x_dict = dict()
        self.task_mask_dict = dict()

        for task, languages_set in task_languages_dict.items():
            languages_list = list(languages_set)
            task_languages_dict[task] = languages_list
            tokens_info = tokenizer(languages_list,
                            padding="max_length",
                            max_length=self.language_tokens,
                            truncation=True,
                            return_tensors="pt"
                            )
            languages_x_dict = bert(input_ids=tokens_info["input_ids"], attention_mask=tokens_info["attention_mask"]) 
            languages_x = languages_x_dict["last_hidden_state"].detach()
            tokens_mask = tokens_info["attention_mask"]
            self.task_languages_x_dict[task] = {"languages_num": languages_x.shape[0]-1, "languages_x": languages_x}
            self.task_mask_dict[task] = tokens_mask

        print("the natural languages of each task are: ", task_languages_dict)
        
        self.vs_enhence = None
        self.vg_enhence = None

        if training_mode == "first":
            if dataset_name == "ABCD_D":
                # Some affine augmentations should be applied during the 1st stage to get a better performance for ABCD_D
                self.vs_enhence = transforms.Compose(
                    [
                        AddResizeCrop(size=200, max_crop_scale=0.85 ,p=0.9),
                        AddRotation(max_angle=15, p=0.9),
                        AddPepperNoise(0.95, p=0.8),
                    ]
                )
    
                self.vg_enhence = transforms.Compose(
                    [
                        AddResizeCrop(size=84, max_crop_scale=0.85 ,p=0.9),
                        AddRotation(max_angle=15, p=0.9),
                        AddPepperNoise(0.95, p=0.8),
                    ]
                )
            else:
                self.vs_enhence = transforms.Compose(
                    [
                        AddResizeCrop(size=200, max_crop_scale=0.85 ,p=0.9),
                        #AddRotation(max_angle=15, p=0.9),
                        #AddPepperNoise(0.95, p=0.8),
                    ]
                )
    
                self.vg_enhence = transforms.Compose(
                    [
                        AddResizeCrop(size=84, max_crop_scale=0.85 ,p=0.9),
                        #AddRotation(max_angle=15, p=0.9),
                        #AddPepperNoise(0.95, p=0.8),
                    ]
                )

        elif training_mode == "second":
            self.vs_enhence = transforms.Compose(
                [
                    AddPepperNoise(0.95, p=0.8),
                ]
            )

            self.vg_enhence = transforms.Compose(
                [
                    AddPepperNoise(0.95, p=0.8),
                ]
            )
        
        self.myStrongCJ = MySequenceColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.mySeqAff_s = MySequenceAffine()
        self.colorKeeper = KeepCriticalColor()
        self.myRandomTranslation = MyRandomTranslation()
        print("ready")

    def __len__(self):
        return len(self.samples_lookup)

    def __getitem__(self, index):
    	# get the language related info
        sample_info = self.samples_lookup[index]
        task_id = sample_info["task_id"]

        obs_start_idx = sample_info["obs_start_idx"]
        obs_end_idx = sample_info["obs_end_idx"]
        pred_start_idx = sample_info["pred_start_idx"]
        pred_end_idx = sample_info["pred_end_idx"]

        task = self.anns["language"]["task"][task_id]
        max_language_idx = self.task_languages_x_dict[task]["languages_num"]
        sampled_language_idx = random.randint(0, max_language_idx)
        language_x = self.task_languages_x_dict[task]["languages_x"][sampled_language_idx]

        # get the observation and action in this timespan, the time before the sequence starting will be filled with 0
        vs_height = 224; vs_width = 224; vs_channel = 3
        vg_height = 224; vg_width = 224; vg_channel = 3
        
        #RN50x16
        prop_dim = 15
        action_dim = 7
        #gripper_dim = 1

        rgb_static = torch.zeros(self.obs_horizon, vs_channel, vs_height, vs_width)
        rgb_gripper = torch.zeros(self.obs_horizon, vg_channel, vg_height, vg_width)        
        robot_obs = torch.zeros(self.obs_horizon, prop_dim)
        rel_actions = torch.zeros(self.pred_horizon, action_dim)
        #rel_gripper = torch.zeros(self.pred_horizon, gripper_dim)
        padding_mask = torch.ones(self.pred_horizon)
        
        task = self.anns["language"]["task"][task_id]
        StrongJittered = None
        if "blue" in task or "red" in task or "pink" in task:
            StrongJittered = False
        else:
            StrongJittered = True

        img_frames = [
            np.load(f"{self.file_prefix}/episode_{frame_idx:07d}.npz")
            for frame_idx in range(pred_start_idx, pred_start_idx + self.obs_horizon)
        ]
        
        self.myStrongCJ.set_params()
        #self.myWeakCJ.set_params()
        #self.mySeqAff_s.set_params()

        #self.myRandomTranslation.set_params()

        for i, frame_idx in enumerate(range(pred_start_idx, pred_end_idx+1)):
            if i < self.obs_horizon:
                rgb_static_i = Image.fromarray(img_frames[i]["rgb_static"])
                rgb_gripper_i = Image.fromarray(img_frames[i]["rgb_gripper"])
                
                ori_rgb_static_i = copy.deepcopy(rgb_static_i)
                ori_rgb_gripper_i = copy.deepcopy(rgb_gripper_i)

                rgb_static_i = self.myStrongCJ.ex(rgb_static_i)
                rgb_gripper_i = self.myStrongCJ.ex(rgb_gripper_i)

                # #rgb_static_i = self.mySeqAff_s.ex(rgb_static_i)
                rgb_static_i = self.colorKeeper.recover_critical_color("all", ori_rgb_static_i, rgb_static_i)
                rgb_gripper_i = self.colorKeeper.recover_critical_color("all", ori_rgb_gripper_i, rgb_gripper_i)

                # rgb_static_i = self.mySeqAff_s.ex(rgb_static_i)
                # rgb_static_i = self.myRandomTranslation.ex(rgb_static_i)

                rgb_static_i = self.vs_enhence(rgb_static_i)
                rgb_gripper_i = self.vg_enhence(rgb_gripper_i)

                if self.test:
                    np_s = np.array(rgb_static_i)
                    np_v = np.array(rgb_gripper_i)

                    np_s = cv.cvtColor(np_s, cv.COLOR_RGB2BGR)
                    np_v = cv.cvtColor(np_v, cv.COLOR_RGB2BGR)

                    ori_np_s = np.array(ori_rgb_static_i)
                    ori_np_v = np.array(ori_rgb_gripper_i)

                    ori_np_s = cv.cvtColor(ori_np_s, cv.COLOR_RGB2BGR)
                    ori_np_v = cv.cvtColor(ori_np_v, cv.COLOR_RGB2BGR)

                    cv.imshow("np_s",np_s)
                    cv.imshow("np_v",np_v)
                    cv.imshow("ori_np_s",ori_np_s)
                    cv.imshow("ori_np_v",ori_np_v)
                    cv.waitKey(0)
            
                rgb_static[i] = self.image_processor(rgb_static_i)
                rgb_gripper[i] = self.image_processor(rgb_gripper_i)
            
            padding_mask[i] = 0
            #frames[i]["rel_actions"][0:6] = np.around(frames[i]["rel_actions"][0:6] * 20)/20
            rel_actions[i] = torch.from_numpy(self.dataset_wo_image[frame_idx]["rel_actions"][0:7])
            #rel_gripper[i] = torch.from_numpy((frames[i]["rel_actions"][6:7] + 1) // 2)

        #padding_mask[0:frames_len] = 0

        observe_space = {"rgb_static": rgb_static, "robot_obs": robot_obs, "rgb_gripper":rgb_gripper}
        action_space = {"rel_actions": rel_actions}

        return observe_space, task, language_x, padding_mask, action_space
    

if __name__ == "__main__":
    #from observation_encoder import make_language_encoder

    print("test")
    #dataset_path = "/home/wibot/Data/SC/task_ABCD_D"
    device = torch.device("cuda:0") 


    num_history = 30

    # define the training style
    
    clip_vision_encoder_path: str = "ViT-B-32"
    clip_vision_encoder_pretrained: str = "openai"
    clip_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    
    calvin_dataset = CalvinDataset(
        dataset_name="ABCD_D",
        image_processor=image_processor,
        obs_horizon=14, 
        pred_horizon=16,
        training_mode="second",
        test=True
    )

    calvin_dataset.__getitem__(80080) #980


    
