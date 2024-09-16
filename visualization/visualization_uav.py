# This is a sample Python script.


from nuscenes.nuscenes import NuScenes
from functions import transform_boxes, draw_boxes, get_pred_boxes, get_sensor, get_gt_boxes


# nusc = NuScenes(version='v1.0-mini', dataroot='./data/town_03/03_row_3/01_row_3_all/', verbose=True)
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-n", type=int, metavar="--num",  default=1, required=False)
ap.add_argument("-t", type=str, metavar="--type", default="gt",  help="gt or pred")

args = vars(ap.parse_args())
num = args["n"]
type = args["t"]

# nusc = NuScenes(version='v1.0-mini', dataroot='./data/town_03/01_row_1/01_row_1_all/', verbose=True)
# 'scene_0021'
# my_scene = nusc.scene[21]

# nusc = NuScenes(version='v1.0-mini', dataroot='./data/town_06/03_row_3/01_row_3_all/', verbose=True)
# 'scene_0361'
# my_scene = nusc.scene[11]

# nusc = NuScenes(version='v1.0-mini', dataroot='./data/town_07/03_row_3/01_row_3_all/', verbose=True)
# 'scene_0621'
# my_scene = nusc.scene[21]


# nusc = NuScenes(version='v1.0-mini', dataroot='./data/town_10/03_row_3/01_row_3_all/', verbose=True)
# my_scene = nusc.scene[11]   # scene_0861

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/sim/Desktop/yh/data_gsu/gsu_2_1/town_all/', verbose=True)
my_scene = nusc.scene[861]   # scene_0861

print(my_scene)

first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)

sample_num = num
temp_num =  num
sample_name = "sample_" + str(sample_num).zfill(2)

while temp_num > 1 :
    sample_token = my_sample['next']
    my_sample = nusc.get('sample', sample_token)
    temp_num = temp_num - 1

# print(my_sample)

nusc.list_sample(my_sample['token'])

# test = my_sample['data']
# print(test)

sensors = ['CAMERA_BACK_id_0', 'CAMERA_BOTTOM_id_0', 'CAMERA_FRONT_id_0', 'CAMERA_LEFT_id_0', 'CAMERA_RIGHT_id_0']


if type == 'pred':
    get_pred_boxes(nusc, sample_num)
elif type == 'gt':
    cam_data = nusc.get('sample_data', my_sample['data']['CAMERA_BOTTOM_id_0'])
    get_gt_boxes(nusc,cam_data['token'])

for sensor in sensors:
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    # print(cam_front_data )
    get_sensor(nusc, cam_front_data['token'])
    transform_boxes(sensor, type)
    draw_boxes(sensor, sample_name, sample_num, type)




