import pickle
# from client_bounding_boxes_airsim_nuscenes import ClientSideBoundingBoxes

from transforms3d.euler import euler2mat, quat2euler, euler2quat

import glob
import os
import sys
import numpy as np

import cv2
from nuscenes.nuscenes import NuScenes


import json
import pickle
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

from pyquaternion import Quaternion

VIEW_WIDTH = 800
VIEW_HEIGHT = 450
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

from transforms3d.euler import euler2mat, quat2euler, euler2quat




class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch  # Rotation around Y-axis in degrees
        self.yaw = yaw      # Rotation around Z-axis in degrees
        self.roll = roll    # Rotation around X-axis in degrees

class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x  # Position along X-axis in meters
        self.y = y  # Position along Y-axis in meters
        self.z = z  # Position along Z-axis in meters

class Transform:
    def __init__(self, location=Location(), rotation=Rotation()):
        self.location = location  # An instance of carla.Location
        self.rotation = rotation  # An instance of carla.Rotation


class Size:
  def __init__(self):
    # self.name = name
    self.x = 0
    self.y = 0
    self.z = 0

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera, camera_name):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]

        file_name = camera_name + '_bounding_boxes.pickle'
        with open(file_name, 'wb') as handle:
            pickle.dump(bounding_boxes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return bounding_boxes

    # @staticmethod
    # def draw_bounding_boxes(display, bounding_boxes):
    #     """
    #     Draws bounding boxes on pygame display.
    #     """
    #
    #     bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
    #     bb_surface.set_colorkey((0, 0, 0))
    #     for bbox in bounding_boxes:
    #         points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
    #         # draw lines
    #         # base
    #         pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
    #         # top
    #         pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
    #         # base-top
    #         pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
    #         pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        # display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """



        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])

        # cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], cords_x_y_z[2, :], cords_x_y_z[0, :]])

        # cords_y_minus_z_x = cords_x_y_z

        bbox = np.transpose(np.dot(camera[1], cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        size = Size()

        cords = np.zeros((8, 4))
        # extent = vehicle.bounding_box.extent
        # size.x, size.y, size.z = vehicle.wlh[0] / 2 , vehicle.wlh[1] / 2, vehicle.wlh[2]  / 2
        size.x, size.y, size.z= vehicle.wlh[0] / 2, vehicle.wlh[1] / 2, vehicle.wlh[2] / 2
        extent = size

        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)

        # ue_cord = ClientSideBoundingBoxes._carla_to_ue(world_cord)
        # test = world_cord.copy()
        # ue_cord[2,:] = ue_cord[2,:] * (-1)

        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        location = Location(vehicle.center[0], vehicle.center[1], vehicle.center[2])

        # bb_transform = carla.Transform(vehicle.bounding_box.location)

        bb_transform = Transform(Location(0,0,vehicle.wlh[2] / 2))

        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        # bb_vehicle_matrix = np.identity(4)

        # vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())

        # vehicle_world_matrix = np.identity(4)
        # temp_matrix = vehicle.rotation_matrix
        # vehicle_world_matrix[:3,:3] = temp_matrix

        w_val, x_val, y_val, z_val = vehicle.orientation.w, vehicle.orientation.x, vehicle.orientation.y, vehicle.orientation.z
        # Quaternion to Euler angle
        angles = quat2euler([w_val, x_val, y_val, z_val])

        angles_1 = Quaternion([w_val, x_val, y_val, z_val]).yaw_pitch_roll
        # angles_2 = Quaternion([x_val, y_val, z_val, w_val]).yaw_pitch_roll

        # yaw = Quaternion([w_val, x_val, y_val, z_val]).yaw_pitch_roll[0]

        # Radians to degrees
        angles = [angle * 180 / (np.pi) for angle in angles]
        angles_1 = [angle_1 * 180 / (np.pi) for angle_1 in angles_1]
        # angles_2 = [angle_2 * 180 / (np.pi) for angle_2 in angles_2]


        # temp1, temp2, temp3 = angles
        # angles = (temp1, temp2, temp3 - 180)

        rotation = Rotation(angles[1], angles[2], angles[0])
        # rotation = carla.Rotation(angles[0], angles[2], angles[1])

        transform = Transform(location, rotation)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(transform)

        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    def _carla_to_ue(cords):
        """
        Transforms coordinates of a vehicle bounding box from carla to unreal engine.
        """
        # x_val, y_val, z_val = 44.07, -5.84, 4.15
        x_val, y_val, z_val = -44.07, 5.84, -4.15

        # roll_ang, pitch_ang, yaw_ang = 0.000044, 0.599745, 173.19838
        roll_ang, pitch_ang, yaw_ang = 0.0, 0.0, 0.0
        location = Location(x_val, y_val, z_val)
        rotation = Rotation(pitch=pitch_ang, yaw=yaw_ang, roll=roll_ang)
        transform = Transform(location, rotation)
        bb_carla_ue = ClientSideBoundingBoxes.get_matrix(transform)

        ue_cords = np.dot(bb_carla_ue, cords)
        return ue_cords


    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        # w_val, x_val, y_val, z_val = quaternion_cam.w_val, quaternion_cam.x_val, quaternion_cam.y_val, \
        #                              quaternion_cam.z_val

        # w_val, x_val, y_val, z_val = sensor['rotation'][0], sensor['rotation'][1], sensor['rotation'][2], sensor['rotation'][3]
        # # Quaternion to Euler angle
        # angles = quat2euler([w_val, x_val, y_val, z_val])
        # # Radians to degrees
        # angles = [angle * 180 / (np.pi) for angle in angles]
        # # print(angles)
        # # rotation = (roll,pitch, yaw) --->  (pitch, yaw, roll)
        # # rotation_cam = carla.Rotation(angles[1], angles[2], angles[0])
        #
        # # rotation_cam = carla.Rotation(angles[1], -angles[2], angles[0])
        # rotation_cam = carla.Rotation(angles[1], angles[2], angles[0])
        #
        # x, y, z = sensor['translation'][0], sensor['translation'][1], sensor['translation'][2]
        #
        # location_cam = carla.Location(x, y, z)
        #
        # transform_cam = carla.Transform(location_cam, rotation_cam)

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor[0])

        # sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(transform_cam)

        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


def transform_boxes(camera_name, type='pred'):

    # boxes_name = camera_name + '_boxes.pickle'
    if type == 'pred':
        boxes_name = 'boxes_pred.pickle'
    else:
        boxes_name =  'gt_boxes.pickle'
    sensor_name = camera_name + '_sensor.pickle'

    with open(boxes_name, 'rb') as fp:
        boxes = pickle.load(fp)
    with open(sensor_name, 'rb') as fp:
        sensor = pickle.load(fp)

    print('test')

    w_val, x_val, y_val, z_val = sensor['rotation'][0], sensor['rotation'][1], sensor['rotation'][2], sensor['rotation'][3]
    # Quaternion to Euler angle
    angles = quat2euler([w_val, x_val, y_val, z_val])
    # Radians to degrees
    angles = [angle * 180 / (np.pi) for angle in angles]
    # print(angles)
    # rotation = (roll,pitch, yaw) --->  (pitch, yaw, roll)
    # rotation_cam = carla.Rotation(angles[1], angles[2], angles[0])

    # rotation_cam = carla.Rotation(angles[1], -angles[2], angles[0])
    rotation_cam = Rotation(angles[1], angles[2], angles[0])

    x, y, z = sensor['translation'][0], sensor['translation'][1], sensor['translation'][2]

    location_cam = Location(x, y, z)

    transform_cam = Transform(location_cam, rotation_cam)

    # bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(transform_cam)

    calibration = np.identity(3)
    calibration[0, 2] = 800 / 2.0
    calibration[1, 2] = 450 / 2.0
    calibration[0, 0] = calibration[1, 1] = 800 / (2.0 * np.tan(90 * np.pi / 360.0))

    camera = [transform_cam, calibration]

    # camera = transform_cam

    # boxes_new = [ ]
    # for box in boxes:
    #     box_new = box.copy()
    #     box_new.wlh = box_new.wlh / 2
    #     boxes_new.append(box_new)


    vehicles = boxes
    # camera = sensor
    bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, camera, camera_name)


def draw_boxes(camera_name, sample_name, sample_num, type='gt'):
    # for camera_name in camera_names:

    # write_name = './vis/' + 'town03/' + 'town_03_' + sample_name + '_' +  camera_name + '.png'
    # path = './data/town_03/03_row_3/01_row_3_all/sweeps/' + camera_name
    # path = './data/town_03/01_row_1/01_row_1_all/sweeps/' + camera_name
    # scene_num = 21

    # write_name = './vis/' + 'town06/' + 'town_06_' + sample_name + '_' +  camera_name + '.png'
    # path = './data/town_06/03_row_3/01_row_3_all/sweeps/' + camera_name
    # scene_num = 11

    write_name = './vis/' + 'town07/' + 'town_07_' + sample_name + '_' +  camera_name + '.png'
    path = './data/town_07/03_row_3/01_row_3_all/sweeps/' + camera_name
    # scene_num = 21

    write_name = './vis/' + 'town10/test/' + 'town_10_'+ sample_name + '_' +  camera_name + '.png'
    path = './data/town_10/03_row_3/01_row_3_all/sweeps/' + camera_name
    scene_num = 11

    # sample_num = 0
    file_names = os.listdir(path)
    file_names = sorted(file_names)
    imag_name = file_names[scene_num * 40 + (sample_num - 1) * 2]

    # path = './data/town_10/03_row_3/01_row_3_all/sweeps/' + camera_name + '/1701672639377349888.png'
    path = path + '/' + imag_name
    image = cv2.imread(path)

    box_name = camera_name + '_bounding_boxes.pickle'
    with open(box_name, 'rb') as handle:
        bounding_boxes = pickle.load(handle)

    count_out = 0

    for bbox in bounding_boxes:
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

        x_list = [item[0] for item in points]
        y_list = [item[1] for item in points]
        x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)

        if x_max > 800 or y_max > 450 or x_min < 0 or y_min < 0:
            count_out += 1
            print(count_out)
            continue

        if type == 'gt':
            color_value = (255, 0,0, 255) # blue
        else:
            color_value = (0,0,255, 255)   # red

        cv2.line(image, points[0], points[1], color_value, 1)
        cv2.line(image, points[1], points[2], color_value, 1)
        cv2.line(image, points[2], points[3], color_value, 1)
        cv2.line(image, points[3], points[0], color_value, 1)
        #
        cv2.line(image, points[4], points[5], color_value, 1)
        cv2.line(image, points[5], points[6], color_value, 1)
        cv2.line(image, points[6], points[7], color_value, 1)
        cv2.line(image, points[7], points[4], color_value, 1)
        #
        cv2.line(image, points[0], points[4], color_value, 1)
        cv2.line(image, points[1], points[5], color_value, 1)
        cv2.line(image, points[2], points[6], color_value, 1)
        cv2.line(image, points[3], points[7], color_value, 1)

    # cv2.imshow('Rectangle', image)
    cv2.imwrite(write_name, image)



def get_pred_boxes(nusc, sample_num):

    filename = "./data/comm/lower/track/val/results_nusc.json"
    # filename = "./data/comm/upper/track/val/results_nusc.json"

    # Open the file in read mode
    with open(filename, "r") as f:
        # Load the JSON data using json.load
        data = json.load(f)

    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/sim/Desktop/yh/data_gsu/gsu_2_1/town_all/', verbose=True)

    # my_scene = nusc.scene[21]  # town03
    # my_scene = nusc.scene[361]  # town06
    # my_scene = nusc.scene[621]  # town07
    my_scene = nusc.scene[861]  # town10

    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    while sample_num > 1:
        sample_token = my_sample['next']
        my_sample = nusc.get('sample', sample_token)
        sample_num = sample_num - 1

    # last_sample_token = my_scene['last_sample_token']
    # my_sample = nusc.get('sample', last_sample_token)

    sample_token = my_sample['token']
    sample_predict = data['results'][sample_token]
    # sample_predict = data['results'][last_sample_token]

    boxes_pred_list = []
    for record in sample_predict:
        box_predict = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                          name=record['detection_name'], token=record['sample_token'])
        boxes_pred_list.append(box_predict)

    import pickle

    with open("boxes_pred.pickle", "wb") as f:
        # Dump the list to the file using pickle.dump
        pickle.dump(boxes_pred_list, f)



def get_sensor(nusc, sample_data_token, type='gt'):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    import pickle

    print(sd_record)
    sensor_name = sensor_record['channel'] + '_sensor.pickle'
    with open(sensor_name, "wb") as f:
        # Dump the list to the file using pickle.dump
        pickle.dump(cs_record, f)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None




def get_gt_boxes(nusc, sample_data_token):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    boxes = nusc.get_boxes(sample_data_token)
    boxes_name = 'gt_boxes.pickle'
    with open(boxes_name, "wb") as f:
        # Dump the list to the file using pickle.dump
        pickle.dump(boxes, f)


