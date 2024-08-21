# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

from nuscenes.utils.data_classes import Box


from transforms3d.euler import euler2mat, quat2euler, euler2quat
from transforms3d.quaternions import quat2mat, mat2quat

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

train_split = ['scene_2', 'scene_4', 'scene_5', 'scene_7', 'scene_8',  'scene_9','scene_10',
               'scene_11', 'scene_12', 'scene_13', 'scene_14', 'scene_15', 'scene_16']
val_split = ['scene_0', 'scene_6']
test_split = ['scene_1', 'scene_3']

train_split = ['scene_0', 'scene_1', 'scene_2']
val_split = ['scene_3']
test_split = ['scene_3']

x_min, x_max = -102.4, 102.4
y_min, y_max = -102.4, 102.4
z_min, z_max = -10, 10
center_cam = 'CAMERA_BOTTOM_id_0'

# train_split = ['scene_0', 'scene_1', 'scene_2', 'scene_3', 'scene_4', 'scene_5',
#                'scene_6', 'scene_7', 'scene_8', 'scene_9', 'scene_10', 'scene_11',
#                'scene_12', 'scene_13', 'scene_14', 'scene_15', 'scene_16', 'scene_17',
#                'scene_18', 'scene_19', 'scene_20', 'scene_21', 'scene_22', 'scene_23'
#                ]
# val_split = ['scene_24', 'scene_25', 'scene_26', 'scene_27', 'scene_28', 'scene_29', 'scene_30', 'scene_31']
# test_split = ['scene_24', 'scene_25', 'scene_26', 'scene_27', 'scene_28', 'scene_29', 'scene_30', 'scene_31']



# train_split = ['scene_0', 'scene_1', 'scene_2', 'scene_3', 'scene_4', 'scene_5',
#                'scene_6', 'scene_7', 'scene_8', 'scene_9', 'scene_10', 'scene_11',
#                'scene_12', 'scene_13', 'scene_14', 'scene_15', 'scene_16', 'scene_17',
#                'scene_18', 'scene_19', 'scene_20', 'scene_21', 'scene_22', 'scene_23',
#                'scene_24', 'scene_25', 'scene_26', 'scene_27', 'scene_28', 'scene_29',
#                'scene_30', 'scene_31', 'scene_32', 'scene_33', 'scene_34', 'scene_35',
#                'scene_36', 'scene_37', 'scene_38', 'scene_39', 'scene_40', 'scene_41',
#                'scene_42', 'scene_43', 'scene_44', 'scene_45', 'scene_46', 'scene_47'
#                ]

# val_split = ['scene_48', 'scene_49', 'scene_50', 'scene_51', 'scene_52', 'scene_53',
#              'scene_54', 'scene_55', 'scene_56', 'scene_57', 'scene_58', 'scene_59',
#              'scene_60', 'scene_61', 'scene_62', 'scene_63']
# test_split = ['scene_48', 'scene_49', 'scene_50', 'scene_51', 'scene_52', 'scene_53',
#              'scene_54', 'scene_55', 'scene_56', 'scene_57', 'scene_58', 'scene_59',
#              'scene_60', 'scene_61', 'scene_62', 'scene_63']

train_split = ['scene_0', 'scene_1', 'scene_2','scene_3', 'scene_4', 'scene_5',
               'scene_6', 'scene_7']
val_split = ['scene_8', 'scene_9']
test_split = ['scene_8', 'scene_9']

train_split = ['scene_00', 'scene_01', 'scene_02', 'scene_03', 'scene_04', 'scene_05',
               'scene_06', 'scene_07', 'scene_08', 'scene_09', 'scene_10', 'scene_11',
               'scene_12', 'scene_13', 'scene_14', 'scene_15', 'scene_16', 'scene_17',
               'scene_18', 'scene_19', 'scene_20', 'scene_21', 'scene_22', 'scene_23',
               'scene_24', 'scene_25', 'scene_26', 'scene_27', 'scene_28', 'scene_29',
               'scene_30', 'scene_31', 'scene_32', 'scene_33', 'scene_34', 'scene_35',
               'scene_36', 'scene_37', 'scene_38', 'scene_39']

val_split = ['scene_40', 'scene_41', 'scene_42', 'scene_43', 'scene_44', 'scene_45',
             'scene_46', 'scene_47', 'scene_48', 'scene_49']
test_split = ['scene_40', 'scene_41', 'scene_42', 'scene_43', 'scene_44', 'scene_45',
             'scene_46', 'scene_47', 'scene_48', 'scene_49']


train_split = ['scene_00', 'scene_03', 'scene_04', 'scene_05',
                'scene_06', 'scene_07', 'scene_08', 'scene_09', 'scene_10',
                'scene_13', 'scene_14', 'scene_15', 'scene_16', 'scene_17',
                'scene_18', 'scene_19', 'scene_20',  'scene_23',
                'scene_24', 'scene_25', 'scene_26', 'scene_27', 'scene_28', 'scene_29',
                'scene_30',  'scene_33', 'scene_34', 'scene_35',
                'scene_36', 'scene_37', 'scene_38', 'scene_39','scene_40',
                'scene_43', 'scene_44', 'scene_45', 'scene_46', 'scene_47', 'scene_48', 'scene_49']

val_split = ['scene_01', 'scene_02', 'scene_11', 'scene_12', 'scene_21', 'scene_22',
              'scene_31', 'scene_32', 'scene_41', 'scene_42']
test_split = ['scene_01', 'scene_02', 'scene_11', 'scene_12', 'scene_21', 'scene_22',
              'scene_31', 'scene_32', 'scene_41', 'scene_42']

train_split = [
		'scene_0000', 'scene_0002', 'scene_0004', 'scene_0006', 'scene_0007', 'scene_0008', 'scene_0009', 'scene_0010', 'scene_0012', 'scene_0014',
		'scene_0016', 'scene_0017', 'scene_0018', 'scene_0019', 'scene_0020', 'scene_0022', 'scene_0024', 'scene_0026', 'scene_0027', 'scene_0028',
		'scene_0029', 'scene_0030', 'scene_0032', 'scene_0034', 'scene_0036', 'scene_0037', 'scene_0038', 'scene_0039', 'scene_0040', 'scene_0042',
		'scene_0044', 'scene_0046', 'scene_0047', 'scene_0048', 'scene_0049', 'scene_0050', 'scene_0052', 'scene_0054', 'scene_0056', 'scene_0057',
		'scene_0058', 'scene_0059', 'scene_0060', 'scene_0062', 'scene_0064', 'scene_0066', 'scene_0067', 'scene_0068', 'scene_0069', 'scene_0070',
		'scene_0072', 'scene_0074', 'scene_0076', 'scene_0077', 'scene_0078', 'scene_0079', 'scene_0080', 'scene_0082', 'scene_0084', 'scene_0086',
		'scene_0087', 'scene_0088', 'scene_0089', 'scene_0090', 'scene_0092', 'scene_0094', 'scene_0096', 'scene_0097', 'scene_0098', 'scene_0099',
		'scene_0100', 'scene_0102', 'scene_0104', 'scene_0106', 'scene_0107', 'scene_0108', 'scene_0109', 'scene_0110', 'scene_0112', 'scene_0114',
		'scene_0116', 'scene_0117', 'scene_0118', 'scene_0119', 'scene_0120', 'scene_0122', 'scene_0124', 'scene_0126', 'scene_0127', 'scene_0128',
		'scene_0129', 'scene_0130', 'scene_0132', 'scene_0134', 'scene_0136', 'scene_0137', 'scene_0138', 'scene_0139', 'scene_0140', 'scene_0142',
		'scene_0144', 'scene_0146', 'scene_0147', 'scene_0148', 'scene_0149', 'scene_0150', 'scene_0152', 'scene_0154', 'scene_0156', 'scene_0157',
		'scene_0158', 'scene_0159', 'scene_0160', 'scene_0162', 'scene_0164', 'scene_0166', 'scene_0167', 'scene_0168', 'scene_0169', 'scene_0170',
		'scene_0172', 'scene_0174', 'scene_0176', 'scene_0177', 'scene_0178', 'scene_0179', 'scene_0180', 'scene_0182', 'scene_0184', 'scene_0186',
		'scene_0187', 'scene_0188', 'scene_0189', 'scene_0190', 'scene_0192', 'scene_0194', 'scene_0196', 'scene_0197', 'scene_0198', 'scene_0199',
		'scene_0200', 'scene_0202', 'scene_0204', 'scene_0206', 'scene_0207', 'scene_0208', 'scene_0209', 'scene_0210', 'scene_0212', 'scene_0214',
		'scene_0216', 'scene_0217', 'scene_0218', 'scene_0219', 'scene_0220', 'scene_0222', 'scene_0224', 'scene_0226', 'scene_0227', 'scene_0228',
		'scene_0229', 'scene_0230', 'scene_0232', 'scene_0234', 'scene_0236', 'scene_0237', 'scene_0238', 'scene_0239', 'scene_0240', 'scene_0242',
		'scene_0244', 'scene_0246', 'scene_0247', 'scene_0248', 'scene_0249', 'scene_0250', 'scene_0252', 'scene_0254', 'scene_0256', 'scene_0257',
		'scene_0258', 'scene_0259', 'scene_0260', 'scene_0262', 'scene_0264', 'scene_0266', 'scene_0267', 'scene_0268', 'scene_0269', 'scene_0270',
		'scene_0272', 'scene_0274', 'scene_0276', 'scene_0277', 'scene_0278', 'scene_0279', 'scene_0280', 'scene_0282', 'scene_0284', 'scene_0286',
		'scene_0287', 'scene_0288', 'scene_0289', 'scene_0290', 'scene_0292', 'scene_0294', 'scene_0296', 'scene_0297', 'scene_0298', 'scene_0299',
		'scene_0300', 'scene_0302', 'scene_0304', 'scene_0306', 'scene_0307', 'scene_0308', 'scene_0309', 'scene_0310', 'scene_0312', 'scene_0314',
		'scene_0316', 'scene_0317', 'scene_0318', 'scene_0319', 'scene_0320', 'scene_0322', 'scene_0324', 'scene_0326', 'scene_0327', 'scene_0328',
		'scene_0329', 'scene_0330', 'scene_0332', 'scene_0334', 'scene_0336', 'scene_0337', 'scene_0338', 'scene_0339', 'scene_0340', 'scene_0342',
		'scene_0344', 'scene_0346', 'scene_0347', 'scene_0348', 'scene_0349', 'scene_0350', 'scene_0352', 'scene_0354', 'scene_0356', 'scene_0357',
		'scene_0358', 'scene_0359', 'scene_0360', 'scene_0362', 'scene_0364', 'scene_0366', 'scene_0367', 'scene_0368', 'scene_0369', 'scene_0370',
		'scene_0372', 'scene_0374', 'scene_0376', 'scene_0377', 'scene_0378', 'scene_0379', 'scene_0380', 'scene_0382', 'scene_0384', 'scene_0386',
		'scene_0387', 'scene_0388', 'scene_0389', 'scene_0390', 'scene_0392', 'scene_0394', 'scene_0396', 'scene_0397', 'scene_0398', 'scene_0399',
		'scene_0400', 'scene_0402', 'scene_0404', 'scene_0406', 'scene_0407', 'scene_0408', 'scene_0409', 'scene_0410', 'scene_0412', 'scene_0414',
		'scene_0416', 'scene_0417', 'scene_0418', 'scene_0419', 'scene_0420', 'scene_0422', 'scene_0424', 'scene_0426', 'scene_0427', 'scene_0428',
		'scene_0429', 'scene_0430', 'scene_0432', 'scene_0434', 'scene_0436', 'scene_0437', 'scene_0438', 'scene_0439', 'scene_0440', 'scene_0442',
		'scene_0444', 'scene_0446', 'scene_0447', 'scene_0448', 'scene_0449', 'scene_0450', 'scene_0452', 'scene_0454', 'scene_0456', 'scene_0457',
		'scene_0458', 'scene_0459', 'scene_0460', 'scene_0462', 'scene_0464', 'scene_0466', 'scene_0467', 'scene_0468', 'scene_0469', 'scene_0470',
		'scene_0472', 'scene_0474', 'scene_0476', 'scene_0477', 'scene_0478', 'scene_0479', 'scene_0480', 'scene_0482', 'scene_0484', 'scene_0486',
		'scene_0487', 'scene_0488', 'scene_0489', 'scene_0490', 'scene_0492', 'scene_0494', 'scene_0496', 'scene_0497', 'scene_0498', 'scene_0499',
		'scene_0500', 'scene_0502', 'scene_0504', 'scene_0506', 'scene_0507', 'scene_0508', 'scene_0509', 'scene_0510', 'scene_0512', 'scene_0514',
		'scene_0516', 'scene_0517', 'scene_0518', 'scene_0519', 'scene_0520', 'scene_0522', 'scene_0524', 'scene_0526', 'scene_0527', 'scene_0528',
		'scene_0529', 'scene_0530', 'scene_0532', 'scene_0534', 'scene_0536', 'scene_0537', 'scene_0538', 'scene_0539', 'scene_0540', 'scene_0542',
		'scene_0544', 'scene_0546', 'scene_0547', 'scene_0548', 'scene_0549', 'scene_0550', 'scene_0552', 'scene_0554', 'scene_0556', 'scene_0557',
		'scene_0558', 'scene_0559', 'scene_0560', 'scene_0562', 'scene_0564', 'scene_0566', 'scene_0567', 'scene_0568', 'scene_0569', 'scene_0570',
		'scene_0572', 'scene_0574', 'scene_0576', 'scene_0577', 'scene_0578', 'scene_0579', 'scene_0580', 'scene_0582', 'scene_0584', 'scene_0586',
		'scene_0587', 'scene_0588', 'scene_0589', 'scene_0590', 'scene_0592', 'scene_0594', 'scene_0596', 'scene_0597', 'scene_0598', 'scene_0599',
		'scene_0600', 'scene_0602', 'scene_0604', 'scene_0606', 'scene_0607', 'scene_0608', 'scene_0609', 'scene_0610', 'scene_0612', 'scene_0614',
		'scene_0616', 'scene_0617', 'scene_0618', 'scene_0619', 'scene_0620', 'scene_0622', 'scene_0624', 'scene_0626', 'scene_0627', 'scene_0628',
		'scene_0629', 'scene_0630', 'scene_0632', 'scene_0634', 'scene_0636', 'scene_0637', 'scene_0638', 'scene_0639', 'scene_0640', 'scene_0642',
		'scene_0644', 'scene_0646', 'scene_0647', 'scene_0648', 'scene_0649', 'scene_0650', 'scene_0652', 'scene_0654', 'scene_0656', 'scene_0657',
		'scene_0658', 'scene_0659', 'scene_0660', 'scene_0662', 'scene_0664', 'scene_0666', 'scene_0667', 'scene_0668', 'scene_0669', 'scene_0670',
		'scene_0672', 'scene_0674', 'scene_0676', 'scene_0677', 'scene_0678', 'scene_0679', 'scene_0680', 'scene_0682', 'scene_0684', 'scene_0686',
		'scene_0687', 'scene_0688', 'scene_0689', 'scene_0690', 'scene_0692', 'scene_0694', 'scene_0696', 'scene_0697', 'scene_0698', 'scene_0699',
		'scene_0700', 'scene_0702', 'scene_0704', 'scene_0706', 'scene_0707', 'scene_0708', 'scene_0709', 'scene_0710', 'scene_0712', 'scene_0714',
		'scene_0716', 'scene_0717', 'scene_0718', 'scene_0719', 'scene_0720', 'scene_0722', 'scene_0724', 'scene_0726', 'scene_0727', 'scene_0728',
		'scene_0729', 'scene_0730', 'scene_0732', 'scene_0734', 'scene_0736', 'scene_0737', 'scene_0738', 'scene_0739', 'scene_0740', 'scene_0742',
		'scene_0744', 'scene_0746', 'scene_0747', 'scene_0748', 'scene_0749', 'scene_0750', 'scene_0752', 'scene_0754', 'scene_0756', 'scene_0757',
		'scene_0758', 'scene_0759', 'scene_0760', 'scene_0762', 'scene_0764', 'scene_0766', 'scene_0767', 'scene_0768', 'scene_0769', 'scene_0770',
		'scene_0772', 'scene_0774', 'scene_0776', 'scene_0777', 'scene_0778', 'scene_0779', 'scene_0780', 'scene_0782', 'scene_0784', 'scene_0786',
		'scene_0787', 'scene_0788', 'scene_0789', 'scene_0790', 'scene_0792', 'scene_0794', 'scene_0796', 'scene_0797', 'scene_0798', 'scene_0799',
		'scene_0800', 'scene_0802', 'scene_0804', 'scene_0806', 'scene_0807', 'scene_0808', 'scene_0809', 'scene_0810', 'scene_0812', 'scene_0814',
		'scene_0816', 'scene_0817', 'scene_0818', 'scene_0819', 'scene_0820', 'scene_0822', 'scene_0824', 'scene_0826', 'scene_0827', 'scene_0828',
		'scene_0829', 'scene_0830', 'scene_0832', 'scene_0834', 'scene_0836', 'scene_0837', 'scene_0838', 'scene_0839', 'scene_0840', 'scene_0842',
		'scene_0844', 'scene_0846', 'scene_0847', 'scene_0848', 'scene_0849', 'scene_0850', 'scene_0852', 'scene_0854', 'scene_0856', 'scene_0857',
		'scene_0858', 'scene_0859', 'scene_0860', 'scene_0862', 'scene_0864', 'scene_0866', 'scene_0867', 'scene_0868', 'scene_0869', 'scene_0870',
		'scene_0872', 'scene_0874', 'scene_0876', 'scene_0877', 'scene_0878', 'scene_0879', 'scene_0880', 'scene_0882', 'scene_0884', 'scene_0886',
		'scene_0887', 'scene_0888', 'scene_0889', 'scene_0890', 'scene_0892', 'scene_0894', 'scene_0896', 'scene_0897', 'scene_0898', 'scene_0899',
		'scene_0900', 'scene_0902', 'scene_0904', 'scene_0906', 'scene_0907', 'scene_0908', 'scene_0909', 'scene_0910', 'scene_0912', 'scene_0914',
		'scene_0916', 'scene_0917', 'scene_0918', 'scene_0919', 'scene_0920', 'scene_0922', 'scene_0924', 'scene_0926', 'scene_0927', 'scene_0928',
		'scene_0929', 'scene_0930', 'scene_0932', 'scene_0934', 'scene_0936', 'scene_0937', 'scene_0938', 'scene_0939', 'scene_0940', 'scene_0942',
		'scene_0944', 'scene_0946', 'scene_0947', 'scene_0948', 'scene_0949', 'scene_0950', 'scene_0952', 'scene_0954', 'scene_0956', 'scene_0957',
		'scene_0958', 'scene_0959', 'scene_0960', 'scene_0962', 'scene_0964', 'scene_0966', 'scene_0967', 'scene_0968', 'scene_0969', 'scene_0970',
		'scene_0972', 'scene_0974', 'scene_0976', 'scene_0977', 'scene_0978', 'scene_0979', 'scene_0980', 'scene_0982', 'scene_0984', 'scene_0986',
		'scene_0987', 'scene_0988', 'scene_0989', 'scene_0990', 'scene_0992', 'scene_0994', 'scene_0996', 'scene_0997', 'scene_0998', 'scene_0999'
   
            ]

val_split = [ 
		'scene_0001', 'scene_0003', 'scene_0005', 'scene_0021', 'scene_0023', 'scene_0025', 'scene_0041', 'scene_0043', 'scene_0045', 'scene_0061',
		'scene_0063', 'scene_0065', 'scene_0081', 'scene_0083', 'scene_0085', 'scene_0101', 'scene_0103', 'scene_0105', 'scene_0121', 'scene_0123',
		'scene_0125', 'scene_0141', 'scene_0143', 'scene_0145', 'scene_0161', 'scene_0163', 'scene_0165', 'scene_0181', 'scene_0183', 'scene_0185',
		'scene_0201', 'scene_0203', 'scene_0205', 'scene_0221', 'scene_0223', 'scene_0225', 'scene_0241', 'scene_0243', 'scene_0245', 'scene_0261',
		'scene_0263', 'scene_0265', 'scene_0281', 'scene_0283', 'scene_0285', 'scene_0301', 'scene_0303', 'scene_0305', 'scene_0321', 'scene_0323',
		'scene_0325', 'scene_0341', 'scene_0343', 'scene_0345', 'scene_0361', 'scene_0363', 'scene_0365', 'scene_0381', 'scene_0383', 'scene_0385',
		'scene_0401', 'scene_0403', 'scene_0405', 'scene_0421', 'scene_0423', 'scene_0425', 'scene_0441', 'scene_0443', 'scene_0445', 'scene_0461',
		'scene_0463', 'scene_0465', 'scene_0481', 'scene_0483', 'scene_0485', 'scene_0501', 'scene_0503', 'scene_0505', 'scene_0521', 'scene_0523',
		'scene_0525', 'scene_0541', 'scene_0543', 'scene_0545', 'scene_0561', 'scene_0563', 'scene_0565', 'scene_0581', 'scene_0583', 'scene_0585',
		'scene_0601', 'scene_0603', 'scene_0605', 'scene_0621', 'scene_0623', 'scene_0625', 'scene_0641', 'scene_0643', 'scene_0645', 'scene_0661',
		'scene_0663', 'scene_0665', 'scene_0681', 'scene_0683', 'scene_0685', 'scene_0701', 'scene_0703', 'scene_0705', 'scene_0721', 'scene_0723',
		'scene_0725', 'scene_0741', 'scene_0743', 'scene_0745', 'scene_0761', 'scene_0763', 'scene_0765', 'scene_0781', 'scene_0783', 'scene_0785',
		'scene_0801', 'scene_0803', 'scene_0805', 'scene_0821', 'scene_0823', 'scene_0825', 'scene_0841', 'scene_0843', 'scene_0845', 'scene_0861',
		'scene_0863', 'scene_0865', 'scene_0881', 'scene_0883', 'scene_0885', 'scene_0901', 'scene_0903', 'scene_0905', 'scene_0921', 'scene_0923',
		'scene_0925', 'scene_0941', 'scene_0943', 'scene_0945', 'scene_0961', 'scene_0963', 'scene_0965', 'scene_0981', 'scene_0983', 'scene_0985'
            ]

test_split = [ 
		'scene_0011', 'scene_0013', 'scene_0015', 'scene_0031', 'scene_0033', 'scene_0035', 'scene_0051', 'scene_0053', 'scene_0055', 'scene_0071',
		'scene_0073', 'scene_0075', 'scene_0091', 'scene_0093', 'scene_0095', 'scene_0111', 'scene_0113', 'scene_0115', 'scene_0131', 'scene_0133',
		'scene_0135', 'scene_0151', 'scene_0153', 'scene_0155', 'scene_0171', 'scene_0173', 'scene_0175', 'scene_0191', 'scene_0193', 'scene_0195',
		'scene_0211', 'scene_0213', 'scene_0215', 'scene_0231', 'scene_0233', 'scene_0235', 'scene_0251', 'scene_0253', 'scene_0255', 'scene_0271',
		'scene_0273', 'scene_0275', 'scene_0291', 'scene_0293', 'scene_0295', 'scene_0311', 'scene_0313', 'scene_0315', 'scene_0331', 'scene_0333',
		'scene_0335', 'scene_0351', 'scene_0353', 'scene_0355', 'scene_0371', 'scene_0373', 'scene_0375', 'scene_0391', 'scene_0393', 'scene_0395',
		'scene_0411', 'scene_0413', 'scene_0415', 'scene_0431', 'scene_0433', 'scene_0435', 'scene_0451', 'scene_0453', 'scene_0455', 'scene_0471',
		'scene_0473', 'scene_0475', 'scene_0491', 'scene_0493', 'scene_0495', 'scene_0511', 'scene_0513', 'scene_0515', 'scene_0531', 'scene_0533',
		'scene_0535', 'scene_0551', 'scene_0553', 'scene_0555', 'scene_0571', 'scene_0573', 'scene_0575', 'scene_0591', 'scene_0593', 'scene_0595',
		'scene_0611', 'scene_0613', 'scene_0615', 'scene_0631', 'scene_0633', 'scene_0635', 'scene_0651', 'scene_0653', 'scene_0655', 'scene_0671',
		'scene_0673', 'scene_0675', 'scene_0691', 'scene_0693', 'scene_0695', 'scene_0711', 'scene_0713', 'scene_0715', 'scene_0731', 'scene_0733',
		'scene_0735', 'scene_0751', 'scene_0753', 'scene_0755', 'scene_0771', 'scene_0773', 'scene_0775', 'scene_0791', 'scene_0793', 'scene_0795',
		'scene_0811', 'scene_0813', 'scene_0815', 'scene_0831', 'scene_0833', 'scene_0835', 'scene_0851', 'scene_0853', 'scene_0855', 'scene_0871',
		'scene_0873', 'scene_0875', 'scene_0891', 'scene_0893', 'scene_0895', 'scene_0911', 'scene_0913', 'scene_0915', 'scene_0931', 'scene_0933',
		'scene_0935', 'scene_0951', 'scene_0953', 'scene_0955', 'scene_0971', 'scene_0973', 'scene_0975', 'scene_0991', 'scene_0993', 'scene_0995'
             ]









UAV_NameMapping = {}
UAV_NameMapping['vehicle'] = 'vehicle'
UAV_NameMapping['vehicle'] = 'car'

def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = train_split
        val_scenes = val_split
    elif version == 'v1.0-test':
        train_scenes = test_split
        val_scenes = []
    elif version == 'v1.0-mini':
        # train_scenes = splits.mini_train
        # val_scenes = splits.mini_val
        train_scenes = train_split
        val_scenes = val_split

    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        # sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        # has_more_frames = True
        # scene_not_exist = False
        # while has_more_frames:
        #     lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
        #     lidar_path = str(lidar_path)
        #     if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                # lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            # if not mmcv.is_filepath(lidar_path):
            #     scene_not_exist = True
            #     break
            # else:
            #     break
        # if scene_not_exist:
        #     continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):

        # lidar_token = sample['data']['LIDAR_TOP']
        # sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        # cs_record = nusc.get('calibrated_sensor',
        #                      sd_rec['calibrated_sensor_token'])
        # pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        # lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        # mmcv.check_file_exist(lidar_path)
        lidar_path = ''

        sd_rec = nusc.get('sample_data', sample['data'][center_cam])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        ego_pos = cs_record['translation'].copy()
        ego_pos[2] = 0


        cs_record = { }
        cs_record['translation'] = [0.0, 0.0, 0.0]
        cs_record['rotation'] = [1.0, 0.0, 0.0, 0.0]

        pose_record = { }
        # pose_record['translation'] =  [0.0, 0.0, 0.0]
        pose_record['translation'] = ego_pos
        pose_record['rotation'] =   [1.0, 0.0, 0.0, 0.0]

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix



        # obtain 6 image's information per frame
        # camera_types = [
        #     'CAM_FRONT',
        #     'CAM_FRONT_RIGHT',
        #     'CAM_FRONT_LEFT',
        #     'CAM_BACK',
        #     'CAM_BACK_LEFT',
        #     'CAM_BACK_RIGHT',
        # ]

        camera_types = ['CAMERA_FRONT_id_0', 'CAMERA_BACK_id_0', 'CAMERA_LEFT_id_0', 'CAMERA_RIGHT_id_0', 'CAMERA_BOTTOM_id_0',
                   'CAMERA_FRONT_id_1', 'CAMERA_BACK_id_1', 'CAMERA_LEFT_id_1', 'CAMERA_RIGHT_id_1', 'CAMERA_BOTTOM_id_1',
                   'CAMERA_FRONT_id_2', 'CAMERA_BACK_id_2', 'CAMERA_LEFT_id_2', 'CAMERA_RIGHT_id_2', 'CAMERA_BOTTOM_id_2',
                   'CAMERA_FRONT_id_3', 'CAMERA_BACK_id_3', 'CAMERA_LEFT_id_3', 'CAMERA_RIGHT_id_3', 'CAMERA_BOTTOM_id_3',
                   'CAMERA_FRONT_id_4', 'CAMERA_BACK_id_4', 'CAMERA_LEFT_id_4', 'CAMERA_RIGHT_id_4', 'CAMERA_BOTTOM_id_4']

        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

            # if cam is center_cam:
            #     sd_rec_1 = nusc.get('sample_data', cam_token)
            #     cs_record_1 = nusc.get('calibrated_sensor',
            #                          sd_rec_1['calibrated_sensor_token'])
            #     center_pos = cs_record_1['translation'].copy()
            #     center_pos[2] = 0


        # obtain sweeps for a single key-frame
        # sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        # while len(sweeps) < max_sweeps:
        #     if not sd_rec['prev'] == '':
        #         sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
        #                                   l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
        #         sweeps.append(sweep)
        #         sd_rec = nusc.get('sample_data', sd_rec['prev'])
        #     else:
        #         break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]

            boxes = [ ]
            for anno in annotations:
                center, size, rotation, name, token   = anno['translation'], anno['size'], \
                                    anno['rotation'], anno['category_name'], anno['token']

                center = np.array(center) - np.array(ego_pos)
                if center[0] > x_min and center[0] < x_max and center[1] > y_min and center[1] < y_max and \
                    center[2] > z_min and center[2] < z_max:
                    center = center.tolist()

                    # angles_1 = quat2euler(rotation)
                    # angles_1 = [angle * 180 / (np.pi) for angle in angles_1]
                    # (roll, pitch, yaw) ---> (yaw, pitch, roll)
                    # yaw_pitch_roll = [angles_1[2], angles_1[1], angles_1[0]]
                    # box_0 = Box(center, size, yaw_pitch_roll, name=name, token=token)

                    box = Box(center, size, Quaternion(rotation), name=name, token=token)
                    boxes.append(box)

            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            # (roll, pitch, yaw)
            rots = np.array([quat2euler(b.orientation.q)[2]
                             for b in boxes]).reshape(-1, 1)
            # rots = np.array([b.orientation.yaw_pitch_roll[0]
            #                  for b in boxes]).reshape(-1, 1)

            # pitch = np.array([b.orientation.yaw_pitch_roll[1]
            #                  for b in boxes]).reshape(-1, 1)
            # roll = np.array([b.orientation.yaw_pitch_roll[2]
            #                   for b in boxes]).reshape(-1, 1)

            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])

            velocity = velocity[:len(boxes),:]

            # valid_flag = np.array(
            #     [(anno['num_lidar_pts'] + anno['num_radar_pts']) >= 0
            #      for anno in annotations],
            #     dtype=bool).reshape(-1)

            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) != 0
                 for anno in annotations],
                dtype=bool).reshape(-1)

            valid_flag = valid_flag[:len(boxes)]

            # convert velo from global to lidar
            # for i in range(len(boxes)):
            #     velo = np.array([*velocity[i], 0.0])
            #     velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
            #         l2e_r_mat).T
            #     velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            if len(names) < 1:
                print('no ground truth :')
            for i in range(len(names)):
                splits = names[i].split('.')
                if splits[0] in UAV_NameMapping:
                    names[i] = UAV_NameMapping[splits[0]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

            print(len(gt_boxes))

            # assert len(gt_boxes) == len(
            #     annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])[:len(boxes)]
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])[:len(boxes)]
            info['valid_flag'] = valid_flag

            info['location'] = 'Atlanta'

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        if sample['scene_token'] in val_scenes:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path

    pose_record['translation'] =  e2g_t

    cs_record['translation'][0] -= e2g_t[0]
    cs_record['translation'][1] -= e2g_t[1]
    cs_record['translation'][2] -= e2g_t[2]

    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    # l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    # e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    # test1 = Quaternion(l2e_r_s).yaw_pitch_roll
    # test1 = [angle * 180 / (np.pi) for angle in test1]

    # transforms3d
    # l2e_r_s_mat_1 = quat2mat(l2e_r_s)
    # e2g_r_s_mat_1 = quat2mat(e2g_r_s)

    # carla
    angles_1 = quat2euler(l2e_r_s)
    angles_2 = quat2euler(e2g_r_s)
    angles_1 = [angle * 180 / (np.pi) for angle in angles_1]
    angles_2 = [angle * 180 / (np.pi) for angle in angles_2]
    #(roll, pitch, yaw) ---> (pitch, yaw, roll)
    l2e_r_s_mat = get_matrix(pitch=angles_1[1], yaw=angles_1[2], roll=angles_1[0])
    e2g_r_s_mat = get_matrix(pitch=angles_2[1], yaw=angles_2[2], roll=angles_2[0])


    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec

# https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/client_bounding_boxes.py
def get_matrix(pitch=0, yaw=0, roll=0):
    """
    Creates matrix from carla transform.
    """

    # rotation = transform.rotation
    # location = transform.location
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    # matrix = np.matrix(np.identity(3))
    matrix = np.identity(3)
    # matrix[0, 3] = location.x
    # matrix[1, 3] = location.y
    # matrix[2, 3] = location.z
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
