# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Dict, List

from nuscenes import NuScenes


train_detect = [
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

train_track = [
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

train = [
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

val = [ 
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

test = [ 
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

mini_train = ['scene_00', 'scene_03', 'scene_04', 'scene_05',
                'scene_06', 'scene_07', 'scene_08', 'scene_09', 'scene_10',
                'scene_13', 'scene_14', 'scene_15', 'scene_16', 'scene_17',
                'scene_18', 'scene_19']

mini_val = ['scene_01','scene_02','scene_11','scene_12']
mini_test = ['scene_01','scene_02','scene_11','scene_12']


def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    #assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val,
                    'train_detect': train_detect, 'train_track': train_track}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
