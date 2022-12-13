#!/usr/bin/env python2
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, ros_inference_detector

import sys, re
import numpy as np
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages")

import rospy
from pyquaternion import Quaternion
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from numpy.lib.recfunctions import structured_to_unstructured
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
import math
import argparse
import torch

sys.path.append("~/Desktop/ROS-tracker/catkin_ws/src:/opt/ros/melodic/share")
sys.path.append("~/Desktop/mmdetection3d")

sys.path.append("/home/khushdeep/Desktop/3D-Multi-Object-Tracker")
from tracker.config import cfg, cfg_from_yaml_file
from zoe_3DMOT import Track_seq,save_seq

sys.path.append("/home/khushdeep/Desktop/3D-Detection-Tracking-Viewer")
from zoe_3D_tracking_viewer import zoe_viewer

sys.path.append("/home/khushdeep/Desktop/3D-Detection-Tracking-Viewer")
from zoe_3D_tracking_viewer import zoe_viewer

import tf
from tf2_ros import Buffer,TransformListener
from tf2_msgs.msg import TFMessage
from message_filters import Subscriber,TimeSynchronizer,ApproximateTimeSynchronizer


# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

DUMMY_FIELD_PREFIX = '__'

class SecondROS:
    def __init__(self):
        rospy.init_node('second_ros')
        self.scenenum=1
        self.tf_listener = tf.TransformListener()
        # Tracking
        self.camera_info_msg = rospy.wait_for_message("/zoe/camera_front/camera_out/camera_info", CameraInfo)
        self.cam_P = np.array(self.camera_info_msg.P).reshape((3, 4))
        print(self.cam_P)
        now = rospy.Time.now()
        self.tf_listener.waitForTransform('zoe/camera_front', 'zoe/velodyne', rospy.Time(0), rospy.Duration(5.0))
        (trans_camera, rot_camera) = self.tf_listener.lookupTransform('zoe/camera_front', 'zoe/velodyne',rospy.Time(0))
        print(f'translation_camera = {trans_camera}')
        self.velo2Cam = np.asarray(self.tf_listener.fromTranslationRotation(trans_camera, rot_camera))
        print(f'velo2cam = {self.velo2Cam}')
        print(f'Camera translation={trans_camera}')

        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str,
                            default="/home/khushdeep/Desktop/3D-Multi-Object-Tracker/config/online"
                                    "/zoe.yaml",
                            help='specify the config for tracking')
        args = parser.parse_args()
        yaml_file = args.cfg_file

        self.config = cfg_from_yaml_file(yaml_file, cfg)
        self.Tracker = Track_seq(self.config)

        parser = ArgumentParser()
        parser.add_argument('--pcd', 
            default='demo/data/kitti/kitti_000008.bin',
            help='Point cloud file')       
        parser.add_argument('--config', 
            # default='configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py',
            # default='configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py',
            # default='configs/point_rcnn/point_rcnn_2x8_kitti-3d-3classes.py',  # car boxes are not displayed
            # default='configs/3dssd/3dssd_4x4_kitti-3d-car.py',
            default='configs/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d.py',
            help='Config file')
        parser.add_argument('--checkpoint', 
            # default='checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth',
            # default='checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth',
            # default='checkpoints/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth', # car boxes are not displayed
            # default='checkpoints/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth',
            default='checkpoints/nuscenes/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090.pth',
            help='Checkpoint file')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--score-thr', type=float, default=0.3, help='bbox score threshold')
            # 0.15 - 3dssd (as expected), 0.25 - pointpillar, 0.2 - pointrcnn
        parser.add_argument(
            '--out-dir', type=str, default='demo', help='dir to save results')
        parser.add_argument(
            '--show',
            default=True,
            action='store_true',
            help='show online visualization results')
        parser.add_argument(
            '--snapshot',
            default=True,
            action='store_true',
            help='whether to save online visualization results')
        self.args = parser.parse_args()
        # build the model from a config file and a checkpoint file
        self.model = init_model(self.args.config, self.args.checkpoint, device=self.args.device)

        # Subscriber
        self.sub_lidar = rospy.Subscriber("/zoe/velodyne_points", PointCloud2, self.lidar_callback, queue_size=1)
        self.sub_image = rospy.Subscriber("/zoe/camera_front/image_color", Image, self.image_callback)

        # Publisher
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=1)



        rospy.spin()
    
    def image_callback(self, image_data):
        # br = CvBridge()
        # img = br.imgmsg_to_cv2(image_data)
        img = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        zoe_viewer(img, self.track_obj, self.velo2Cam, self.cam_P)
    
    def lidar_callback(self, msg):

        self.tf_listener.waitForTransform('zoe/world', 'zoe/velodyne', rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = self.tf_listener.lookupTransform('zoe/world', 'zoe/velodyne', rospy.Time(0))
        (trans_inv, rot_inv) = self.tf_listener.lookupTransform('zoe/velodyne', 'zoe/world', rospy.Time(0))
        # print(trans)
        ego_pose= np.asarray(self.tf_listener.fromTranslationRotation(trans,rot))
        world_2_ego = np.asarray(self.tf_listener.fromTranslationRotation(trans_inv,rot_inv))
        self.scenenum += 1
        # print(f'Matrix:{matrix}')

        intensity_fname = None
        intensity_dtype = None
        for field in msg.fields:
            if field.name == "i" or field.name == "intensity":
                intensity_fname = field.name
                intensity_dtype = field.datatype
            
        dtype_list = self._fields_to_dtype(msg.fields, msg.point_step)
        pc_arr = np.frombuffer(msg.data, dtype_list)
        
        if intensity_fname:
            pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z", intensity_fname]]).copy()
            pc_arr[:, 3] = pc_arr[:, 3] / 255
        else:
            pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z"]]).copy()
            pc_arr = np.hstack((pc_arr, np.zeros((pc_arr.shape[0], 1))))

        if 'nuscenes' in self.args.checkpoint:
            pc_arr = np.hstack((pc_arr, np.zeros((pc_arr.shape[0], 1))))

        # Passing the zoe pointcloud 
        result, data = ros_inference_detector(self.model, pc_arr)
        lidar_boxes = []

        if 'nuscenes' in self.args.checkpoint:
            lidar_boxes.append( {'pred_boxes':result[0]['pts_bbox']['boxes_3d'].tensor.numpy(),
                            'pred_labels': result[0]['pts_bbox']['labels_3d'].numpy(),
                            'pred_scores': result[0]['pts_bbox']['scores_3d'].numpy()
            })
        else:
            lidar_boxes.append( {'pred_boxes':result[0]['boxes_3d'].tensor.numpy(),
                        'pred_labels': result[0]['labels_3d'].numpy(),
                        'pred_scores': result[0]['scores_3d'].numpy()
        })

        ####Test code
        lidar_boxes[0]['pred_boxes'] = torch.tensor([[10.0, 0.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57],
                                                     [-10.0, 0.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57],
                                                     [0.0, 10.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57],
                                                     [0.0, -10.0, -1.0, 3.7428, 1.6276, 1.5223, 1.57]])
        lidar_boxes[0]['pred_scores'] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        lidar_boxes[0]['pred_labels'] = torch.tensor([1, 1, 1, 1], dtype=int)

        detec_shape = lidar_boxes[0]['pred_boxes'].shape[0]
        print(f'detection shape={detec_shape}')
        #####

        self.tracker(lidar_boxes, ego_pose,world_2_ego)
        self.plot_bbox_lidar(lidar_boxes,msg)


    def plot_bbox_lidar(self,lidar_boxes,msg):

        for batch_id in range(len(lidar_boxes)):
            if self.args.score_thr is not None:
                    mask = lidar_boxes[batch_id]['pred_scores'] > self.args.score_thr
                    lidar_boxes[batch_id]['pred_boxes'] = lidar_boxes[batch_id]['pred_boxes'][mask]
                    lidar_boxes[batch_id]['pred_scores'] = lidar_boxes[batch_id]['pred_scores'][mask]   

        if lidar_boxes is not None:

            num_detects = lidar_boxes[0]['pred_boxes'].shape[0]
            arr_bbox = BoundingBoxArray()

            for i in range(num_detects):
                bbox = BoundingBox()

                bbox.header.frame_id = msg.header.frame_id
                bbox.header.stamp = rospy.Time.now()
                
                bbox.pose.position.x = float(lidar_boxes[0]['pred_boxes'][i][0])
                bbox.pose.position.y = float(lidar_boxes[0]['pred_boxes'][i][1])
                bbox.pose.position.z = float(lidar_boxes[0]['pred_boxes'][i][2])

                bbox.dimensions.x = float(lidar_boxes[0]['pred_boxes'][i][3])  # width
                bbox.dimensions.y = float(lidar_boxes[0]['pred_boxes'][i][4])  # length
                bbox.dimensions.z = float(lidar_boxes[0]['pred_boxes'][i][5])  # height
                
                if 'nuscenes' in self.args.checkpoint:
                    q = Quaternion(axis=(0, 0, 1), radians=float(lidar_boxes[0]['pred_boxes'][i][6])+np.pi/2)
                else:
                    q = Quaternion(axis=(0, 0, 1), radians=float(lidar_boxes[0]['pred_boxes'][i][6]))

                bbox.pose.orientation.x = q.x
                bbox.pose.orientation.y = q.y
                bbox.pose.orientation.z = q.z
                bbox.pose.orientation.w = q.w

                if int(lidar_boxes[0]['pred_labels'][i]) == 1: # change for Car label
                    arr_bbox.boxes.append(bbox)
                    bbox.label = i
                    bbox.value = i
        
            arr_bbox.header.frame_id = msg.header.frame_id
            arr_bbox.header.stamp = rospy.Time.now()
            print(f"Scence num: {self.scenenum} Number of detections: {num_detects}")

            self.pub_bbox.publish(arr_bbox)

        else:
            boxes = None

    def tracker(self,lidar_boxes,ego_pose,world_2_ego):

        if lidar_boxes is not None:
            boxes = lidar_boxes[0]['pred_boxes'].cpu().numpy()
            boxes_coord = np.append(boxes[:, 0:3], np.ones((boxes.shape[0], 1)), axis=1)
            scores = lidar_boxes[0]['pred_scores'].cpu().numpy()
            boxes_world = np.dot(ego_pose, boxes_coord.T).T
            boxes[:, 0:3] = boxes_world[:, 0:3]

        else:
            boxes = None
            scores = None

        tracker = self.Tracker.track_scene(boxes, scores, self.scenenum)
        self.track_obj = save_seq(tracker,self.config, world_2_ego,self.velo2Cam,self.cam_P)

    def _fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        return np_dtype_list

if __name__ == '__main__':
    second_ros = SecondROS()
