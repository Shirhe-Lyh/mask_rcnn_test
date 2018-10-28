#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 10:57:09 2018

@author: shirhe-lyh
"""

"""Convert raw dataset to TFRecord for object_detection.

Please note that this tool only applies to labelme's annotations(json file).

Example usage:
    python3 create_tf_record.py \
        --images_dir=your absolute path to read images.
        --annotations_json_dir=your path to annotaion json files.
        --label_map_path=your path to label_map.pbtxt
        --output_path=your path to write .record.
"""

import cv2
import glob
import hashlib
import io
import json
import numpy as np
import os
import PIL.Image
import tensorflow as tf

import read_pbtxt_file


flags = tf.app.flags

flags.DEFINE_string('images_dir', None, 'Path to images directory.')
flags.DEFINE_string('annotations_json_dir', 'datasets/annotations', 
                    'Path to annotations directory.')
flags.DEFINE_string('label_map_path', None, 'Path to label map proto.')
flags.DEFINE_string('output_path', None, 'Path to the output tfrecord.')

FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(annotation_dict, label_map_dict=None):
    """Converts image and annotations to a tf.Example proto.
    
    Args:
        annotation_dict: A dictionary containing the following keys:
            ['height', 'width', 'filename', 'sha256_key', 'encoded_jpg',
             'format', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'masks',
             'class_names'].
        label_map_dict: A dictionary maping class_names to indices.
            
    Returns:
        example: The converted tf.Example.
        
    Raises:
        ValueError: If label_map_dict is None or is not containing a class_name.
    """
    if annotation_dict is None:
        return None
    if label_map_dict is None:
        raise ValueError('`label_map_dict` is None')
        
    height = annotation_dict.get('height', None)
    width = annotation_dict.get('width', None)
    filename = annotation_dict.get('filename', None)
    sha256_key = annotation_dict.get('sha256_key', None)
    encoded_jpg = annotation_dict.get('encoded_jpg', None)
    image_format = annotation_dict.get('format', None)
    xmins = annotation_dict.get('xmins', None)
    xmaxs = annotation_dict.get('xmaxs', None)
    ymins = annotation_dict.get('ymins', None)
    ymaxs = annotation_dict.get('ymaxs', None)
    masks = annotation_dict.get('masks', None)
    class_names = annotation_dict.get('class_names', None)
    
    labels = []
    for class_name in class_names:
        label = label_map_dict.get(class_name, 'None')
        if label is None:
            raise ValueError('`label_map_dict` is not containing {}.'.format(
                class_name))
        labels.append(label)
            
    encoded_masks = []
    for mask in masks:
        pil_image = PIL.Image.fromarray(mask.astype(np.uint8))
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_masks.append(output_io.getvalue())
        
    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(filename.encode('utf8')),
        'image/key/sha256': bytes_feature(sha256_key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/mask': bytes_list_feature(encoded_masks),
        'image/object/class/label': int64_list_feature(labels)}
    example = tf.train.Example(features=tf.train.Features(
        feature=feature_dict))
    return example


def _get_annotation_dict(images_dir, annotation_json_path):  
    """Get boundingboxes and masks.
    
    Args:
        images_dir: Path to images directory.
        annotation_json_path: Path to annotated json file corresponding to
            the image. The json file annotated by labelme with keys:
                ['lineColor', 'imageData', 'fillColor', 'imagePath', 'shapes',
                 'flags'].
            
    Returns:
        annotation_dict: A dictionary containing the following keys:
            ['height', 'width', 'filename', 'sha256_key', 'encoded_jpg',
             'format', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'masks',
             'class_names'].
#            
#    Raises:
#        ValueError: If images_dir or annotation_json_path is not exist.
    """
#    if not os.path.exists(images_dir):
#        raise ValueError('`images_dir` is not exist.')
#    
#    if not os.path.exists(annotation_json_path):
#        raise ValueError('`annotation_json_path` is not exist.')
    
    if (not os.path.exists(images_dir) or
        not os.path.exists(annotation_json_path)):
        return None
    
    with open(annotation_json_path, 'r') as f:
        json_text = json.load(f)
    shapes = json_text.get('shapes', None)
    if shapes is None:
        return None
    image_relative_path = json_text.get('imagePath', None)
    if image_relative_path is None:
        return None
    image_name = image_relative_path.split('/')[-1]
    image_path = os.path.join(images_dir, image_name)
    image_format = image_name.split('.')[-1].replace('jpg', 'jpeg')
    if not os.path.exists(image_path):
        return None
    
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = cv2.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    masks = []
    class_names = []
    hole_polygons = []
    for mark in shapes:
        class_name = mark.get('label')
        class_names.append(class_name)
        polygon = mark.get('points')
        polygon = np.array(polygon)
        if class_name == 'hole':
            hole_polygons.append(polygon)
        else:
            mask = np.zeros(image.shape[:2])
            cv2.fillPoly(mask, [polygon], 1)
            masks.append(mask)
            
            # Boundingbox
            x = polygon[:, 0]
            y = polygon[:, 1]
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            xmins.append(float(xmin) / width)
            xmaxs.append(float(xmax) / width)
            ymins.append(float(ymin) / height)
            ymaxs.append(float(ymax) / height)
    # Remove holes in mask
    for mask in masks:
        mask = cv2.fillPoly(mask, hole_polygons, 0)
        
    annotation_dict = {'height': height,
                       'width': width,
                       'filename': image_name,
                       'sha256_key': key,
                       'encoded_jpg': encoded_jpg,
                       'format': image_format,
                       'xmins': xmins,
                       'xmaxs': xmaxs,
                       'ymins': ymins,
                       'ymaxs': ymaxs,
                       'masks': masks,
                       'class_names': class_names}
    return annotation_dict


def main(_):
    if not os.path.exists(FLAGS.images_dir):
        raise ValueError('`images_dir` is not exist.')
    if not os.path.exists(FLAGS.annotations_json_dir):
        raise ValueError('`annotations_json_dir` is not exist.')
    if not os.path.exists(FLAGS.label_map_path):
        raise ValueError('`label_map_path` is not exist.')
        
    label_map = read_pbtxt_file.get_label_map_dict(FLAGS.label_map_path)
    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
        
    num_annotations_skiped = 0
    annotations_json_path = os.path.join(FLAGS.annotations_json_dir, '*.json')
    for i, annotation_file in enumerate(glob.glob(annotations_json_path)):
        if i % 100 == 0:
            print('On image %d', i)
            
        annotation_dict = _get_annotation_dict(
            FLAGS.images_dir, annotation_file)
        if annotation_dict is None:
            num_annotations_skiped += 1
            continue
        tf_example = create_tf_example(annotation_dict, label_map)
        writer.write(tf_example.SerializeToString())
    
    print('Successfully created TFRecord to {}.'.format(FLAGS.output_path))


if __name__ == '__main__':
    tf.app.run()
                
