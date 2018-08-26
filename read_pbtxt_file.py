#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:42:50 2018

@author: shirhe-lyh
"""

"""A tool to read .pbtxt file.

See Details at:
    TensorFlow models/research/object_detetion/protos/string_int_label_pb2.py
    TensorFlow models/research/object_detection/utils/label_map_util.py
"""

import tensorflow as tf

from google.protobuf import text_format

import string_int_label_map_pb2


def load_pbtxt_file(path):
    """Read .pbtxt file.
    
    Args: 
        path: Path to StringIntLabelMap proto text file (.pbtxt file).
        
    Returns:
        A StringIntLabelMapProto.
        
    Raises:
        ValueError: If path is not exist.
    """
    if not tf.gfile.Exists(path):
        raise ValueError('`path` is not exist.')
        
    with tf.gfile.GFile(path, 'r') as fid:
        pbtxt_string = fid.read()
        pbtxt = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(pbtxt_string, pbtxt)
        except text_format.ParseError:
            pbtxt.ParseFromString(pbtxt_string)
    return pbtxt


def get_label_map_dict(path):
    """Reads a .pbtxt file and returns a dictionary.
    
    Args:
        path: Path to StringIntLabelMap proto text file.
        
    Returns:
        A dictionary mapping class names to indices.
    """
    pbtxt = load_pbtxt_file(path)
    
    result_dict = {}
    for item in pbtxt.item:
        result_dict[item.name] = item.id
    return result_dict
        