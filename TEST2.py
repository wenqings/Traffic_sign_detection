import cv2
import re
import os
import random
from tkinter import filedialog
from tkinter import *

import cv2
from PIL import Image
import os
from os import walk
import numpy as np
import imutils
import math
from xml.dom import minidom
import xml.etree.ElementTree as ET




annotation = ET.Element("annotation")



for i in range(10):
    object = ET.SubElement(annotation, "object")
    name = ET.SubElement(object, "name")
    pose = ET.SubElement(object, "pose")
    truncated = ET.SubElement(object, "truncated")
    difficult = ET.SubElement(object, "difficult")
    bndbox = ET.SubElement(object, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    ymin = ET.SubElement(bndbox, "ymin")
    xmax = ET.SubElement(bndbox, "xmax")
    ymax = ET.SubElement(bndbox, "ymax")

    name.text = str(i)+'name'
    pose.text = str(i)+'pose'
    truncated.text = str(i)+'truncated'
    difficult.text = str(i)+'difficult'
    xmin.text = str(i)+'xmin'
    ymin.text = str(i)+'ymin'
    xmax.text = str(i)+'xmax'
    ymax.text = str(i)+'ymax'

XML_data = ET.tostring(annotation)
print(XML_data)
file = open('TEST.XML', "w")
file.write(str(XML_data)[2:len(str(XML_data)) - 1])