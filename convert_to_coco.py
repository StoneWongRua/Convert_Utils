# -*- coding: utf-8 -*-
import argparse
import json
import matplotlib.pyplot as plt
import xml.dom.minidom as xmldom
import cv2
import numpy as np
import glob
import PIL.Image
import os,sys
import operator
class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path=''):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.save_json()

    def data_transfer(self):
        for num, xmlfile in enumerate(self.xml):
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()
            self.xmlfile = xmlfile
            self.num = num
            path = os.path.dirname(os.path.dirname(self.xmlfile))
            obj_path = glob.glob(os.path.join(path, 'JPEGImages(val)', '*.jpg'))
            DomTree = xmldom.parse(xmlfile)
            annotation = DomTree.documentElement
            filenamelist = annotation.getElementsByTagName('filename')
            widthlist = annotation.getElementsByTagName('width')
            heightlist = annotation.getElementsByTagName('height')
            objectlist = annotation.getElementsByTagName('object')

            filename = (filenamelist[0].childNodes[0].data).encode('unicode-escape').decode('string_escape')
            self.filen_ame = filename[:-4] + '.jpg'
            self.width = int(widthlist[0].childNodes[0].data)
            self.height = int(heightlist[0].childNodes[0].data)
            self.images.append(self.image())
            self.path = os.path.join(path, 'JPEGImages(val)', self.filen_ame)
            # if self.path not in obj_path:
            #     break
            for objects in objectlist:
                namelist = objects.getElementsByTagName('name')
                self.name = (namelist[0].childNodes[0].data).encode('unicode-escape').decode('string_escape')
                if self.name == 'box':
                    self.supercategory = 'person'
                elif self.name == 'fileholder' or self.name == 'plasticbag' or self.name == 'wovenbag':
                    self.supercategory = 'vehicle'
                if self.name not in self.label:
                    self.categories.append(self.categorie())
                    self.label.append(self.name)
                    print(self.label)
                bndbox = objects.getElementsByTagName('bndbox')
                for box in bndbox:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(x1_list[0].childNodes[0].data)
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(y1_list[0].childNodes[0].data)
                    x2_list = box.getElementsByTagName('xmax')
                    x2 = int(x2_list[0].childNodes[0].data)
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(y2_list[0].childNodes[0].data)
                    w = x2 - x1
                    h = y2 - y1
                    #self.rectangle = [x1, y1, x2, y2]  #用于计算segmentation
                    self.bbox = [x1, y1, w, h]  # COCO 对应格式[x,y,w,h]
                    self.annotations.append(self.annotation())
                    self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        #categorie['id'] = len(self.label) + 1  # 0 默认为背景
        if self.name == 'box':
            categorie['id'] = 1
            categorie['supercategory'] = 'person'
        elif self.name == 'fileholder':
            categorie['id'] = 2
            categorie['supercategory'] = 'vehicle'
        elif self.name == 'plasticbag':
            categorie['id'] = 3
            categorie['supercategory'] = 'vehicle'
        elif self.name == 'wovenbag':
            categorie['id'] = 4
            categorie['supercategory'] = 'vehicle'
        else:
            categorie['id'] = 0
        print (categorie['id'])
        categorie['name'] = self.name
        return categorie

    def annotation(self):
        annotation = {}
        # annotation['segmentation'] = [self.getsegmentation()]
        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        annotation['area'] = self.bbox[2]*self.bbox[3]
        annotation['bbox'] = list(map(float, self.bbox))
        #annotation['bbox'] = self.bbox
        if self.name == 'box':
            annotation['category_id'] = 1
        elif self.name == 'fileholder':
            annotation['category_id'] = 2
        elif self.name == 'plasticbag':
            annotation['category_id'] = 3
        elif self.name == 'wovenbag':
            annotation['category_id'] = 4
        else:
            annotation['category_id'] = 0

        #annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox=[]
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox # list(contours[1][0].flatten())

    # '''
    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    # '''
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['annotations'] = self.annotations
        data_coco['categories'] = sorted(self.categories, key=operator.itemgetter('name'))
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示
        print('done')


xml_file = glob.glob('/home/manager/pt/object_detection/coco_hst/annotations/*.xml')
save_json_path = '/home/manager/pt/object_detection/coco_hst/annotations/box_voc_val.json'
PascalVOC2coco(xml_file, save_json_path)