import cv2
import numpy as np
import argparse
import os
import signal


label_list = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
color_list = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

parser = argparse.ArgumentParser()
parser.add_argument("--dst", type = str , default = "test_predictions/")
parser.add_argument("--path", type = str , default = "../data/3obj/")
parser.add_argument("--number", type = int , default = 200)
parser.add_argument("--visualize", type = int , default = 1)


args = parser.parse_args()
IMG_NUM = args.number


def keyBoardINT(signum, frame):  
    print ('You pressed Ctrl+C!')  
    os.kill(os.getpid(), signal.SIGKILL)
    exit(0)  

class Accuracy(object):

    
    def __init__(self, path, dst):
        self.path = path
        self.dst = dst
        self.img_id = 0
        self.accuracy_sum = 0

    def loadPicture(self, path, dst):
        self.label = cv2.imread(path+'testannot/'+str(self.img_id)+'.png')
        self.predict = cv2.imread(path+dst+str(self.img_id)+'.png')

    def accuracyCalculate(self, size, label, color):
        for i in range(size):
            error = 0
            self.loadPicture(self.path, self.dst)
            if(self.label.shape[0] != self.predict.shape[0]
                    or self.label.shape[1] != self.predict.shape[1]):
                self.label = self.resize(self.label, self.predict)
            self.error_map = self.label.copy()

            for i in range(self.label.shape[0]):
                for j in range(self.label.shape[1]):
                    if (((self.label[i, j, :] == label_list[0]).all() and (self.predict[i, j, :] == color_list[0]).all())
                            or ((self.label[i, j, :] == label_list[1]).all() and (self.predict[i, j, :] == color_list[1]).all())
                            or ((self.label[i, j, :] == label_list[2]).all() and (self.predict[i, j, :] == color_list[2]).all())
                            or ((self.label[i, j, :] == label_list[3]).all() and (self.predict[i, j, :] == color_list[3]).all())):
                        self.error_map[i, j, :] = [0, 0, 0]
                    else:
                        self.error_map[i, j, :] = [255, 255, 255]
                        error += 1
            
            self.img_id += 1
            print('\rimg_id: '+str(self.img_id)+'\t accuracy:'+'%.2f' %(100 - (error/(self.error_map.shape[0]*self.error_map.shape[1]))*100)+'%', end='')
            self.accuracy_sum += 100 - (error/(self.error_map.shape[0]*self.error_map.shape[1]))*100;
            if self.img_id == IMG_NUM:
                print('\naverage accuracy: %.2f' %(self.accuracy_sum/self.img_id) +'%')
            
            if args.visualize:
                cv2.imshow('error_map', self.error_map)
                cv2.imshow('predict', self.predict)
                cv2.imshow('label', self.label*60)
                cv2.waitKey(0)

    def resize(self, src, dst):
        src = cv2.resize(src, (dst.shape[0], dst.shape[1]), interpolation=cv2.INTER_AREA)
        return src

if __name__ == "__main__":

    signal.signal(signal.SIGINT, keyBoardINT)  
    signal.signal(signal.SIGTERM, keyBoardINT)  
    accuracy = Accuracy(args.path, args.dst)
    accuracy.accuracyCalculate(IMG_NUM, label_list, color_list)