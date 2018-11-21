'''
python3 check_img_label_to_digit.py --image=cube_shadow_label.png
python3 check_img_label_to_digit.py --image=../data/dataset1_ori/annotations_prepped_test/0016E5_07959.png
'''
import argparse
import cv2
def to_digit(img, f_head_name):
    fo = open("%s.txt" % f_head_name, "w")
    for i in range(img.shape[0]):
        fo.write('%03d:' % i )
        for j in range(img.shape[1]):
            if len(img.shape)==3:
                fo.write('[%3d,%3d,%3d] ' % ( img[i][j][0], img[i][j][1], img[i][j][2]) )
            else:
                fo.write('%2d ' % img[i][j] )

        fo.write('\n' )
    fo.write( str(img) )
    fo.close()
    print('Save annotations(label) to  {}.txt'.format(f_head_name))

parser = argparse.ArgumentParser()
parser.add_argument("--image", type = str  )
args = parser.parse_args()

img = cv2.imread(args.image)
to_digit(img, args.image)