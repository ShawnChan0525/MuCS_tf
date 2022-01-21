import numpy as np
import six
import json
import math
from modeling import create_LM_att_mask,BertConfig
import tensorflow as tf
def getJson():
    files = []
    for i in range(20000):
        s = str(i)+('.txt')
        files.append(s)
    data = {
        'my_project':files
        }
    with open('project_dict.json','w')as f:
        json.dump(data,f)
    projects = json.loads(open('project_dict.json', 'r').read())
    print(projects)

def testBandPart():
    a = tf.ones([5,5])
    b = tf.linalg.band_part(a,1,0)
    c = tf.linalg.band_part(a,-1,0)
    d = tf.linalg.band_part(a,0,0)
    print(a)
    print(b)
    print(c)
    print(d)

def test_LM_Mask():
    print(create_LM_att_mask(10,10,0))

def testJson(file):
    a = BertConfig.from_json_file(file)
    print(a)

def ifGPU():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == '__main__':
    t1 = tf.ones([16,1])
    sess = tf.Session()
    print(sess.run(t1[0][0]))