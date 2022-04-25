import pandas as pd
import cv2
import numpy as np
from builtins import object



class Data(object):
  def __init__(self, batch_size, dataset_directory):

    train_list = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08']
    val_list = ['p09', 'p10', 'p11']
    test_list = ['p12', 'p13', 'p14']
    data_list = train_list
    data_list.extend(val_list)
    data_list.extend(test_list)

    data = pd.concat([pd.read_csv(dataset_directory+'Label/'+j+'.label', delim_whitespace=True) for j in data_list])
    data = data[['Left', 'Right', '2DGaze', '3DGaze']]

    gaze = data['2DGaze'].str.split(',', expand=True)
    data['x'], data['y'] = gaze[0].astype('float'), gaze[1].astype('float')
    data['x'] = -1+2*(data['x']-data['x'].min())/(data['x'].max()-data['x'].min())
    data['y'] = -1+2*(data['y']-data['y'].min())/(data['y'].max()-data['y'].min())

    labels = []
    for i, row in data.iterrows():
        if row['x']<0 and row['y']>0:
            labels.append(0)
        elif row['x']>0 and row['y']>0:
            labels.append(1)
        elif row['x']>0 and row['y']<0:
            labels.append(2)
        elif row['x']<0 and row['y']<0:
            labels.append(3)
            
    data['Label'] = labels

    self.image_directory = dataset_directory+'Image/'
    self.train_data = data.loc[data['Left'].str.contains('|'.join(train_list))]
    self.num_train_batches = len(self.train_data)/batch_size
    self.val_data = data.loc[data['Left'].str.contains('|'.join(val_list))]
    self.num_val_batches = len(self.val_data)/batch_size
    self.test_data = data.loc[data['Left'].str.contains('|'.join(test_list))]
    self.num_test_batches = len(self.test_data)/batch_size

  def load_train_data(self):
      for chunk in np.array_split(self.train_data, self.num_train_batches):
          left = np.array([cv2.imread(self.image_directory+k, 0) for k in chunk['Left'].values])/255.0
          right= np.array([cv2.imread(self.image_directory+k, 0) for k in chunk['Right'].values])/255.0
          inputs, labels = np.concatenate([left, right], axis=-1), np.array(chunk['Label'])
          yield inputs, labels

  def load_val_data(self):
      for chunk in np.array_split(self.val_data, self.num_val_batches):
          left = np.array([cv2.imread(self.image_directory+k, 0) for k in chunk['Left'].values])/255.0
          right= np.array([cv2.imread(self.image_directory+k, 0) for k in chunk['Right'].values])/255.0
          inputs, labels = np.concatenate([left, right], axis=-1), np.array(chunk['Label'])
          yield inputs, labels

  def load_test_data(self):
      for chunk in np.array_split(self.test_data, self.num_test_batches):
          left = np.array([cv2.imread(self.image_directory+k, 0) for k in chunk['Left'].values])/255.0
          right= np.array([cv2.imread(self.image_directory+k, 0) for k in chunk['Right'].values])/255.0
          inputs, labels = np.concatenate([left, right], axis=-1), np.array(chunk['Label'])
          yield inputs, labels
