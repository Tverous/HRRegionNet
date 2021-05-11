import torch
import torchvision
import numpy as np
from PIL import Image

def dataset_generator(data_dir, data_list, crop_size, crop_ratio, output_size, train=False):
    
    train_tx = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])

    data_set = HanDataset(data_list, data_dir, crop_ratio, crop_size, output_size, transform=train_tx, train=train)
    
    return data_set
    
class HanDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list, image_path, crop_ratio, crop_size, output_size, transform=None, train=False):
        
        self.data_list = data_list
        self.image_path = image_path
        self.transform = transform
        self.crop_ratio = crop_ratio
        self.crop_size = crop_size
        self.output_size = output_size
        self.output_size2 = output_size * 2
        self.train = train
    
    def __len__(self):
        
        return len(self.data_list)
    
    def __getitem__(self,idx):
        
        file_name = self.data_list[idx][0]
        img = Image.open(self.image_path + file_name).convert('RGB')
        
        origin_pic_width, origin_pic_height = img.size
        output_width, output_height = (self.output_size, self.output_size)
        output_width2, output_height2 = (self.output_size2, self.output_size2)

        #low_label = np.zeros((output_height, output_width, 6))      
        #high_label = np.zeros((output_height2, output_width2, 6))      
        
        if origin_pic_height < self.crop_size or origin_pic_width < self.crop_size:
            img = torchvision.transforms.functional.resize(img, (self.crop_size, self.crop_size))
        
        pic_width, pic_height = img.size
        _CROPPED = False
        
        if np.random.randint(0, 101) < self.crop_ratio * 100 and pic_height >= self.crop_size and pic_width >= self.crop_size:
            
            top = np.random.randint(0, pic_height - self.crop_size + 1)
            left = np.random.randint(0, pic_width - self.crop_size + 1)
            img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
            _CROPPED = True
            centerX = (2*left + self.crop_size)/2
            centerY = (2*top + self.crop_size)/2
            offsetX = (centerX-self.crop_size/2)*self.output_size/self.crop_size
            offsetY = (centerY-self.crop_size/2)*self.output_size/self.crop_size
            
            offsetX2 = (centerX-self.crop_size/2)*self.output_size2/self.crop_size
            offsetY2 = (centerY-self.crop_size/2)*self.output_size2/self.crop_size
            
        else:
            
            top = 0
            left = 0
            img = torchvision.transforms.functional.resize(img, (self.crop_size, self.crop_size))
            offsetX = 0
            offsetY = 0
            offsetX2 = 0
            offsetY2 = 0
            
        low_label = self.get_label(idx, origin_pic_width, origin_pic_height, pic_width, pic_height, output_width, output_height, self.crop_size, offsetX, offsetY, _CROPPED)
        high_label = self.get_label(idx, origin_pic_width, origin_pic_height, pic_width, pic_height, output_width2, output_height2, self.crop_size, offsetX2, offsetY2, _CROPPED)

        if self.transform:
            img = self.transform(img)
            
        if self.train:
            sample = {'image': img, 'low_labels': low_label, 'high_labels': high_label}
        elif not self.train:
            sample = {'image': img, 'low_labels': low_label, 'high_labels': high_label, 'img_size': [origin_pic_width, origin_pic_height]}
        
        return sample
   
    def get_label(self, idx, origin_pic_width, origin_pic_height, pic_width, pic_height, output_width, output_height, crop_size, offsetX, offsetY, _CROPPED):
        
        label = np.zeros((output_height, output_width, 6))      
        
        for annotation in self.data_list[idx][1]: 
            
            if _CROPPED:
                x_c = annotation[1] * (pic_width / origin_pic_width) * (output_width / self.crop_size) - offsetX
                y_c = annotation[2] * (pic_height / origin_pic_height) * (output_height / self.crop_size) - offsetY
                width = annotation[3] * (pic_width / origin_pic_width)  * (output_width / self.crop_size) 
                height = annotation[4] * (pic_height / origin_pic_height) * (output_height / self.crop_size) 
            else:
                x_c = annotation[1] * (output_width / origin_pic_width) 
                y_c = annotation[2] * (output_height / origin_pic_height) 
                width = annotation[3] * (output_width / origin_pic_width)  
                height = annotation[4] * (output_height / origin_pic_height) 
    
            if x_c >= output_width or y_c >= output_height or x_c <= 0 or y_c <= 0 :
                continue
    
            heatmap = ((np.exp(-(((np.arange(output_width) - x_c)/(width/10))**2)/2)).reshape(1,-1)
                    *(np.exp(-(((np.arange(output_height) - y_c)/(height/10))**2)/2)).reshape(-1,1))
            
            label[:, :, 0] = np.maximum(label[:,:,0], heatmap[:,:])
            label[int(y_c//1), int(x_c//1), 1] = 1
            label[int(y_c//1), int(x_c//1),2] = y_c % 1
            label[int(y_c//1), int(x_c//1),3] = x_c % 1
            label[int(y_c//1), int(x_c//1), 4] = width / output_width
            label[int(y_c//1), int(x_c//1), 5] = height / output_height
        

        return label
