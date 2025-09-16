import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd




import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

class IEMOCAPDataset(Dataset):
    def __init__(self, train=True):

        data_path = '/data/USER/SURE-main/data/iemocap_multimodal_features.pkl'
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(data_path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len


    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        

        max_length = 110
        
        result = []

        for i in range(4):
   
            padded = pad_sequence(dat[i])
            current_length = padded.size(0)
            
            if current_length > max_length:
       
                padded = padded[:max_length]
            elif current_length < max_length:
      
                padding = torch.zeros(max_length - current_length, *padded.shape[1:], 
                                    device=padded.device, dtype=padded.dtype)
                padded = torch.cat([padded, padding], dim=0)
            
            result.append(padded)
        

        for i in range(4, 6):
          
            padded = pad_sequence(dat[i], batch_first=True)
            batch_size = padded.size(0)
            current_length = padded.size(1)
            
            if current_length > max_length:
           
                padded = padded[:, :max_length]
            elif current_length < max_length:
             
                padding = torch.zeros(batch_size, max_length - current_length, 
                                    *padded.shape[2:], device=padded.device, 
                                    dtype=padded.dtype)
                padded = torch.cat([padded, padding], dim=1)
            
            result.append(padded)
        
   
        result.append(dat[6].tolist())
        
        return result



class MELDDataset(Dataset):
    def __init__(self, path, train=True):
  
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
   
        max_length = 110
        
        result = []

        for i in range(4):
      
            padded = pad_sequence(dat[i])
            
  
            if len(padded.shape) > 3 and padded.shape[2] == 1:
                padded = padded.squeeze(2)
                
            current_length = padded.size(0)
            
            if current_length > max_length:
         
                padded = padded[:max_length]
            elif current_length < max_length:
        
                padding = torch.zeros(max_length - current_length, *padded.shape[1:], 
                                     device=padded.device, dtype=padded.dtype)
                padded = torch.cat([padded, padding], dim=0)
            
            result.append(padded)
        
     
        for i in range(4, 6):
       
            padded = pad_sequence(dat[i], batch_first=True)
            

            if len(padded.shape) > 3 and padded.shape[2] == 1:
                padded = padded.squeeze(2)
                
            batch_size = padded.size(0)
            current_length = padded.size(1)
            
            if current_length > max_length:
       
                padded = padded[:, :max_length]
            elif current_length < max_length:
       
                padding = torch.zeros(batch_size, max_length - current_length, 
                                     *padded.shape[2:], device=padded.device, 
                                     dtype=padded.dtype)
                padded = torch.cat([padded, padding], dim=1)
            
            result.append(padded)
        
     
        result.append(dat[6].tolist())
        
        return result