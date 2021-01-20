import logging 
import soundfile as sf 
import pandas as pd 
import numpy as np
from tqdm import tqdm 
import librosa 
import sys 
import librosa.display 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from os import listdir 
from os.path import isfile, join 
from numpy import inf 
from PIL import Image 
import pickle 
from itertools import chain 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython import embed

logging.getLogger().setLevel(logging.INFO)

class PreProcessor():
    
    def __init__(self, dirname, debug): 
        self.dirname = f"../{dirname}/" 
        self.train_fp = self.read_csv(f'../{dirname}/train_fp.csv')
        self.train_tp = self.read_csv(f'../{dirname}/train_tp.csv')
        self.img_path = f'data_img/'
        self.debug = debug 
    
    def debug(self): 
        return self.debug
    
    def read_csv(self, path):
        if self.debug:
            logging.info(f"|-> Reading path {path}")
        data = pd.read_csv(path) 
        if self.debug:
            logging.info(f"|-> Successfully read data of shape {data.shape} \n")
        return data
    
    def export_csv(self, data, name):
        if self.debug:
            logging.info(f"|-> Exporting Data to csv, filename : {name}")
        data.to_csv(name, index=False)

        if self.debug:
            logging.info(f"|-> Done writing data to csv, filename : {name}")
    
    def export_pickle(self, data, name):
        logging.info(f'|-> Opening and saving data to {name} \n')
        with open(name, 'wb') as f:
            pickle.dump(data, f)

    def read_pickle(self, name):
        logging.info(f'|-> Reading data from {name} \n')
        with open(name, 'rb') as f:
            pickle.load(f)

    def get_audio_file(self, name, d='train'):
        if self.debug:
           logging.info(f'|-> Searching for file {name}')
        path = self.dirname+d+"/"
        files = listdir(self.dirname+d+"/")
        for f in files:
            if name in f:
                if self.debug:
                    logging.info(f'|-> File {name} found \n')
                return path+f
        if self.debug:
            logging.info(f'|-> Couldnt find file {name}')

    def split_into_chunks(self, name):
        path  = self.get_audio_file(name)
        audio_data, samplerate = self.open_audio_file(path)
        chunk_times = [(0,5), 
                (5,10), 
                (10,15), 
                (15,20), 
                (20,25),
                (25,30), 
                (30,35), 
                (35,40), 
                (40,45), 
                (45,50),
                (50,55),
                (55,60),
                (2, 7),
                (7, 12), 
                (12,17), 
                (17,22), 
                (22,27), 
                (27,32),
                (32,37),
                (37,42),
                (42,47),
                (47,52),
                (52,57)]
        chunks = []
        for chunk in chunk_times:
            start = chunk[0]
            end   = chunk[1]
            chunks.append(self.get_slice(audio_data, start, end))
        return chunks 

    def open_audio_file(self, path):
        if self.debug:
            logging.info(f'|-> Openning file {path}')
        audio_data, samplerate = sf.read(path)
        if self.debug:
            logging.info(f'|-> File {path} opened Successfully')
            logging.info(f'|-> Read file with shape : {audio_data.shape} and samplerate : {samplerate}\n')
        return audio_data, samplerate

    def write_audio(self, audio_data, samplerate, filename):
        if self.debug:
            logging.info(f'|-> Writing audio data  to {filename}')
        sf.write(filename+'.flac', audio_data, samplerate)        
        if self.debug:
            logging.info(f'|-> Wrote audio data sucessfully\n')

    def get_slice(self, audio_data, t_min, t_max):
        if self.debug:
            logging.info(f'|-> Slicing audio data from second {t_min} to second {t_max}')
        t_min = self.time_to_index(t_min)
        t_max = self.time_to_index(t_max)
        if self.debug:
            logging.info(f'|-> Slicing audio data from index {t_min} to index {t_max} \n')
        return audio_data[t_min:t_max]
    
    def time_to_index(self, time, upper_bound=2880000):
        index = int((time * upper_bound) / 60)
        assert index <= upper_bound, f'Index calculated : {index} is not <= {upper_bound}'
        return index 
    
    def plot_waveform(self, audio_data, samplerate): 
        if self.debug:
            logging.info('|-> Plotting audio_data \n')
        plt.figure(figsize=(15,5))
        if self.debug:
            librosa.display.waveplot(audio_data, sr=samplerate)
        plt.show()
    

    def spec_img_show(self, spec): 
        if self.debug:
            logging.info(f'|-> Showing melspec \n')
        img = Image.fromarray(np.uint8(cm.gist_earth(spec)*255))
        img.show()
        img.save('GG.png')

    def spec_to_img(self, spec, eps =1e-6):
        if self.debug:
            logging.info(f'|-> Converting melspec to image ')
        mean = spec.mean()
        std  = spec.std()
        spec_norm = (spec - mean) / (std +eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min)  / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        if self.debug:
            logging.info(f'|-> Converting melspec to image done \n ')

        return spec_scaled

    def get_mel_spectrogram(self, audio_data, samplerate, name):
        if self.debug:
            logging.info('|-> Extracting Spectrogram ')
        window_size = 512  
        window = np.hanning(window_size)
        stft  = librosa.core.spectrum.stft(audio_data, n_fft=window_size, hop_length=1099, window=window)
        spectrogram = np.abs(stft)
        # out = 2 * np.abs(stft) / np.sum(window)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        p = librosa.display.specshow(log_spectrogram, ax=ax, y_axis='log', x_axis='time')
        
        ##### If i want to save and check it ! 
        path = f'data_img/{name}.png'
        fig.savefig(path)
        return path


        # logging.info('|-> Representing audio data as spectrogram ')
        # S = librosa.feature.melspectrogram(y=audio_data, sr=samplerate)
        # spec = librosa.amplitude_to_db(S, ref = np.max)
        return log_spectrogram  
    
    def plot_mel_spectrogram(self, audio_data, samplerate):
        librosa.display.specshow(self.get_mel_spectrogram(audio_data, samplerate))
        if self.debug:
            logging.info('|-> Done Converting and now showing mel spectrogram \n')
        plt.show()
    
    def plot_fft(self, audio_data):
        fft_data = self.fft(audio_data)
        fft_freq = np.fft.fftfreq(len(audio_data))
        if self.debug:
            logging.info(f'|-> Plotting FFT \n')
        plt.plot(fft_freq, fft_data)
        plt.show()

    def fft(self, audio_data):
        if self.debug:
            logging.info(f'|-> Computing FFT on audio data\n')
        return np.fft.fft(audio_data)
    
    def get_mel_band_energy(self, audio_data, samplerate, nfft=2048, nb_mel=40):
        if self.debug:
            logging.info('|-> Extract Mel-Band-Energy (MBE) ')
        spec, n_fft = librosa.core.spectrum._spectrogram(y=audio_data, n_fft=nfft, hop_length=int(nfft/2), power=1)
        mel_basis   = librosa.filters.mel(sr=samplerate, n_fft=nfft, n_mels=nb_mel)
        data = np.dot(mel_basis, spec)

        if self.debug:
            logging.info('|-> Done extracting MBE \n')
        return np.log(data)

    def get_mfcc(self, audio_data, samplerate):
        if self.debug:
            logging.info('|-> Extract MFCC')
        mfcc = librosa.feature.mfcc(audio_data, sr = samplerate, n_mfcc=5)
        if self.debug:
            logging.info('|-> Done extracting MFCC \n')
        return mfcc
    

    def get_img(self, name ):
        
        images = listdir(self.img_path)
        for image in images: 
            if name in image:
                return self.img_path+image

    def crop_imgs(self, name):
        image = self.get_img(name)
        img   = Image.open(image)
        width, hight = img.size

        dim = (79.5, 79.5, width-70, hight-70)
        cropped_img = img.crop(dim)
        
        plt.imshow(cropped_img)
        plt.axis('off')
        plt.show()

    def crop_all_and_save(self):
        images = listdir(self.img_path)
        for image in images:
            img = Image.open(self.img_path+image)
            width, height = img.size
            dim = (79.5, 79.5, width-70, height-70)
            cropped_img = img.crop(dim)
            cropped_img.save(f'data_img_crop/{image}') 
    
    def normalize(self, data):
        img_data = data['img_data']
        print(img_data.mean())

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def create_train_dataset(p, d='train'):
    print('----------------------------------------------')

    data   = {
                        'recording_id'    : [],
                        'img_data'        : [],
                        's_0'             : [],
                        's_1'             : [],
                        's_2'             : [],
                        's_3'             : [],
                        's_4'             : [],
                        's_5'             : [],
                        's_6'             : [],
                        's_7'             : [],
                        's_8'             : [],
                        's_9'             : [],
                        's_10'            : [],
                        's_11'            : [],
                        's_12'            : [],
                        's_13'            : [],
                        's_14'            : [],
                        's_15'            : [],
                        's_16'            : [],
                        's_17'            : [],
                        's_18'            : [],
                        's_19'            : [],
                        's_20'            : [],
                        's_21'            : [],
                        's_22'            : [],
                        's_23'            : []
            }

    logging.info(f'|-> Openning train_tp ')  

    unique_species = p.train_tp['species_id'].unique()
    unique_species.sort()

    for index, row in tqdm(chain(p.train_tp.iterrows(), p.train_fp.iterrows())):
        data['recording_id'].append(row['recording_id'])

        for i in range(24):
            if i == row['species_id']:
                data[f's_{i}'].append(1)
            else:
                data[f's_{i}'] .append(0)
            
        img_path                  = f"data_img_crop/{row['recording_id']}.png"
        
        img = Image.open(img_path)
        img.load() # required for png.split()

        background = Image.new("RGB", img.size, (255,255,255))
        background.paste(img, mask = img.split()[3])

        # img = img.convert('1')

        #plt.imshow(img)
        #plt.axis('off')
        #plt.show()
         
        img_data = np.array(background)
        # Normalize data 
        
        data['img_data'].append(img_data)
          
        
    data = pd.DataFrame.from_dict(data)
    
    return data  
    
def main(dirname):

    p = PreProcessor(dirname, False)
    #path = p.get_audio_file('')
    #audio_data, samplerate = p.open_audio_file(path)
        
    #spec = p.get_mel_spectrogram(audio_data, samplerate)

    #### Slicing Audio  
    #audio_sliced = p.get_slice(audio_data, int(24.498), int(28.6187))
    
    #### Extracting Features 
    #mbe = p.get_mel_band_energy(audio_data, samplerate)
    
    # mfcc = p.get_mfcc(audio_data, samplerate)
    

    # This methods reads train_tp and train_fp and stores data as training data with labeled s_0...s_23. 
    data = create_train_dataset(p)
    data = data.groupby(['recording_id']).agg({ 
                                    'img_data': ['first'],
                                    's_0':  ['max'],
                                    's_1':  ['max'],
                                    's_2':  ['max'],
                                    's_3':  ['max'],
                                    's_4':  ['max'],
                                    's_5':  ['max'],
                                    's_6':  ['max'],
                                    's_7':  ['max'],
                                    's_8':  ['max'],
                                    's_9':  ['max'],
                                    's_10': ['max'],
                                    's_11': ['max'],
                                    's_12': ['max'],
                                    's_13': ['max'],
                                    's_14': ['max'],
                                    's_15': ['max'],
                                    's_16': ['max'],
                                    's_17': ['max'],
                                    's_18': ['max'],
                                    's_19': ['max'],
                                    's_20': ['max'],
                                    's_21': ['max'],
                                    's_22': ['max'],
                                    's_23': ['max'],
                                }).reset_index()

    print('Shape of data after aggregating -> ', data.shape)
    print(data.shape)
    # export training_data_mbe  
    p.export_csv(data, 'train_img_spec.csv')
    p.export_pickle(data, 'train_img_spec.pkl')
    
if __name__  == '__main__':

    dirname = 'rfcx-species-audio-detection'
     
    main(dirname)
