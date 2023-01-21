import streamlit as st
import torch
import matplotlib.pyplot as plt 
from PIL import Image
import tempfile
import glob
from datetime import datetime
import os
import numpy as np
import librosa
import librosa.display as ld
import soundfile as sf


## CFG
cfg_model_path = 'models/SoundBeast_Model.pt' 

cfg_enable_url_download = False
if cfg_enable_url_download:
   url = "https://archive.org/download/yoloTrained/yoloTrained.pt" #Configure this if you set cfg_enable_url_download to True
   cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
## END OF CFG

def imageInput(device, src):
    _ = open(os.devnull, 'w')
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')
            num_bboxes = len(pred.xyxy[0])
            st.write(f'Number of bounding boxes: {num_bboxes}')

    elif src == 'From test set.': 
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:
                #call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                #--Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')
                num_bboxes = len(pred.xyxy[0])
                st.write(f'Number of bounding boxes: {num_bboxes}')

          





def audioInput():
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "wave", "flac", "mp3", "ogg"])
    if uploaded_audio is not None:
        # create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        temp_file.write(uploaded_audio.getvalue())
        temp_file.close()
        # load audio file
        y, sr = librosa.load(temp_filename)
        # convert the audio to wav format
        sf.write(temp_filename, y, sr, format='wav')
        #create a spectrogram from the audio
        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        # Convert amplitude spectrogram to Decibels-scaled spectrogram
        DB = librosa.amplitude_to_db(D, ref = np.max)
        # Create the spectogram
        fig, ax = plt.subplots(figsize = (16, 6))
        im = librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log',ax=ax)
        plt.colorbar(im)
        plt.title('Decibels-scaled spectrogram', fontsize=20)
        st.pyplot(fig)

        # Save the spectrogram as a PNG file
        spectrogram_filename = "spectrogram.png"
        plt.savefig(os.path.join('data', spectrogram_filename))

        # Run the spectrogram through the YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
        model.cpu()
        pred = model(os.path.join('data', spectrogram_filename))
        pred.render()  # render bbox in image
        for im in pred.ims:
            im_base64 = Image.fromarray(im)
            im_base64.save(os.path.join('data/outputs', os.path.basename(spectrogram_filename)))

        # Display the output image
        img_ = Image.open(os.path.join('data/outputs', os.path.basename(spectrogram_filename)))
        st.image(img_, caption='Model Prediction(s)')
        
        num_bboxes = len(pred.xyxy[0])
        st.write(f'Number of bounding boxes: {num_bboxes}')
        
        os.remove(temp_filename)
  

        if st.button('Download Spectrogram'):
            d = datetime.now()
            filename = "spectrogram" + d.strftime("%Y-%m-%d %H-%M-%S") + '.png'
            plt.savefig(os.path.join('data/Spectrogram', filename))
            st.success(f'Image saved to {filename} path is data/Spectrogram')









         
 
      ##  ts = datetime.timestamp(datetime.now())
     ##   imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name)
      ##  outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

       ## with open(imgpath, mode='wb') as f:
         ##   f.write(uploaded_video.read())  # save video to disk

       ## st_video = open(imgpath, 'rb')
       ## video_bytes = st_video.read()
       ## st.video(video_bytes)
       ## st.write("Uploaded Video")
      ##  detect(weights=cfg_model_path, source=imgpath, device=0) if device == 'cuda' else detect(weights=cfg_model_path, source=imgpath, device='cpu')
       ## st_video2 = open(outputpath, 'rb')
      ##  video_bytes2 = st_video2.read()
       ## st.audio(video_bytes2)
         # audio_byte.write("Model Prediction")


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    option = st.sidebar.radio("Select input type.", ['Image', 'Audio'])
    if option == 'Audio':
        datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'], disabled = True, index=1)
    else:
        datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])


    ## datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])
    
        
                
    
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar

    st.header('❤️BabySoundbeast Detection')
    st.subheader('👈🏽 Select options left-haned menu bar.')
    st.sidebar.markdown("Visit my github: https://github.com/sushisanu/BabySoundbeastsWithStreamlit")
    if option == "Image":    
        imageInput(deviceoption, datasrc)
    elif option == "Audio": 
        audioInput()

    

if __name__ == '__main__':
  
    main()

# Downlaod Model from url.    
## @st.cache
##def loadModel():
  ##  start_dl = time.time()
   ## model_file = wget.download(url, out="models/")
  ##  finished_dl = time.time()
  ##  print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
##if cfg_enable_url_download:
  ##  loadModel()


 
 