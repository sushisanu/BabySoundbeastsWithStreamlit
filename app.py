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
import easygui
import tkinter as tk
from tkinter import filedialog




## CFG
##cfg_model_path = 'models/SoundBeast_Model.pt' 
## CFG V2
cfg_model_path = 'models/SoundBeast_ModelV2.pt' 


## END OF CFG

def save_file_dialog(default_file_name, file_types):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultfilename=default_file_name, filetypes=file_types)
    return file_path


def imageInput(device, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            file_name = image_file.name
            file_ext = os.path.splitext(file_name)[-1].lower()
            ts = datetime.timestamp(datetime.now())
            img_name = f"{str(ts)}_{file_name}"
            imgpath = os.path.join('data/uploads', img_name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # Call Model prediction
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
            model.cpu()
            with torch.no_grad():
                pred = model(imgpath)
                pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # Get the number at the back of the file name before the underscore
            num_str = file_name.split("_")[-1]
            num_second = float(num_str.split(".")[0] + "." + num_str.split(".")[1])
            num_second = round(num_second, 2)
            num_minutes = num_second/60

            num_bboxes = len(pred.xyxy[0])

            bboxes_per_minute = round(num_bboxes / num_minutes ,2)

            # Display prediction
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

            st.write(f"Baby's beasts =  {num_bboxes} times.")
            st.write(f"Time of this audio is : {num_second} second.")
            st.write(f"Baby's beasts : {bboxes_per_minute} per minute. ")
            if bboxes_per_minute  < 5:
                st.write("Alert! Abnormal case' Baby's soundbeasts less than 5 times per minute.")
                #  result = "Baby's beasts {num_bboxes} per all time. \n Baby's soundbeasts less than 5 peaks per minute."


            else:
                st.write("Baby's soundbeasts is stable!")
                #  result = "Baby's beasts {num_bboxes} per minute. \n Baby's soundbeasts is stable!"





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
                st.write(f"Baby's beasts = {num_bboxes} times")
                st.write(f"Baby's beasts : ... per minute. ")
                if num_bboxes  < 5:
                    st.write("Alert! Abnormal case' Baby's soundbeasts less than 5 times per minute.")
                    #  result = "Baby's beasts {num_bboxes} per all time. \n Baby's soundbeasts less than 5 peaks per minute."


                else:
                    st.write("Baby's soundbeasts is stable!")
                    #  result = "Baby's beasts {num_bboxes} per minute. \n Baby's soundbeasts is stable!"
                

          





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
        
        # Check Audio type is correct?
        if temp_filename.endswith('wav'):
            st.audio(temp_filename, format='audio/wav')
        elif temp_filename.endswith('mp3'):
            st.audio(temp_filename, format='audio/mp3')
        elif temp_filename.endswith('flac'):
            st.audio(temp_filename, format='audio/flac')
        elif temp_filename.endswith('ogg'):
            st.audio(temp_filename, format='audio/ogg')
        else:
            st.error('The audio format is not supported')
        
        
        # convert the audio to wav format
        #sf.write(temp_filename, y, sr, format='wav')
        # create a spectrogram from the audio
        #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

        # create a new figure
        #fig, ax = plt.subplots()
        # display the spectrogram as an image
        #ld.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, fmax=8000, ax=ax)
        #st.pyplot(fig)
        
        # Define the desired size for the output image
        output_size = (750, 550)
        save_size = (1400, 1100)

        # create a spectrogram from the audio
        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        # Convert amplitude spectrogram to Decibels-scaled spectrogram
        DB = librosa.amplitude_to_db(D, ref = np.max)
        # Create the spectogram
        fig, ax = plt.subplots(figsize = (16, 6))
        im = librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log',ax=ax)
        plt.colorbar(im)
        plt.title('Decibels-scaled spectrogram', fontsize=20)
        # Resize the figure to the desired output size
        fig.set_size_inches(output_size[0]/100, output_size[1]/100)
        fig.tight_layout()
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
            # Resize the image to the desired output size
            im_base64 = im_base64.resize(output_size)
            im_base64.save(os.path.join('data/outputs', os.path.basename(spectrogram_filename)))

        # Display the output image at the desired output size
        img_ = Image.open(os.path.join('data/outputs', os.path.basename(spectrogram_filename)))
        img_resized = img_.resize(output_size)
        st.image(img_resized, caption='Model Prediction(s)', width=output_size[0])
        img_resized = img_.resize(save_size)

        
        
        # Calculate how many bounding boxies were detected in 1 minute.
        num_bboxes = len(pred.xyxy[0])
        
        

        #duration second time
        duration_second = librosa.get_duration(y, sr)
        
        #duration divine minute
        duration_minute = round(librosa.get_duration(y, sr) / 60, 2)

        

        bboxes_per_minute = num_bboxes / (duration_minute)
        # round to 2 decimal numbers.
        bboxes_per_minute = round(bboxes_per_minute, 2)
        
        
        
       
        st.write(f"Baby's beasts = {num_bboxes} times.")
        st.write(f"Baby's beasts : {bboxes_per_minute} per minute. ")
        if bboxes_per_minute  < 5:
            st.write("Alert! Abnormal case' Baby's soundbeasts less than 5 times per minute.")
          #  result = "Baby's beasts {num_bboxes} per all time. \n Baby's soundbeasts less than 5 peaks per minute."
        

        else:
            st.write("Baby's soundbeasts is stable!")
          #  result = "Baby's beasts {num_bboxes} per minute. \n Baby's soundbeasts is stable!"

        os.remove(temp_filename)
        
        # Add text input for username
        username = st.text_input("Enter your name before download files:")
        if not username:
            st.warning("Please enter your name to download provide data")
        else:
            # Download Spectrogram image file
            if st.button('Download Spectrogram image file'):
                d = datetime.now()
                filename = str(username) + d.strftime("-Date %d-%m-%Y Time-%H-%M-%S Duration_") + str(duration_second) + '.png'
                folder_path = easygui.diropenbox(title="Select directory to save file")
                if folder_path is not None:
                    plt.savefig(os.path.join(folder_path, filename))
                    st.success(f'"{filename}" is file name save to {folder_path}')

            # Download results with text files format
            if st.button('Download Result in text file'):
                d = datetime.now()
                filename = str(username) + d.strftime("_%d-%m-%Y_Time-%H-%M-%S") + ".txt"
                with open(filename, "w") as f:
                    f.write(f"Baby's beasts = {num_bboxes} times.\n")
                    f.write(f"Baby's beasts : {bboxes_per_minute:.2f} per minute.\n")
                    if bboxes_per_minute < 5:
                        f.write("Alert! Abnormal case' Baby's soundbeasts less than 5 times per minute.")
                    else:
                        f.write("Baby's soundbeasts is stable!\n")
                with open(filename, "r") as f:
                    file_contents = f.read()
                st.download_button(label="Download", data=file_contents, file_name=filename, mime="text/plain")











         
 



def main():
    # -- Sidebar
    st.markdown(
        f"""
        <style>
            .sidebar .sidebar-content {{
                background-color: #1E90FF;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title('BabySoundbeast Application')
    option = st.sidebar.radio("Select input type.", ['Image', 'Audio'])
    if option == 'Audio':
        datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'], disabled = True, index=1)
    else:
        datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])
    
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar

    st.header(chr(0x1F6BC) + 'BabySoundbeast Detection')
    st.subheader('Select options left-haned menu bar.')
    st.sidebar.markdown("-----")
    st.sidebar.markdown("Develope By Mr.sushisanu kulprom")
    st.sidebar.markdown("supervisor : Prof. Wanida Kanarkard ")
    st.sidebar.markdown("co-supervisors :")
    st.sidebar.markdown("1.Mr. Nawapak Eua-anant")
    st.sidebar.markdown("2.Mr. Wasu Chaopanon")
    st.sidebar.markdown("Project leader: Dr.thiwawan thepha")
    st.sidebar.markdown("-----")
    
    st.sidebar.markdown("This project is 4th year project of the Faculty of Computer Engineering. Khon Kaen University, Academic Year 2022")

    if option == "Image":    
        imageInput(deviceoption, datasrc)
    elif option == "Audio": 
        audioInput()




    

if __name__ == '__main__':
  
    main()




 
 