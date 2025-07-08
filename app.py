
import pickle
import streamlit as st
import numpy as np
import pyparsing
import pandas as pd
import matplotlib as plt
import seaborn as sns
from PIL import Image 
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 
from streamlit_option_menu import option_menu
st.set_page_config(page_title='Lung Cancer Detection')
#Loading models
cancer_model = pickle.load(open('models/final_model.sav', 'rb'))


with st.sidebar:
    selection = option_menu('Lung Cancer Detection System',
    ['Introduction',
    'About the Dataset',
    'Lung Cancer Detection',
    'CNN Based disease Detection'],
    icons = ['activity','heart','person', 'heart'],
    default_index = 0)
    

#Introduction page
if (selection == 'Introduction'):

    from PIL import Image

    gg = Image.open("images/lung-cancer.jpg")

    st.image(gg, caption='Introduction to Lung Cancer',width=600)
    #page title
    st.title('How common is lung cancer?')

    st.write("Lung cancer (both small cell and non-small cell) is the second most common cancer in both men and women in the United States (not counting skin cancer). In men, prostate cancer is more common, while in women breast cancer is more common.")
    st.markdown(
    """
    The American Cancer Society’s estimates for lung cancer in the US for 2023 are:
    - About 238,340 new cases of lung cancer (117,550 in men and 120,790 in women)
    - About 127,070 deaths from lung cancer (67,160 in men and 59,910 in women)

    """
    )

    st.write("")
    st.title("Is Smoking the only cause ?")
    mawen = Image.open("images/menwa.png")

    st.image(mawen, caption='Smoking is not the major cause',width=650)
    #page title
    
    st.write("The association between air pollution and lung cancer has been well established for decades. The International Agency for Research on Cancer (IARC), the specialised cancer agency of the World Health Organization, classified outdoor air pollution as carcinogenic to humans in 2013, citing an increased risk of lung cancer from greater exposure to particulate matter and air pollution.")




    st.markdown(
    """
    The following list won't indent no matter what I try:
    - A 2012 study by Mumbai’s Tata Memorial Hospital found that 52.1 per cent of lung cancer patients had no history of smoking. 
    - The study contrasted this with a Singapore study that put the number of non-smoking lung cancer patients at 32.5 per cent, and another in the US that found the number to be about 10 per cent.
    - The Tata Memorial study found that 88 per cent of female lung cancer patients were non-smokers, compared with 41.8 per cent of males. It concluded that in the case of non-smokers, environmental and genetic factors were implicated.
    """
    )

    st.title("Not just a Delhi phenomenon ")
    stove = Image.open("images/stove.png")

    st.image(stove, caption='Smoking is not the major cause',width=650)
    #page title
    st.markdown(
    """
    The following list won't indent no matter what I try:
    - In January 2017, researchers at AIIMS, Bhubaneswar, published a demographic profile of lung cancer in eastern India, which found that 48 per cent of patients had not been exposed to active or passive smoking
    - 89 per cent of women patients had never smoked, while the figure for men was 28 per cent.
    - From available research, very little is understood about lung cancer among non-smokers in India. “We need more robust data to identify how strong is the risk and link,” Guleria of AIIMS says.
    """
    )

if (selection == 'About the Dataset'):
    tab1, tab2, tab3 , tab4 ,tab5= st.tabs(["Dataset analysis", "Training Data", "Test Data","Algorithms Used",'CNN Based Indentification'])

    with tab1:
        
        st.header("Lung Cancer Dataset")
        data=pd.read_csv("datasets/data.csv")
        st.write(data.head(10))
        code = '''
        Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
       'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
       'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
       'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
       'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
       'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'],
      dtype='object')'''
        st.code(code, language='python')

        
        st.header("Pearson Correlation Matrix")
        coors = Image.open("images/coors.png")

        st.image(coors, caption='Pearson Correlation Matrix',width=800)
        st.write("From the above co-relation matrix we did apply a function which picks out values based on their high correlation with a particular attribute which could be dropped to improve Machine Learning Models Performance")
        st.markdown( """
            
            - The Following Attributed are as follows :-
            """)
        code = '''{'Chest Pain',
 'Coughing of Blood',
 'Dust Allergy',
 'Genetic Risk',
 'OccuPational Hazards',
 'chronic Lung Disease'}'''
        st.code(code, language='python')

    with tab2:
        st.header("Lung Cancer Training Dataset")

        st.subheader("X_Train Data")
        data=pd.read_csv("datasets/train.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data=pd.read_csv("datasets/trainy.csv", index_col=0)
        st.subheader("Y_Train Data")
        st.dataframe(data, use_container_width=True)


       

    with tab3:
        st.header("Lung Cancer Training Dataset")

        st.subheader("X_Test Data")
        data=pd.read_csv("datasets/testx.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data=pd.read_csv("datasets/testy.csv", index_col=0)
        st.subheader("Y_Test Data")
        st.dataframe(data, use_container_width=True)
        
    with tab4:
        st.header("List of Algorithms Used")
        algo = Image.open("images/algo.png")

        st.image(algo, caption='ML Algorithms',width=500)

        st.write("Since this is a Mutlti-Class Classification we have used Algorithms which are maily used for Supervised Learning for the following Problem Statement ")

        st.markdown(
            """
            Supervised Learning Algorithms:
            - Linear Regression
            - Support Vector Machine
            - K-Nearest Neighbours (KNN)
            - Decision Tree Classifier
            """
            )
        
        st.write("The accuracy of all the above algorithms is as follows:- ")
        code = '''The accuracy of the SVM is: 95 %
        The accuracy of the SVM is: 100 %
        The accuracy of Decision Tree is: 100 %
        The accuracy of KNN is: 100 %'''
        st.code(code, language='python')

        st.header("Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:
            algo = Image.open("images/lg.png")

            st.image(algo, caption='LG Confusion Matrix',width=350)

        with col2:
            algo = Image.open("images/svm.png")

            st.image(algo, caption='SVM Confusion Matrix',width=390)


    with tab5:
        st.header("Convolutional Neural Network Model")
        st.write("Apart from detecting cancer using various parameters in the dataset we can also make out predictions using CT Scan Images by using Convolutional Neural Networks. Link to the image dataset is given below :- ")
        url = "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images"
        st.write("Check out this [Images Dataset](%s)" % url)

        st.subheader("Approach Followed :- ")
        st.markdown(
            """
            - For training our model we have used the Keras API.
            - We have used 2D Convolution Layer along with consecutive MaxPooling Layers to improve the models performance.
            - Because we are facing a two-class classification problem, i.e. a binary classification problem, we will end the network with a sigmoid activation. The output of the network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).
            """
            )
        st.subheader("Model Summary")
        summ = Image.open("images/summary.png")

        st.image(summ, caption='Model Summary',width=700)
        st.subheader("Model Compile ")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''from tensorflow.keras.optimizers import RMSprop

        model.compile(optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics = ['accuracy'])'''
        st.code(code, language='python')

        st.subheader("Fitting Data to the Model")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=2
            )'''
        st.code(code, language='python')

        epoc = Image.open("images/epoc.png")

        st.image(epoc, caption='Number of Epocs',width=700)

        st.subheader("Plotting the Traning vs Validation (Accuracy and Loss)")
        col1, col2 = st.columns(2)

        with col1:
            acc = Image.open("images/acc.png")

            st.image(acc, caption='Traning vs Validation Accuracy',width=350)

        with col2:
            loss = Image.open("images/loss.png")

            st.image(loss, caption='Traning vs Validation Loss',width=350)

        st.write("As we can see from the above diagram that our Models performs well on the Training as well as Validation Data")


        
    

# Lung Cancer disease Prediction pages
if selection == 'Lung Cancer Detection':
    import pandas as pd
    import streamlit as st

    # Load data
    testx = pd.read_csv("datasets/testx.csv", index_col=0)
    testy = pd.read_csv("datasets/testy.csv", index_col=0)
    testx.reset_index(drop=True, inplace=True)
    testy.reset_index(drop=True, inplace=True)
    concate_data = pd.concat([testx, testy], axis=1)

    st.title('Lung Cancer Prediction using ML')

    idn = st.slider('Select any index from Testing Data', 0, len(concate_data)-1, 25)
    selected_row = concate_data.iloc[idn]

    # Show the dataset shape for debugging
    st.write("Dataset shape: ", concate_data.shape)

    st.write("Displaying values of index ", idn)
    if st.button('Show me this value'):
        st.write(list(selected_row))

    # Ensure you're unpacking the right number of values (17 expected)
    concate_data = concate_data.iloc[:, :17]  # Keep only the first 17 columns

# Proceed with the rest of the code
    selected_row = concate_data.iloc[idn]

# Unpack the row if it has exactly 17 values
    if len(selected_row) == 17:
       a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q = selected_row.values
    else:
     st.error("The selected row does not have 17 values!")
     st.stop() # Stop execution if the row doesn't have the expected number of values

    # UI columns
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input('Age', min_value=1, max_value=120, value=int(a), key="1")

    with col2:
        Gender = st.selectbox('Gender', options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female', index=int(b)-1, key="2")

    with col3:
        AirPollution = st.slider('Air Pollution', 1, 10, int(c), key="3")

    with col1:
        Alcoholuse = st.slider('Alcohol Use', 1, 10, int(d), key="4")

    with col2:
        BalancedDiet = st.slider('Balanced Diet', 1, 10, int(e), key="5")

    with col3:
        Obesity = st.slider('Obesity', 1, 10, int(f), key="6")

    with col1:
        Smoking = st.slider('Smoking', 1, 10, int(g), key="7")

    with col2:
        PassiveSmoker = st.slider('Passive Smoker', 1, 10, int(h), key="8")

    with col3:
        Fatigue = st.slider('Fatigue', 1, 10, int(i), key="9")

    with col1:
        WeightLoss = st.slider('Weight Loss', 1, 10, int(j), key="10")

    with col2:
        ShortnessofBreath = st.slider('Shortness of Breath', 1, 10, int(k), key="11")

    with col3:
        Wheezing = st.slider('Wheezing', 1, 10, int(l), key="12")

    with col1:
        SwallowingDifficulty = st.slider('Swallowing Difficulty', 1, 10, int(m), key="13")

    with col2:
        ClubbingofFingerNails = st.slider('Clubbing of Finger Nails', 1, 10, int(n), key="14")

    with col3:
        FrequentCold = st.slider('Frequent Cold', 1, 10, int(o), key="15")

    with col1:
        DryCough = st.slider('Dry Cough', 1, 10, int(p), key="16")

    with col2:
        Snoring = st.slider('Snoring', 1, 10, int(q), key="17")

    lung_diagnosis = ''

    if st.button('Lung Cancer Test Result'):
        prediction = cancer_model.predict([[Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss, ShortnessofBreath, Wheezing, SwallowingDifficulty, ClubbingofFingerNails, FrequentCold, DryCough, Snoring]])
        
        if prediction[0] == 'High':
            lung_diagnosis = "The person has lung cancer"
            st.error(lung_diagnosis)
        elif prediction[0] == 'Medium':
            lung_diagnosis = "The person has a chance of having lung cancer"
            st.warning(lung_diagnosis)
        else:
            lung_diagnosis = "The person does not have lung cancer"
            st.success(lung_diagnosis)
            st.balloons()


        

    # expander = st.expander("Here are some more random values from Test Set")
    
    # expander.write(concate_data.head(5))
    

    
        
   

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
from tempfile import NamedTemporaryFile

if (selection == 'CNN Based disease Detection'):
    # Set option to hide deprecated warning for file uploader encoding
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    # Using st.cache_resource to cache the model loading function
    @st.cache_resource
    def loading_model():
        fp = "models/keras_model.h5"
        model_loader = load_model(fp)
        return model_loader

    # Load the CNN model
    cnn = loading_model()
    
    st.write("""
    # Lung Cancer Detection using CNN and CT-Scan Images
    """)

    # File uploader for CT-Scan images
    temp = st.file_uploader("Upload CT-Scan Image", type=['png', 'jpeg', 'jpg'])
    if temp is not None:
        file_details = {"FileName": temp.name, "FileType": temp.type, "FileSize": temp.size}
        st.write(file_details)

        # Process the image
        buffer = temp
        temp_file = NamedTemporaryFile(delete=False)
        if buffer:
            temp_file.write(buffer.getvalue())
            st.write(image.load_img(temp_file.name))

        if buffer is None:
            st.text("Oops! that doesn't look like an image. Try again.")
        else:
            ved_img = image.load_img(temp_file.name, target_size=(224, 224))

            # Preprocessing the image
            pp_ved_img = image.img_to_array(ved_img)
            pp_ved_img = pp_ved_img / 255
            pp_ved_img = np.expand_dims(pp_ved_img, axis=0)

            # Predict the result using the CNN model
            hardik_preds = cnn.predict(pp_ved_img)
            st.write(hardik_preds[0])

            # Display the uploaded image
            image = Image.open(temp)
            st.image(image, use_column_width=True)

            # Display prediction result
            if hardik_preds[0][0] >= 0.5:
                out = ('I am {:.2%} percent confirmed that this is a Normal Case'.format(hardik_preds[0][0]))
                st.balloons()
                st.success(out)
            else: 
                out = ('I am {:.2%} percent confirmed that this is a Lung Cancer Case'.format(1 - hardik_preds[0][0]))
                st.error(out)

# Hide Streamlit's main menu, footer, and header
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)


    
