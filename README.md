# Automatic Gender Classiffication from Handwritten Images
Using a handwritten sample to automatically classify the writer's gender is an essential task in a wide range of areas, e.g., psychology,historical documents classiffication, and forensic analysis.<br>
The challenge of gender prediction from offline handwriting is demonstrated by the relatively low (below 90%) performance of state-of-the-art systems.<br>
Despite a high interest within a broad spectrum of research communities, the published works in this area generally concentrate on English and Arabic languages. 
Most of the existing approaches focus on manual feature selection.<br>
In this work, we study an application of deep neural networks for gender classiffication, where we investigate cross-domain transfer learning with ImageNet pre-training.<br>
The study was performed on two datasets, the QUWI dataset, consisting of handwritten documents in English and Arabic, and a new dataset of documents in Hebrew script.<br>
We perform extensive experiments, analyze and compare the results obtained with different neural networks. We also compare the obtained results against human-level performance.<br>
In this website you will be able to upload handwritting images and get an automatically gender classification. 

# Project Research
In order to understand the steps and what we did you are welcome to look at the <a href="Documentation/project_book.pdf" >Project book.</a>

# Project Setup and Run
In order to run this project on local environment please follow this steps:

1. Clone this repository.
2. Install the requirements file - pip install -r requirements.txt
3. Run the app by : 
3.1. On Linux(bash) - export FLASK_APP=predict_app.py and flask run --host=0.0.0.0 
3.2. On Windows(cmd) - set FLASK_APP=predict_app.py and flask run --host=0.0.0.0 
4. Enjoy the application.
