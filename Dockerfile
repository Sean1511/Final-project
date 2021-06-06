# set base image (host OS)
FROM python:3.8

# copy files to the working directory
COPY . .

# 
ENV FLASK_ENV=development
ENV FLASK_APP=app.py

# update pip
RUN python -m pip install --upgrade pip

# install dependencies
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install -r requirements.txt

# expose port
EXPOSE 5000

# command to run on container start
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
