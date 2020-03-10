FROM python
COPY . /app
WORKDIR /app
RUN python -m pip install pip
RUN pip install flask
RUN pip install opencv-python
RUN pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install Pillow==6.0.0
EXPOSE 5000
CMD ["python", "app.py"]
