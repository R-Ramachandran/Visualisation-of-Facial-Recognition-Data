from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import tensorflow
from keras.models import load_model
import plotly.graph_objects as go
import plotly.offline as plo
from plotly import subplots
from skimage import io

model = load_model('model25.h5')
def emotion_analysis(emotions):
    objects = ('Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    y_pos = np.arange(len(objects))
    fig1 = go.Bar(x=objects, y=emotions*100, marker={'color': 'crimson'}, showlegend=False, name="")

    fig2 = go.Pie(labels=objects, values=emotions*100, name="")

    fig3 = go.Funnel(y=objects,x=emotions*100,name="",marker={'color' : 'tan'}, showlegend=False)              #funnel
    
    fig4 = go.Scatter(x=objects, y=emotions*100, name="",marker={'color' : 'teal'}, showlegend=False, fill= 'tonexty', fillcolor='rgb(111, 231, 219)')
    
    fig5 = go.Scatter(x=objects, y=emotions*100, name="",mode= 'markers', marker={'color' : 'yellow', 'size' : emotions*150}, showlegend=False)

    img = io.imread("photo.jpeg")
    
    fig6 = go.Image(z=img)
    
    figure = subplots.make_subplots(
    rows=2,
    cols=3,
    specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "pie"}],
           [{"type": "scatter"}, {"type": "funnel"}, {"type": "image"}] ],
    subplot_titles= ("Visualisation through Bar Graph","Visualisation through Bubble Chart","Visualisation through Pie Chart",
                    "Visualisation through Area Chart","Visualisation through Funnel Chart","Image for Emotion Recognition")
    )

    figure.add_trace(fig1, 1, 1)
    figure.add_trace(fig5, 1, 2)
    figure.add_trace(fig2, 1, 3)
    figure.add_trace(fig4, 2, 1)
    figure.add_trace(fig3, 2, 2)
    figure.add_trace(fig6, 2, 3)
    figure.update_layout(
        {
            "autosize" : True,
            "title" :{"text" : "Data Visualisation for Emotion Recognition", "font" : {"size" : 30}},
            "xaxis_title" : {"text" : "Emotion", "font" : {"size" : 20}},
            "yaxis_title" : {"text" : "Percentage", "font" : {"size" : 20}},
            "template" : "plotly_dark"
        }
    )
    figure.show()

file = 'photo.jpeg'
true_image = image.load_img(file)
img = image.load_img(file, color_mode = "grayscale", target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])