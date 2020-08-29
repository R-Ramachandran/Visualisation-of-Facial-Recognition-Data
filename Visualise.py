from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import tensorflow
from keras.models import load_model
import plotly.graph_objects as go
import plotly.offline as plo
from plotly import subplots
from skimage import io
import tkinter
import random

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

happy_message = {
    1 : "Happy Message 1",
    2 : "Happy Message 2",
    3 : "Happy Message 3",
    4 : "Happy Message 4",
    5 : "Happy Message 5",
    6 : "Happy Message 6",
    7 : "Happy Message 7",
    8 : "Happy Message 8",
    9 : "Happy Message 9",
    10 : "Happy Message 10",
}

angry_message = {
    1 : "Angry Message 1",
    2 : "Angry Message 2",
    3 : "Angry Message 3",
    4 : "Angry Message 4",
    5 : "Angry Message 5",
    6 : "Angry Message 6",
    7 : "Angry Message 7",
    8 : "Angry Message 8",
    9 : "Angry Message 9",
    10 : "Angry Message 10",
}

fear_message = {
    1 : "Fear Message 1",
    2 : "Fear Message 2",
    3 : "Fear Message 3",
    4 : "Fear Message 4",
    5 : "Fear Message 5",
    6 : "Fear Message 6",
    7 : "Fear Message 7",
    8 : "Fear Message 8",
    9 : "Fear Message 9",
    10 : "Fear Message 10",
}

surprise_message = {
    1 : "Surprise Message 1",
    2 : "Surprise Message 2",
    3 : "Surprise Message 3",
    4 : "Surprise Message 4",
    5 : "Surprise Message 5",
    6 : "Surprise Message 6",
    7 : "Surprise Message 7",
    8 : "Surprise Message 8",
    9 : "Surprise Message 9",
    10 : "Surprise Message 10",
}

neutral_message = {
    1 : "Neutral Message 1",
    2 : "Neutral Message 2",
    3 : "Neutral Message 3",
    4 : "Neutral Message 4",
    5 : "Neutral Message 5",
    6 : "Neutral Message 6",
    7 : "Neutral Message 7",
    8 : "Neutral Message 8",
    9 : "Neutral Message 9",
    10 : "Neutral Message 10",
}

sad_message = {
    1 : "Sad Message 1",
    2 : "Sad Message 2",
    3 : "Sad Message 3",
    4 : "Sad Message 4",
    5 : "Sad Message 5",
    6 : "Sad Message 6",
    7 : "Sad Message 7",
    8 : "Sad Message 8",
    9 : "Sad Message 9",
    10 : "Sad Message 10",
}

custom = model.predict(x)
expression = list(custom[0]*100)
emotion_analysis(custom[0])

root = tkinter.Tk()
root.title("Analysis Report")
root.geometry("300x200")
var = tkinter.StringVar()
label = tkinter.Message( root, textvariable=var, relief= tkinter.RAISED, width= 300 )

# ('Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

if expression.index(max(expression)) == 0:
    var.set("Angry : " + angry_message[random.randint(1, 10)])
elif expression.index(max(expression)) == 1:
    var.set("Fear : " + fear_message[random.randint(1, 10)])
elif expression.index(max(expression)) == 2:
    var.set("Happy : " + happy_message[random.randint(1, 10)])
elif expression.index(max(expression)) == 3:
    var.set("Sad : " + sad_message[random.randint(1, 10)])
elif expression.index(max(expression)) == 4:
    var.set("Surprise : " + surprise_message[random.randint(1, 10)])
elif expression.index(max(expression)) == 5:
    var.set("Neutral : " + neutral_message[random.randint(1, 10)])

label.pack()
root.mainloop()