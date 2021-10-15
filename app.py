import streamlit as st

import numpy as np
from sklearn.datasets import load_digits
import pandas as pd

from io import BytesIO
from PIL import Image
import base64

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

def embeddable_image(data: np.array) -> str:
    img_data = 255 - 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

digits = load_digits()
embedding = np.load('embedding.npy')

digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
digits_df['digit'] = [str(x) for x in digits.target]
digits_df['image'] = list(map(embeddable_image, digits.images))

## Generate bokeh plot
datasource = ColumnDataSource(digits_df)
color_mapping = CategoricalColorMapper(factors=[str(9 - x) for x in digits.target_names], palette=Spectral10)

plot_figure = figure(
    title='UMAP projection of the Digits dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Digit:</span>
        <span style='font-size: 18px'>@digit</span>
    </div>
</div>
"""))

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    color=dict(field='digit', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
)

st.bokeh_chart(plot_figure)

HTML_FILE = 'umap.html'
output_file(filename=HTML_FILE, title="Static HTML file")
save(plot_figure)
with open(HTML_FILE, "rb") as file:
        btn = st.download_button(
        label="Download HTML",
        data=file,
        file_name=HTML_FILE,
        mime='text/html'
    )

if st.checkbox('Show data table'):
    st.dataframe(digits_df)