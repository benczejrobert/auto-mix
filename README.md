Folder with data: https://drive.google.com/drive/folders/18j7PVTl0-DfWDNRXMmLzGaBgetMfpfJh 


# Automatic drum mixing using deep learning
\
This is an ongoing project aims to create a drum mixing tool using deep learning. \
The tool will take a multitrack recording as input and output the suggested parameters for each track. \
&nbsp;&nbsp;&nbsp;&nbsp;X features: some sort of difference between the spectral representation of the raw and target signal\
&nbsp;&nbsp;&nbsp;&nbsp;Y labels: estmation of the parameters that have been applied to the raw signal the said transformations \
The overall architecture of the tool is as follows:\
&nbsp;&nbsp;&nbsp;&nbsp;- signal processor: loads each track separately and preprocesses it\
&nbsp;&nbsp;&nbsp;&nbsp;(applies EQ and saves the processed track along with its EQ settings as file metadata)\
&nbsp;&nbsp;&nbsp;&nbsp;The preprocessor also loads the parameters and scales them to a convenient range for the model\
&nbsp;&nbsp;&nbsp;&nbsp;- feature extractor: extracts features from the processed tracks\
&nbsp;&nbsp;&nbsp;&nbsp;- model: takes the features as input and outputs the suggested parameters\


