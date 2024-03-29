Folder with data: https://drive.google.com/drive/folders/18j7PVTl0-DfWDNRXMmLzGaBgetMfpfJh


# Automatic drum mixing using deep learning

This is an ongoing project aims to create a drum mixing tool using deep learning. 
The tool will take a multitrack recording as input and output the suggested parameters for each track. 
    X features: some sort of difference between the spectral representation of the raw and target signal
    Y labels: parameters that have been applied to ... the said transformations
The overall architecture of the tool is as follows:
    - signal processor: loads each track separately and preprocesses it
    (applies EQ and saves the processed track along with its EQ settings as file metadata)
    The preprocessor also loads the parameters and scales them to a convenient range for the model
    - feature extractor: extracts features from the processed tracks
    - model: takes the features as input and outputs the suggested parameters


