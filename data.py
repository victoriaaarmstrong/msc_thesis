import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import biosppy as bsp
import time
import heartpy as hp

from sklearn import preprocessing

"""
WINDOW = 100
"""

def normalization(data):
    """
    :param data:
    :return:
    """
    data = data.values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    df = pd.DataFrame(x_scaled)

    return df


def extract_features(x, y, window):
    """
    Uses a window to extract statistical features using the Pandas DataFrame .describe method for feature extraction.
    """
    i = 0
    length = len(x)
    features = []
    labels = []

    while i < length:
        size = i + window
        temp1 = x.iloc[i:size].describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        temp2 = y.iloc[i:size].mode()

        temp1 = np.array(temp1)
        temp1 = temp1.flatten()
        temp2 = np.array(temp2)
        temp2 = temp2.flatten()

        features.append(temp1)
        labels.append(temp2)

        i += window

    generated_features = pd.DataFrame(features)
    generated_features = generated_features.fillna(0)
    label_average = pd.DataFrame(labels)

    return generated_features, label_average


def read_data_file(file_name, window_size):
    """
    Extacts data from the .pkl file, here BVP, ACC and label are extracted. Only BVP and the labels are used for training currently.
    :param file_name:
    :return: dataframe
    """

    ## Get the data as a df from the pkl file
    unpickled = pd.read_pickle(file_name)

    ## Get only the columns that we want, ignoring other data in the set
    temp1 = pd.DataFrame.from_dict(unpickled['signal']['wrist']['BVP'])
    temp1.columns = ['BVP']

    temp2 = pd.DataFrame.from_dict(unpickled['signal']['wrist']['ACC'])
    temp2.columns = ['ACC X', 'ACC Y', 'ACC Z']

    temp3 = pd.DataFrame.from_dict(unpickled['label'])
    temp3.columns = ['label']

    ## Concatenate these dfs together
    df = pd.concat([temp1, temp2, temp3], axis=1)#.reindex(temp1.index)

    ## Remove labeled data we don't care about, as per the WESAD documentation
    #df = df[~df['label'].isin(['0', '4', '5', '6', '7'])] #should work but didn't?
    df = df[df.label != 0]
    df = df[df.label != 4]
    df = df[df.label != 5]
    df = df[df.label != 6]
    df = df[df.label != 7]

    ## Removes outliers and nan values in the table that might cause issues
    id_names= df[df['BVP'] <= 0].index
    df.drop(id_names, inplace=True)
    df.dropna()

    ## Get bvp data alone
    bvp = df['BVP']
    bvp = normalization(bvp)
    bvp = bvp.fillna(0)

    ## Get labels alone
    labels = df['label']

    ## Extract info about bvp over a sliding window
    x_values, y_values = extract_features(bvp, labels, window_size)
    y_values = y_values.values.ravel()

    print(x_values.shape)
    print(y_values.shape)

    return x_values, y_values


def combine_participants():
    """
    combines all of the participant .pkl files into one to save time
    removes all 0 or nan values from sensor faults
    :return:
    """
    participants = ["S3.pkl", "S4.pkl", "S5.pkl", "S6.pkl", "S7.pkl", "S8.pkl", "S9.pkl", "S10.pkl", "S11.pkl", "S13.pkl", "S14.pkl", "S15.pkl", "S16.pkl", "S17.pkl"]

    ## Initialize dataframe with the first participant file
    print("Starting S2.pkl...")
    x, y = read_data_file("participant_data/S2.pkl")
    data = pd.concat([x, y], axis=1)

    ## Concatenate remaining participants together
    for participant in participants:
        print("Starting " + str(participant) + "...")
        temp1, temp2 = read_data_file("participant_data/"+participant)
        temp3 = pd.concat([temp1, temp2], axis=1)
        data = pd.concat([data,temp3], axis=0)

    print(data.head())
    data.to_pickle("all_data.pkl")
    print("participants concatenated")

    return



