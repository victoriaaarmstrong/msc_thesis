import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import preprocessing


def dataset_stats(df):
    """
    Descriptive stats of the dataset so we can see what it looks like.
    :param df:
    :return:
    """
    print("Dataset Label Spread:")
    print(df.count_values())
    print('/n')
    print("Number of Data Points")
    print(len(df))

    return


def plot_segment(df, start, end):
    df = df.iloc[start:end]
    df.plot()
    plt.show()

    return


def rebalance_dataset(features, labels):
    df = pd.concat([features, labels], axis=1)
    df = df.drop(df[df.iloc[:,-1] == 0].sample(frac=0.8).index)
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    return features, labels


def normalization(data):
    """
    Takes a dataframe and normalizes values in it
    :param data: dataframe that needs data to be normalized
    :return df: dataframe that's now normalized
    """

    ## Extract values and reshape
    data = data.values.reshape(-1, 1)

    ## Scale and trasform
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)

    ## Put back into dataframe
    df = pd.DataFrame(x_scaled)

    return df


def window(df, window_length, step):
    """

    :param df:
    :param window_length:
    :return:
    """
    ## Initialize counters
    i = 0
    df_length = len(df)
    f = df['BVP']
    l = df['label']
    x = []
    y = []

    ## Go through to get data in the window size
    while i < df_length:
        interval = i + window_length
        temp1 = f.iloc[i:interval]
        temp2 = l.iloc[i:interval].mode()

        temp1 = np.array(temp1)
        temp1 = temp1.flatten()

        temp2 = np.array(temp2)
        temp2 = temp2.flatten()

        x.append(temp1)
        y.append(temp2)

        i += step

    feature_df = pd.DataFrame(x)
    feature_df = feature_df.fillna(0) ##might be better if I just keep this as an array maybe?
    label_df = pd.DataFrame(y)

    print(feature_df.shape)
    print(label_df.shape)

    return feature_df, label_df


def read_data_file(file_name):
    """
    Extacts data from the .pkl file, here BVP, ACC and label are extracted. Only BVP and the labels are used for training currently.
    :param file_name:
    :return: dataframe with csv data
    """

    ## Get the data as a df from the pkl file
    unpickled = pd.read_pickle(file_name)

    ## Get only the columns that we want, ignoring other data in the set
    temp1 = pd.DataFrame.from_dict(unpickled['signal']['wrist']['BVP'])
    temp1.columns = ['BVP']
    temp2 = pd.DataFrame.from_dict(unpickled['label'])
    temp2.columns = ['label']

    ## Convert to a binary classification (stress, other)
    temp2 = temp2.replace({1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0})

    ## Concatenate these dfs together
    df = pd.concat([temp1, temp2], axis=1)

    ## This is the deepflow mean centering, but the mean is so small idk what the point is?
    ## Probably makes more sense to normalize?
    mu = df['BVP'].mean()
    df['BVP'] = df['BVP'] - mu
    df.fillna(mu)

    df['BVP'] = normalization(df['BVP'])

    ##should I add normalization here? would it work well with the data centering, or after it

    return df


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
        data = pd.concat([data, temp3], axis=0)

    ## Print head and save file so you don't have to go through the process many times
    print(data.head())
    data.to_pickle("all_data.pkl")
    print("participants concatenated")

    return




