"""
Dataset constructor
"""

"""
Imports
"""

import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'

class Dataset:
    """
    Class corresponding to the dataset and several properties related
    Attributes:
        (1) seed_int:               Seed integer,
        (2) train_fraction:         Train fraction,
        (3) name:                   Dataset name,
        (5) raw:                    Raw dataset dataframe,
        (6) binary:                 List of binary features,
        (7) categorical:            List of categorical features, 
        (8) ordinal:                List of ordinal features,
        (9) continuous:             List of continuous features,
       (10) label_name:             Label column name,
       (11) balanced:               Balanced dataset,
       (12) balanced_label:         Balanced dataset label,
       (12) train_pd:               Training dataset dataframe,
       (15) test_pd:                Testing dataset dataframe, 
       (18) train_target:           Training dataset targets,
       (19) test_target:            Testing dataset targets,
       (22) bin_encoder:            One-Hot Encoder used for binary feature preprocessing,
       (23) cat_encoder:            One-Hot Encoder used for categorical feature preprocessing,
       (24) ord_scaler:             Ordinal feature scaler,
       (25) con_scaler:             Continuous feature scaler,
       (26) processed_train_pd:     Processed training dataset,
       (27) processed_test_pd:      Processed testing dataset,
       (28) features:               Feature names in the original dataset,
       (29) processed_features:     Feature names in the processed dataset,
       (30) bin_enc_cols:           Binary feature column names,
       (31) cat_enc_cols:           Categorical feature column names,
       (32) feature_dist:           Feature distribution in the original dataset, 
       (33) processed_feat_dist:    Feature distribution in the processed dataset,
    """
    def __init__(self,dataset_name,
                 train_fraction,
                 seed) -> None:

        self.seed = seed
        self.train_fraction = train_fraction
        self.name = dataset_name
        self.raw, self.binary, self.categorical, self.ordinal, self.continuous, self.label_name = self.load_file()
        self.balanced, self.balanced_label = self.balance_data()
        self.train_pd, self.test_pd, self.train_target, self.test_target = train_test_split(self.balanced,self.balanced_label,train_size=self.train_fraction,random_state=self.seed)
        self.bin_encoder, self.cat_encoder, self.ord_scaler, self.con_scaler = self.encoder_scaler_fit()
        self.processed_train_pd, self.processed_test_pd = self.encoder_scaler_transform()
        self.features = self.train_pd.columns.to_list()
        self.bin_enc_cols = list(self.bin_encoder.get_feature_names_out(self.binary))
        self.cat_enc_cols = list(self.cat_encoder.get_feature_names_out(self.categorical))
        self.processed_features = self.bin_enc_cols + self.cat_enc_cols + self.ordinal + self.continuous
        self.feature_dist, self.processed_feat_dist = self.feature_distribution()

    def erase_missing(self,data):
        """
        Function that eliminates instances with missing values
        Output data: Filtered dataset without points with missing values
        """
        data = data.replace({'?':np.nan})
        data = data.replace({' ?':np.nan})
        if self.name == 'compass':
            for i in data.columns:
                if data[i].dtype == 'O' or data[i].dtype == 'str':
                    if len(data[i].apply(type).unique()) > 1:
                        data[i] = data[i].apply(float)
                        data.fillna(0,inplace=True)    
                    data.fillna('0',inplace=True)
                else:
                    data.fillna(0,inplace=True)
        data.dropna(axis=0,how='any',inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def nom_to_num(self,data):
        """
        Function to transform categorical features into encoded numerical values.
        Input data: The dataset to encode the categorical features.
        Output data: The dataset with categorical features encoded into numerical features.
        """
        encoder = LabelEncoder()
        if data['label'].dtypes == object or data['label'].dtypes == str:
            encoder.fit(data['label'])
            data['label'] = encoder.transform(data['label'])
        return data, encoder

    def load_file(self):
        """
        The UCI and Propublica datasets are preprocessed according to the MACE algorithm. Please, see: https://github.com/amirhk/mace.
        Function to load dataset_name file
        Output raw_df: Pandas DataFrame with raw data from read file
        Output binary: The binary columns names
        Output categorical: The categorical columns names
        Output ordinal: The ordinal columns names
        Output continuous: The continuous columns names
        Output label: Name of the column label
        """
        file_path = dataset_dir+self.name+'/'+self.name+'.csv'
        label = None
        if self.name == 'german':
            binary = ['Sex']
            categorical = []
            ordinal = []
            continuous = ['Age','Credit','LoanDuration']
            label = 'GoodCustomer (label)'
            raw_df = pd.read_csv(dataset_dir+'/german/german_raw.csv')
            raw_data = pd.DataFrame()
            raw_data['GoodCustomer (label)'] = raw_df['GoodCustomer']
            raw_data['GoodCustomer (label)'] = (raw_data['GoodCustomer (label)'] + 1) / 2
            raw_data.loc[raw_df['Gender'] == 'Male', 'Sex'] = 1
            raw_data.loc[raw_df['Gender'] == 'Female', 'Sex'] = 0
            raw_data['Age'] = raw_df['Age']
            raw_data['Credit'] = raw_df['Credit']
            raw_data['LoanDuration'] = raw_df['LoanDuration']
        elif self.name == 'compass':
            raw_data = pd.DataFrame()
            binary = ['Race','Sex','ChargeDegree']
            categorical = []
            ordinal = ['AgeGroup']
            continuous = ['PriorsCount']
            label = 'TwoYearRecid (label)'
            FEATURES_CLASSIFICATION = ['age_cat','race','sex','priors_count','c_charge_degree']  # features to be used for classification
            CONT_VARIABLES = ['priors_count']  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
            CLASS_FEATURE = 'two_year_recid'  # the decision variable
            SENSITIVE_ATTRS = ['race']
            df = pd.read_csv(dataset_dir+'/compass/compas-scores-two-years.csv')
            df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals
            # """ Data filtering and preparation """ (As seen in MACE algorithm, based on Propublica methodology. Please, see: https://github.com/amirhk/mace)
            tmp = \
                ((df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)) & \
                (df["is_recid"] != -1) & \
                (df["c_charge_degree"] != "O") & \
                (df["score_text"] != "NA") & \
                ((df["race"] == "African-American") | (df["race"] == "Caucasian"))
            df = df[tmp == True]
            df = pd.concat([df[FEATURES_CLASSIFICATION],df[CLASS_FEATURE],], axis=1)
            raw_data['TwoYearRecid (label)'] = df['two_year_recid']
            raw_data.loc[df['age_cat'] == 'Less than 25', 'AgeGroup'] = 1
            raw_data.loc[df['age_cat'] == '25 - 45', 'AgeGroup'] = 2
            raw_data.loc[df['age_cat'] == 'Greater than 45', 'AgeGroup'] = 3
            raw_data.loc[df['race'] == 'African-American', 'Race'] = 1
            raw_data.loc[df['race'] == 'Caucasian', 'Race'] = 2
            raw_data.loc[df['sex'] == 'Male', 'Sex'] = 1
            raw_data.loc[df['sex'] == 'Female', 'Sex'] = 2
            raw_data['PriorsCount'] = df['priors_count']
            raw_data.loc[df['c_charge_degree'] == 'M', 'ChargeDegree'] = 1
            raw_data.loc[df['c_charge_degree'] == 'F', 'ChargeDegree'] = 2
            raw_data = raw_data.reset_index(drop=True)
        elif self.name == 'ionosphere':
            columns = [str(i) for i in range(34)]
            columns = columns + ['label']
            raw_data = pd.read_csv(dataset_dir+'/ionosphere/ionosphere.data',names=columns)
            columns = ['2','4','5','6','26','label']
            raw_data = raw_data[columns]
            raw_data, lbl_encoder = self.nom_to_num(raw_data)
            binary = []
            categorical = []
            ordinal = []
            continuous = ['2','4','5','6','26']
        elif self.name == 'synthetic_diagonal_plane':
            raw_data = pd.read_csv(file_path)
            columns = [str(i) for i in range(raw_data.shape[1]-1)] + ['label']
            binary = ['0','1']
            categorical = []
            ordinal = []
            continuous = ['2']
            raw_data.columns = columns
            if label is None:
                label_name = raw_data.columns[-1]
            else:
                label_name = label
        else:
            raw_data = pd.read_csv(file_path)
            columns = [str(i) for i in range(raw_data.shape[1]-1)] + ['label']
            binary = []
            categorical = []
            ordinal = []
            continuous = columns[:-1]
            raw_data.columns = columns
        if label is None:
            label_name = raw_data.columns[-1]
        else:
            label_name = label
        return raw_data, binary, categorical, ordinal, continuous, label_name
    
    def balance_data(self):
        """
        Method to balance the dataset using undersampling of majority class
        """
        data_label = self.raw[self.label_name]
        label_value_counts = data_label.value_counts()
        samples_per_class = label_value_counts.min()
        balanced_df = pd.concat([self.raw[(data_label == 0).to_numpy()].sample(samples_per_class, random_state = self.seed),
        self.raw[(data_label == 1).to_numpy()].sample(samples_per_class, random_state = self.seed),]).sample(frac = 1, random_state = self.seed)
        balanced_df_label = balanced_df[self.label_name]
        del balanced_df[self.label_name]
        return balanced_df, balanced_df_label

    def encoder_scaler_fit(self):
        """
        Method to fit encoder and scaler for the dataset
        """
        bin_train_pd, cat_train_pd, ord_train_pd, con_train_pd = self.train_pd[self.binary], self.train_pd[self.categorical], self.train_pd[self.ordinal], self.train_pd[self.continuous]
        cat_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        bin_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        ord_scaler = MinMaxScaler(clip=True)
        con_scaler = MinMaxScaler(clip=True)
        bin_enc.fit(bin_train_pd)
        cat_enc.fit(cat_train_pd)
        if ord_train_pd.shape[1] > 0:
            ord_scaler.fit(ord_train_pd)
        if con_train_pd.shape[1] > 0:
            con_scaler.fit(con_train_pd)
        return bin_enc, cat_enc, ord_scaler, con_scaler

    def encoder_scaler_transform(self):
        """
        Method to fit encoder and scaler for the dataset
        """
        processed_train_pd = pd.DataFrame()
        bin_train_pd, cat_train_pd, ord_train_pd, con_train_pd = self.train_pd[self.binary], self.train_pd[self.categorical], self.train_pd[self.ordinal], self.train_pd[self.continuous]
        if bin_train_pd.shape[1] > 0:
            bin_enc_train_np, bin_enc_cols = self.bin_encoder.transform(bin_train_pd).toarray(), self.bin_encoder.get_feature_names_out(self.binary)
            bin_enc_train_pd = pd.DataFrame(bin_enc_train_np,index=bin_train_pd.index,columns=bin_enc_cols)
            processed_train_pd = pd.concat((processed_train_pd,bin_enc_train_pd),axis=1)
        if cat_train_pd.shape[1] > 0:
            cat_enc_train_np, cat_enc_cols = self.cat_encoder.transform(cat_train_pd).toarray(), self.cat_encoder.get_feature_names_out(self.categorical) 
            cat_enc_train_pd = pd.DataFrame(cat_enc_train_np,index=cat_train_pd.index,columns=cat_enc_cols)
            processed_train_pd = pd.concat((processed_train_pd,cat_enc_train_pd),axis=1)
        if ord_train_pd.shape[1] > 0:
            ord_scaled_train_np = self.ord_scaler.transform(ord_train_pd)
            ord_scaled_train_pd = pd.DataFrame(ord_scaled_train_np,index=ord_train_pd.index,columns=self.ordinal)
            processed_train_pd = pd.concat((processed_train_pd,ord_scaled_train_pd),axis=1)
        if con_train_pd.shape[1] > 0:
            con_scaled_train_np = self.con_scaler.transform(con_train_pd)
            con_scaled_train_pd = pd.DataFrame(con_scaled_train_np,index=con_train_pd.index,columns=self.continuous)
            processed_train_pd = pd.concat((processed_train_pd,con_scaled_train_pd),axis=1)
        
        processed_test_pd = pd.DataFrame()
        bin_test_pd, cat_test_pd, ord_test_pd, con_test_pd = self.test_pd[self.binary], self.test_pd[self.categorical], self.test_pd[self.ordinal], self.test_pd[self.continuous]
        if bin_test_pd.shape[1] > 0:
            bin_enc_test_np = self.bin_encoder.transform(bin_test_pd).toarray()
            bin_enc_test_pd = pd.DataFrame(bin_enc_test_np,index=bin_test_pd.index,columns=bin_enc_cols)
            processed_test_pd = pd.concat((processed_test_pd,bin_enc_test_pd),axis=1)
        if cat_test_pd.shape[1] > 0:
            cat_enc_test_np = self.cat_encoder.transform(cat_test_pd).toarray()
            cat_enc_test_pd = pd.DataFrame(cat_enc_test_np,index=cat_test_pd.index,columns=cat_enc_cols)
            processed_test_pd = pd.concat((processed_test_pd,cat_enc_test_pd),axis=1)
        if ord_test_pd.shape[1] > 0:
            ord_scaled_test_np = self.ord_scaler.transform(ord_test_pd)
            ord_scaled_test_pd = pd.DataFrame(ord_scaled_test_np,index=ord_test_pd.index,columns=self.ordinal)
            processed_test_pd = pd.concat((processed_test_pd,ord_scaled_test_pd),axis=1)
        if con_test_pd.shape[1] > 0:
            con_scaled_test_np = self.con_scaler.transform(con_test_pd)
            con_scaled_test_pd = pd.DataFrame(con_scaled_test_np,index=con_test_pd.index,columns=self.continuous)
            processed_test_pd = pd.concat((processed_test_pd,con_scaled_test_pd),axis=1)

        return processed_train_pd, processed_test_pd 
    
    def feature_distribution(self):
        """
        Method to calculate the distribution for all features
        """
        num_instances_balanced = self.balanced.shape[0]
        num_instances_processed_train = self.processed_train_pd.shape[0]
        feat_dist = {}
        processed_feat_dist = {}
        all_non_con_feat = self.binary+self.categorical+self.ordinal
        all_non_con_processed_feat = self.bin_enc_cols+self.cat_enc_cols+self.ordinal
        if len(all_non_con_feat) > 0:
            for i in all_non_con_feat:
                feat_dist[i] = ((self.balanced[i].value_counts()+1)/(num_instances_balanced+len(np.unique(self.balanced[i])))).to_dict() # +1 for laplacian counter
        if len(self.continuous) > 0:
            for i in self.continuous:
                feat_dist[i] = {'mean': self.balanced[i].mean(), 'std': self.balanced[i].std()}
                processed_feat_dist[i] = {'mean': self.processed_train_pd[i].mean(), 'std': self.processed_train_pd[i].std()}
        if len(all_non_con_processed_feat) > 0:
            for i in all_non_con_processed_feat:
                processed_feat_dist[i] = ((self.processed_train_pd[i].value_counts()+1)/(num_instances_processed_train+len(np.unique(self.processed_train_pd[i])))).to_dict() # +1 for laplacian counter
        return feat_dist, processed_feat_dist