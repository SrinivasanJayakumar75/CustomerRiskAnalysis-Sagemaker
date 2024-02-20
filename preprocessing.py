import os
os.system('python3 -m pip install -U sagemaker')

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.experiments.run import Run, load_run

session = Session(boto3.session.Session(region_name="us-east-1"))

def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=10)
    parser.add_argument('--target_col', type=str, default='LABEL')
    parser.add_argument('--input_path', type=str, default="/opt/ml/processing/input")
    parser.add_argument('--output_path', type=str, default="/opt/ml/processing/output")
    params, _ = parser.parse_known_args()
    return params


if __name__ == "__main__":

  with load_run(sagemaker_session=session) as sm_run:

    args = read_parameters()
    print(f"Parameters read: {args}")

    sm_run.log_parameters(vars(args))

    df = pd.read_csv(os.path.join(args.input_path, "customer_data.csv"))

    df['fea_2'] = df['fea_2'].replace(np.NaN, df['fea_2'].mean())

    df_risk = df.loc[df['label'] == 1]
    df_norisk = df.loc[df['label'] == 0].sample(n=len(df_risk)*2)

    df_balanced = pd.concat([df_norisk, df_risk])

    df_balanced = df_balanced.sample(frac=1, random_state=args.random_state)

    perc_norisk = round(df_balanced['label'].value_counts()[0]/len(df_balanced) * 100,3)
    perc_risk = round(df_balanced['label'].value_counts()[1]/len(df_balanced) * 100,3)

    print(f"Balanced DataFrame, Risk: {perc_risk}% {df_balanced[df_balanced['label']==1].shape} vs Norisk: {perc_norisk}% {df_balanced[df_balanced['label']==0].shape}")

    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=args.val_size,
        random_state=args.random_state
    )

    train_dir = os.path.join(args.output_path, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(args.output_path, "validation")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(args.output_path, "test")
    os.makedirs(test_dir, exist_ok=True)


    df_train = pd.concat([y_train, X_train], axis=1)
    df_train.to_csv(os.path.join(train_dir, "train.csv"), header=False, index=False)

    df_validation = pd.concat([y_val, X_val], axis=1)
    df_validation.to_csv(os.path.join(val_dir, "validation.csv"), header=False, index=False)

    df_test = pd.concat([y_test, X_test], axis=1)
    df_test.to_csv(os.path.join(test_dir, "test.csv"), header=False, index=False)



    sm_run.log_parameters(
        {
            "train_data_size": len(X_train),
            "test_data_size": len(X_test),
            "val_data_size": len(X_val)
        }
    )

    print("Done")