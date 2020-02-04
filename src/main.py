import boto3
import os 
import numpy as np
boto3_connection = boto3.resource('s3')

def print_s3_contents_boto3(connection):
    for bucket in connection.buckets.all():
        for key in bucket.objects.all():
            print(key.key)

# print_s3_contents_boto3(boto3_connection)


user = os.environ['LOGNAME'] 
bucket_name = user + '-assign'
remote = 'Train/'
remote1 = 'Test/Edgar_Degas/'
def downloadDirectoryFroms3(bucketName,remoteDirectoryName):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucketName)
    i=0
    for key in bucket.objects.filter(Prefix = remoteDirectoryName):
        if i==0:
            continue
        else:
            print(key.key)
        i+=1

train = ['Edgar_Degas', 'Pablo_Picasso', 'Vincent_Van_Gogh']
os.mkdir('/Users/abeldesta531/dsi/repos/Capstone-2/data/Train')
os.mkdir('/Users/abeldesta531/dsi/repos/Capstone-2/data/Test')
test_dest = '/Users/abeldesta531/dsi/repos/Capstone-2/data/Test'
train_dest = '/Users/abeldesta531/dsi/repos/Capstone-2/data/Train'
for i in train:
    home = '/Users/abeldesta531/dsi/repos/Capstone-2/data/{0}'.format(i)
    os.chdir(home)
    os.mkdir(test_dest + / + i)
    os.mkdir(train_dest + / + i)
    files = os.listdir()
    test = np.round(len(files)*.2)
    for idx, file in enumerate(files):
        if idx < test:
            os.rename(home + '/' + file, test_dest + '/' + i)
        else:
            os.rename(home + '/' + file, train_dest + '/' + i)