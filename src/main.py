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



train = ['Edgar_Degas', 'Pablo_Picasso', 'Vincent_Van_Gogh']
test_dest = '/Users/abeldesta531/dsi/repos/Capstone-2/data/Test'
train_dest = '/Users/abeldesta531/dsi/repos/Capstone-2/data/Train'
os.mkdir('{0}'.format(test_dest))
os.mkdir('{0}'.format(train_dest))
for i in train:
    home = '/Users/abeldesta531/dsi/repos/Capstone-2/data/{0}'.format(i)
    os.mkdir('{0}/{1}'.format(test_dest, i))
    os.mkdir('{0}/{1}'.format(train_dest, i))
    os.chdir(home)
    files = os.listdir()
    test = np.round(len(files)*.2)
    os.chdir('/Users/abeldesta531/dsi/repos/Capstone-2/data')
    for idx in np.arange(len(files)):
        if idx < test:
            os.rename(home + '/' + files[idx], '{0}/{1}/{2}'.format(test_dest, i, files[idx]))
        else:
            os.rename(home + '/' + files[idx], '{0}/{1}/{2}'.format(train_dest, i, files[idx]))
