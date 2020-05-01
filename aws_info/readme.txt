Account information:
Account Id: 305884190279 

Using EC2 instance on AWS to run tensorflow

Follow instructions here:
https://aws.amazon.com/getting-started/hands-on/train-deep-learning-model-aws-ec2-containers/

Ensure to terminate instance after usage to save $$


Step-by-step instructions:
1. AWS management console
2. Navigate to the EC2 console -> Launch an Amazon EC2 instance
3. Choose the AWS Marketplace tab on the left
   Search for 'deep learning base ubuntu'
   Select the AWS Deep Learning Base AMI (Ubuntu 18.04) -> Use c5.2xlarge instance type (has 8 vCPU and 16 GiB @ $0.34/hour) -> review and Launch
   Other instances: c5.large (2 vCPU and 4GiB @ $0.085/hour) ran into memeory issues
                    c5.xlarge (4 vCPU and 8GiB @ $0.17/hour) gets killed randomly
                    Maybe explore other GPU options
4. Use existing private key -> dl_keypair
5. Click on instance ID to lauch the instance management tensorflow
6. Copy Public DNS (IPv4)
7. chmod 0400 dl_keypair.pem
8. ssh -L localhost:8888:localhost:8888 -i dl_keypair.pem ubuntu@<your instance DNS> 
Sometimes the username ubuntu needs to be replace with ec2-user
9. Log in to Amazon ECR
    a. aws configure
    b. Enter there details: AWSAccessKeyId = AKIAJC23QXBROKQHN56A
        AWSSecretKey = z+tWuinSIbgd3g/HLyC7LCRMwBlPqnqI9TVqqJjA
    c. $(aws ecr get-login --region us-east-1 --no-include-email --registry-ids 763104351884)
10. Get Deep Learning Containers
    a. docker run -it 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.13-cpu-py36-ubuntu16.04
11. Install dependancies and get our github repo
    a. pip install matplotlib pandas
    b. git clone git@github.com:srimukhp/UniRep_finetune.git (SSH keys already taken care of)
12. Upload all data (weights etc.,) to github as once the instance is terminated all the data will be lost 
13. After running, ensure to go to EC2 management console and terminate the instance so that we stop getting charged
    

