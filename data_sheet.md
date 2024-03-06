# Datasheet for UCF101 Dataset

The dataset can be sourced from https://www.crcv.ucf.edu/research/data-sets/ucf101/
and https://www.crcv.ucf.edu/data/UCF101.php

UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 data set which has 50 action categories.

With 13320 videos from 101 action categories, UCF101 gives the largest diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions, etc, it is the most challenging data set to date. As most of the available action recognition data sets are not realistic and are staged by actors, UCF101 aims to encourage further research into action recognition by learning and exploring new realistic action categories.

## Motivation

Purpose: The UCF101 dataset was created for action recognition in videos, intended to support the development and evaluation of action recognition methods in realistic action scenes.

Creators and Funders: The dataset was created by the University of Central Florida's Center for Research in Computer Vision. Details about specific funding sources are not provided in the dataset's documentation, but research grants typically fund such projects.

UCF101 was created to address the challenges of action recognition in videos, such as variations in camera viewpoints, object appearances, cluttered backgrounds, and illumination changes. Realistic videos from YouTube offer a more practical scenario compared to staged datasets.


## Composition

Instance Representation: Each instance in the dataset represents a video clip showing a particular action from a predefined list of 101 action categories.

Instance Counts: There are 13,320 video clips in the dataset, typicall with a resolution of 320x240 pixels

Missing Data: There is no known missing data; all video clips are fully labeled.

Confidentiality: The dataset is composed of publicly available videos, and there is no confidential information as per the dataset's documentation.

## Collection process

Data Acquisition: Videos were sourced from YouTube and manually cut to lengths of 7-10 seconds.

Sampling Strategy: The videos were selected to represent a diverse range of actions, backgrounds, and lighting conditions. The dataset's creators aimed to ensure variety in the subjects' appearance, pose, and camera motion. The sampling strategy aimed for diversity in viewpoints (e.g., front-on, side view), camera motion (e.g., static, zooming), and object scales (close-up, far-away) in addition to actions, backgrounds, and lighting conditions.

Time Frame: The dataset compilation was not conducted over a fixed period; it was an accumulation of suitable video clips as found by the researchers.

## Preprocessing/cleaning/labelling

Preprocessing Steps: The videos were preprocessed to have a consistent frame rate and size. No other information on preprocessing is provided.

Raw Data: It is unclear if the raw, uncut videos are available alongside the dataset.

Preprocessing ensured consistent frame rate (e.g., 25 FPS) and size (320x240 pixels).
 
## Uses

Potential Tasks: Beyond action recognition, UCF101 can be used for video classification, object detection in videos, and transfer learning for video-related tasks.

Impact on Future Uses: The dataset has a bias towards the type of actions included and the demographics visible in the videos. Future users should be cautious when extending models trained on this dataset to different demographic groups or action types.

Inappropriate Uses: The dataset should not be used for applications that require the recognition of actions not represented in the dataset, nor should it be used in contexts where demographic representation is critical without additional, complementary data.

## Distribution

Current Distribution: The UCF101 dataset is publicly available and has been widely distributed within the academic community.

IP and ToU: The dataset is released under a license that allows for research use. Any commercial application should review the terms of use and ensure compliance with any copyright regulations.

You can access the dataset directly from the official website: https://www.crcv.ucf.edu/data/UCF101.php.

## Maintenance

Maintainers: The dataset is maintained by the University of Central Florida's Center for Research in Computer Vision.
