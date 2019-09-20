# PXA573MscProject
MSc Robotics MSc Project

Copyright (c) 2019, Andreas Papachristodoulou, PXA573, MSc Robotics Final Project.

The source codes provided in this project are:
- my personal work;
- modified work from ST-GCN: https://github.com/yysijie/st-gcn/blob/master/OLD_README.md, ST-GCN 

The files stored in:
- The PXA573 folder are my personal work.
- The UseInST-GCN folder are the modified files. ST-GCN was used as the baseline for the deep learning framework 
and integration, the files we altered so that we have extra features are the ones stored in the UseInST-GCN.
Simply replace their main.py with ours, and add the pxa573_rec.py stored in the USEInST-GCN.

*NOTE: The input MoCap files are in c3d format.*

# Data Pre-processing
If you want to pre-process data from MoCap use MainPreprocessor.py in PXA573/preprocessing:
  - First, change the directories to match your files and select frate -> division of frames ;Run the program, 
  and follow the instructions and questions printed by the console; Answer in lower format. 
  - This file maps the joints to NTU, adjusts the dimensions to fit the NTU structure and allows you to use automatic or 
manual selection of the actions from a long video that includes multiple.

# Data Post-Processing
If you want to add distortions to data from MoCap use files in PXA573/postprocessing:
  - First, change the directories to match your files;Run the programs, and follow the instructions and questions printed
  by the console; Answer in lower format. 
  - 'CorrDistortion' file is the correct file of distortion. The initial upload is a different wrong version with bugs. Using 
  this file occlusion and random noise distortions as wel as the combination of the two are added to the skeletons.
  - 'AltSpeed' changes the speed of files from 100 Hz by selecting every 'frate' frame. To select speed, simply adjust frate.
  - 'Comb' file can be used to combine the different files created into a single dataset that can be used for train and test.
  Using the NTUcomb you can also add NTU data samples. (i.e, Distortions + Pure data + NTU). It also includes the 
  functionalities of the 'Split' file.
  - 'SelectData' file is used to select data which are predicted correctly by the models. Firstly, the dataset used for the
  selection must be ran by Pxa573_rec code as explained later, with batch number as 1, so the top1 array is saved. Using this
  array the 'optimal' data are selected.
  - 'Split' file provides the feature of splitting a dataset randmoly to train and test datasets.
  - 'Stats' file provides statistics using the top1 array, for general and for each action class.
  - 'Combine' and 'Altdata' are for the old format of data and were used for the conversion from CMU files. Distortion is the
a different wrong version with bags. They are not useful!
If you want to turn your data back to c3d format use the homonymous file in backtoc3d from PXA573/Misc. In addition, you can
visualise your skeletons with the files from the same folder.

*REMINDER* - In the codes provided you need to change the directories required.

Our Benchmarks and data used for training as well as videos and a presentation of our project can be found:
- download data: https://drive.google.com/open?id=1YNgApnmZqXGQNHL1WLXyhZRSbzr-aq1V

If you simply want to run our benchmarks using the ST-GCN:

- Clone the https://github.com/yysijie/st-gcn/blob/master/OLD_README.md project, ST-GCN and follow their instructions
- assign the appropriate directories and weights in the /config/st_gcn/xx/test.yaml files
- run the command 'python main.py pxa573rec -c config/st_gcn/< fill >/test.yaml'
* In the empty to fill space write ntu-xview or ntu-xsub with regard to which model you want to test
 
Feel free to use the code and data, if you have any queries contact me at: Papachrist@outlook.com
