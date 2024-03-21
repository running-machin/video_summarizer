# Issues

Right now there is an issue.
 - I need to Generate summaries for any sample. But the thing is, I am not able to do that because in the main.py file, The model is trained and then The model is saved. later, The model is used Generate the samples, especially machine summary samples. And then saved as hdf5 file. This is all done at the same time

        - So I need to somehow able to change the Code in such a way that model should be `loaded` And `generate` the machine summary samples from it.     
            
            - The next issue is that There is no code for this. There is a only definition for dataset_predict, That generates `machine_summary` for the dataset as a whole.
            -   we were abble to load the model but prediction is had
        Now, i have downloaded the TVSum and decided to predict the summary for the TVSum dataset. Then use those to test the revise -tool on them. Hence i will train the model on the TVSum dataset and then predict the summary.